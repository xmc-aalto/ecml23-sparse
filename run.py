import time
import json
import hashlib
from copy import copy
from pathlib import Path
import argparse

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras import Model, Sequential

from data import load_mixed_dataset, load_metadata
from sparse import SparseAdam, RewireLayer, StratifiedRewireLayer, SparseOptWrapper, PrecisionAtK, \
    SparseSquaredHingeLoss, SparsePrecision, SparseRecall, StratifiedRewireLayerV2, FixedFractionSelector, SparseBinaryCrossEntropyLoss, RewireCallback


def make_identifier(config):
    mode = config['architecture']

    identifier = hashlib.sha256(json.dumps(config).encode()).hexdigest()
    identifier = f"{config['realization']}-{identifier}"

    if mode == "dense":
        return f"dense-{identifier}"
    elif mode == "sparse":
        connectivity = config['connectivity']
        return f"sparse-{connectivity}-{identifier}"
    else:
        connectivity = config['connectivity']
        intermediate = config['intermediate']
        return f"{mode}-{connectivity}-{intermediate}-{identifier}"


def standardize_config(config: dict):
    mode = config['architecture']
    if mode == "dense":
        config['connectivity'] = 0
        config['intermediate'] = 0
        config['rewire_interval'] = 0
        config['rewire_fraction'] = 0
    elif mode == "sparse":
        config['intermediate'] = 0


def get_layers(mode: str, num_labels: int, intermediate=None, connectivity=None, dropout=None, rewire_fraction=None) -> list:
    layers = []
    if dropout is not None:
        layers.append(kl.Dropout(dropout))

    if mode == "dense":
        layers.append(kl.Dense(num_labels, trainable=True))
    elif mode == "sparse":
        layers.append(RewireLayer(num_labels, connections=connectivity * num_labels))
    elif mode == "dense-sparse":
        layers.append(kl.Dense(intermediate, trainable=True, activation="relu"))
        layers.append(RewireLayer(num_labels, connections=connectivity * num_labels,
                                  rewire_selector=FixedFractionSelector(rewire_fraction)))
    elif mode == "dense-stratified-old":
        layers.append(kl.Dense(intermediate, trainable=True, activation="relu"))
        layers.append(StratifiedRewireLayer(num_labels, connections=connectivity * num_labels,
                                            rewire_selector=FixedFractionSelector(rewire_fraction)))
    elif mode == "dense-stratified":
        layers.append(kl.Dense(intermediate, trainable=True, activation="relu"))
        layers.append(StratifiedRewireLayerV2(num_labels, connections=connectivity * num_labels,
                                              rewire_selector=FixedFractionSelector(rewire_fraction)))
    elif mode == "stratified":
        layers.append(StratifiedRewireLayerV2(num_labels, connections=connectivity * num_labels,
                                              rewire_selector=FixedFractionSelector(rewire_fraction)))
    elif mode == "bottleneck":
        layers.append(kl.Dense(intermediate, trainable=True, activation="relu"))
        layers.append(kl.Dense(num_labels, trainable=True))
    return layers


def make_loss_fn(identifier: str):
    if identifier == "squared_hinge":
        return SparseSquaredHingeLoss()
    elif identifier == "bce":
        return SparseBinaryCrossEntropyLoss()
    raise NotImplementedError()


def run_training(data_base_dir, config):
    feature_src = config["features"]
    data_src = config["dataset"]
    data_dir = data_base_dir / data_src
    meta = load_metadata(data_dir / "train-labels.txt")
    train = load_mixed_dataset(data_dir / "train-labels.txt", data_dir / f"train-{feature_src}.npy",
                               sparse_labels=True).shuffle(
        50000).batch(32).prefetch(4)
    test = load_mixed_dataset(data_dir / "test-labels.txt", data_dir / f"test-{feature_src}.npy",
                              sparse_labels=True).batch(32).prefetch(4)

    num_labels: int = meta["labels"]
    standardize_config(config)
    connectivity = config['connectivity']
    intermediate = config['intermediate']
    dropout = config['dropout']
    mode = config['architecture']
    learning_rate = config['learning_rate']
    rewire_interval = config['rewire_interval']
    rewire_fraction = config['rewire_fraction']
    loss = config['loss_fn']
    realization = config['realization']
    run_id = make_identifier(config)
    epochs = 100

    inner_model = Sequential(get_layers(mode, num_labels=num_labels,
                                        connectivity=connectivity, intermediate=intermediate,
                                        dropout=dropout, rewire_fraction=rewire_fraction))
    model = SparseOptWrapper(rewire_interval=rewire_interval, base_model=inner_model)

    print(f"Mode = {mode}")
    print(f"Connectivity = {connectivity}")
    print(f"Intermediate = {intermediate}")
    print(f"LR = {learning_rate}")
    print(f"Dropout = {dropout}")
    print(f"Rewire int = {rewire_interval}")
    print(f"Rewire % = {100 * rewire_fraction}")
    print(f"Loss = {loss}")
    print(f"Dataset = {data_src}")
    print(f"Features = {feature_src}")

    model.compile(loss=make_loss_fn(loss),
                  optimizer=SparseAdam(learning_rate, update_moments=True),
                  metrics=[
                            SparsePrecision(threshold=0.0),
                            SparseRecall(threshold=0.0),
                            PrecisionAtK(k=1, name="p@1"),
                            PrecisionAtK(k=3, name="p@3"),
                            PrecisionAtK(k=5, name="p@5")
    ])
    model.evaluate(test.take(1))
    inner_model.summary()

    start = time.time()

    callback_list = [
        callbacks.ReduceLROnPlateau(verbose=1, monitor="val_p@3", mode="max", patience=2, factor=0.5,
                                    min_delta=1e-3, min_lr=1e-4),
        callbacks.EarlyStopping(monitor="val_p@3", mode="max", restore_best_weights=True, verbose=1, patience=4,
                                min_delta=5e-5),
        callbacks.TensorBoard(log_dir=f"logs/{data_src}-{feature_src}/{realization}-{run_id}", profile_batch=(1001, 1011))
    ]
    if rewire_interval == -1:
        callback_list.append(RewireCallback())

    history = model.fit(train,validation_data=test, epochs=epochs, callbacks=callback_list, verbose=2)

    end = time.time()

    # gather results
    train_results = model.evaluate(train, verbose=2)
    test_results = model.evaluate(test, verbose=2)
    memory_info = tf.config.experimental.get_memory_info("GPU:0")['peak']

    keys = ["loss", "precision", "recall", "p@1", "p@3", "p@5"]

    history = history.history

    result = {"train": dict(zip(keys, train_results)), "test": dict(zip(keys, test_results)),
              "memory": memory_info, "epochs": len(history['loss']),
              "time": end - start, "config": config}

    return result


def build_task_list(task_file: str):
    task_file = json.loads(Path(task_file).read_text())
    default = task_file["defaults"]  # type: dict
    raw_task_list = task_file["tasks"]
    tasks = []  # type: list[dict]

    # expand repeated tasks
    for task in raw_task_list:
        new_task = copy(default)
        new_task.update(task)
        repeats = new_task["repeat"]
        del new_task["repeat"]
        for k in range(repeats):
            tasks.append({**new_task, "realization": k})

    for t in tasks:
        standardize_config(t)

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("result_dir")
    parser.add_argument("--tasks", default="tasks.json")
    parser.add_argument("--index", default=-1, type=int)
    result = parser.parse_args()
    data_dir = Path(result.data_dir)
    result_path = Path(result.result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    index = result.index

    print(f"Loading tasks from {result.tasks}")
    tasks = build_task_list(result.tasks)

    def make_result_path(task):
        run_id = make_identifier(task)
        base_path = result_path / f'{task["dataset"]}-{task["features"]}'
        check_path = base_path / f"{run_id}.json"
        base_path.mkdir(parents=True, exist_ok=True)
        return check_path

    if index == -1:
        for i, task in enumerate(tasks):
            if not make_result_path(task).exists():
                index = i
                break

    print(f"Selected task: {index}")
    print(json.dumps(tasks[index], indent=2))
    task = tasks[index]
    if make_result_path(task).exists():
        exit(1)

    result = run_training(data_dir, task)
    make_result_path(task).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

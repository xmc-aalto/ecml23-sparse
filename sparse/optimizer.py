import tensorflow as tf
from tensorflow import keras
import typing

if typing.TYPE_CHECKING:
    from .layer import RewireLayerBase


class SparseAdam(keras.optimizers.Adam):
    def __init__(self, *args, update_moments=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_moments = update_moments

    def update_sparse(self, layer: "RewireLayerBase"):
        if self.update_moments:
            var_key = self._var_key(layer.sparse_weights)
            m = self._momentums[self._index_dict[var_key]]
            v = self._velocities[self._index_dict[var_key]]
            new_m, new_v = layer.rewire([m, v])
            m.assign(new_m)
            v.assign(new_v)
        else:
            layer.rewire(None)


class SparseOptWrapper(tf.keras.Model):
    def __init__(self, *args, rewire_interval: int, base_model: tf.keras.Model, inputs=None, outputs=None, **kwargs):
        # make sure we build using the sub-network
        assert inputs is None
        assert outputs is None

        super().__init__(*args, **kwargs)
        self.rewire_interval = rewire_interval
        self.base_model = base_model

    def call(self, *args, **kwargs):
        return self.base_model.call(*args, **kwargs)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        if self.optimizer is not None:
            assert hasattr(self.optimizer, "update_sparse")

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.handle_sparse_layers()

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def handle_sparse_layers(self):
        iterations = self.optimizer.iterations
        if self.rewire_interval > 0 and iterations % self.rewire_interval == 1:
            for layer in self._flatten_layers():
                if hasattr(layer, "sparse_weights"):
                    self.optimizer.update_sparse(layer)


class RewireCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch != 0:
            for layer in self.model._flatten_layers():
                if hasattr(layer, "sparse_weights"):
                    self.model.optimizer.update_sparse(layer)

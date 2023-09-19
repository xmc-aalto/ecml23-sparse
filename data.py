from typing import Union, Dict

from pathlib import Path
import logging
import tensorflow as tf


_logger = logging.getLogger(__name__)


def path_arg(f):
    # TODO make this a more general utility function
    import functools

    @functools.wraps(f)
    def wrap(path: Union[str, Path], *args, **kwargs):
        if isinstance(path, str):
            path = Path(path)
        return f(path, *args, **kwargs)
    return wrap


@path_arg
def load_metadata(path: Union[Path, str]) -> Dict[str, int]:
    with path.open() as fd:
        total_pts, num_ftr, num_lbl = fd.readline().split(" ")
    return {"examples": int(total_pts), "labels": int(num_lbl), "features": int(num_ftr)}


@path_arg
def load_xmlrep_dataset(path: Union[Path, str], num_parallel_calls=4, sparse_labels: bool = True):
    """
    Loads datasets found on http://manikvarma.org/downloads/XC/XMLRepository.html. In particular, the data file
    is expected to have the following format:
    Header Line: Total_Points Num_Features Num_Labels
    1 line per datapoint : label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val

    The resulting dataset consists of `SparseTensor`s.
    :param path: Path to the text file containing the dataset.
    or as a `tf.Tensor`.
    :param num_parallel_calls: Number of parallel calls for example parsing.
    :return: A `tf.data.Dataset` that produces `(features, labels)` tuples.
    """
    import tensorflow as tf
    with path.open() as fd:
        total_pts, num_ftr, num_lbl = fd.readline().split(" ")

    text = tf.data.TextLineDataset(str(path)).skip(1)  # load text file and skip header

    def process(x):
        features, labels = process_line(x, num_features=int(num_ftr))
        labels = tf.SparseTensor(tf.cast(labels, tf.int64)[:, None], tf.ones_like(labels, dtype=tf.float32),
                                 dense_shape=(int(num_lbl),))
        if not sparse_labels:
            labels = tf.sparse.to_dense(labels)
        return features, labels

    # the repeat(01).take() construct is just so that the dataset gets a well-defined cardinality
    return text.map(process, num_parallel_calls=num_parallel_calls).repeat(-1).take(int(total_pts))


def split_labels(labels_str: tf.Tensor):
    """
    Takes a string tensor of comma-separated labels and returns a tensor of integers.
    :param labels_str: Comma-separated label string. Either a scalar, or a batch of strings.
    :return: Integer tensor of label ids
    """
    label_split = tf.strings.split(labels_str, ",", name='split-labels')

    def cvt(string_digits: tf.Tensor):
        # one special case: no labels at all. In this case, we get one entry which is the empty string
        return tf.cond(tf.equal(string_digits[0], ""),
                       lambda: tf.constant([], dtype=tf.int32),
                       lambda: tf.strings.to_number(string_digits, tf.int32))

    return tf.ragged.map_flat_values(cvt, label_split)


@tf.function
def process_line(line: tf.Tensor, num_features: int):
    """
    Processes a single line of the dataset, i.e. a single example
    :param line: An example in the format label1,label2,... ftr1:val1 ftr2:val2 ...
    :param num_features: The total number of features, to set the correct shape of the feature tensor.
    :return: features (as Sparse Tensor of shape (num_feature)) and labels
    """
    data = tf.strings.split(line, " ")
    labels = split_labels(data[0])
    features = tf.strings.split(data[1:], ":", maxsplit=1)

    indices = tf.strings.to_number(features[:, 0:1].flat_values, tf.int64)
    values = tf.strings.to_number(features[:, 1:2].flat_values, tf.float32)

    features = tf.sparse.SparseTensor(indices[:, None], values, dense_shape=(num_features,))
    return features, labels


@tf.function
def process_line_labels(line: tf.Tensor, num_features: int):
    """
    Processes a single line of the dataset, i.e. a single example
    :param line: An example in the format label1,label2,... ftr1:val1 ftr2:val2 ...
    :param num_features: The total number of features, to set the correct shape of the feature tensor.
    :return: features (as Sparse Tensor of shape (num_feature)) and labels
    """
    data = tf.strings.split(line, " ")
    labels = split_labels(data[0])
    return labels


def load_mixed_dataset(tfidf: Union[Path, str], dense: Union[Path, str], num_parallel_calls=4, sparse_labels: bool = True):
    """
    Loads dense features from a npy file, but sparse labels from a text file.
    """
    import tensorflow as tf
    import numpy as np
    tfidf = Path(tfidf)
    dense = Path(dense)

    with tfidf.open() as fd:
        total_pts, num_ftr, num_lbl = map(int, fd.readline().split(" "))

    text = tf.data.TextLineDataset(str(tfidf)).skip(1)  # load text file and skip header

    def process(x):
        labels = process_line_labels(x, num_features=int(num_ftr))
        labels = tf.SparseTensor(tf.cast(labels, tf.int64)[:, None], tf.ones_like(labels, dtype=tf.float32),
                                 dense_shape=(int(num_lbl),))
        if not sparse_labels:
            labels = tf.sparse.to_dense(labels)
        return labels

    npy = np.load(dense)
    assert npy.shape[0] == total_pts, f"Mismatching number of points in text {total_pts} and npy {npy.shape[0]}"
    features = tf.data.Dataset.from_tensor_slices(npy)

    # the repeat(01).take() construct is just so that the dataset gets a well-defined cardinality
    return tf.data.Dataset.zip((features, text.map(process, num_parallel_calls=num_parallel_calls))).repeat(-1).take(int(total_pts))

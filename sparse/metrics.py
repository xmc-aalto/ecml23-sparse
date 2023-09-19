import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.losses import Loss

from sparse.utils import sparse_dense_cwise_op, dense_sparse_cwise_op


def topk_as_matrix(data, k):
    """
    Calculates the top_k among each entry in `data`, and returns a matrix with values in
    `{0, 1}`, which indicates for each entry whether it is in the topk.
    :param data: Data to rank.
    :param k: Number of top indices to consider.
    :return: A SparseTensor of the same (dense) shape as data.
    """

    with tf.name_scope("topk_as_matrix"):
        _, tops = tf.nn.top_k(data, k=k)        # tops: [BATCH x K]
        batch_size = tf.cast(tf.shape(data)[0], tf.int32, name="batch_size")
        batch_ids = tf.range(0, batch_size, dtype=tf.int32)
        batch_ids = tf.tile(batch_ids[:, None], (1, k), name="batch_ids")
        idx = tf.stack([batch_ids, tops], axis=-1)  # [B, K, 2]
        idx = tf.reshape(idx, (-1, 2), name="indices")

        result = tf.sparse.SparseTensor(tf.cast(idx, tf.int64), tf.ones(batch_size * k, dtype=tf.float32),
                                        tf.shape(data, out_type=tf.int64))
        return tf.sparse.reorder(result)


class PrecisionAtK(Metric):
    """
    Calculates Precision@K, i.e. which fraction of top-k predictions are
    among the true labels. Given a score vector $y$, a threshold is chosen
    such that only $k$ labels are predicted as true, and the corresponding
    precision is calculated. This means that if the dataset contains examples
    with less than $k$ labels, no score function can reach a P@k of 1.
    """
    def __init__(self, k=1, name=None, **kwargs):
        """

        :param k: Number of highest ranking predictions to consider.
        :param name: A name for this metric. If `None` is given, defaults to `P_at_{k}`.
        """
        name = name or "P_at_{}".format(k)
        super().__init__(name=name, **kwargs)
        self._k = k
        self._correct = self.add_weight("NumCorrect", (), initializer=Zeros())       # type: tf.Variable
        self._total = self.add_weight("NumTotal", (), initializer=Zeros())           # type: tf.Variable

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v.value()))

    def update_state(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is None
        with tf.name_scope("precision_at_k"):
            top_indices = topk_as_matrix(y_pred, self._k)
            top_indices_expanded = tf.sparse.add(top_indices, y_true.with_values(tf.zeros_like(y_true.values)))  # top_indices.with_values(-np.inf)
            y_true_expanded = tf.sparse.add(top_indices.with_values(tf.zeros_like(top_indices.values)), y_true)  # top_indices.with_values(-np.inf)

            is_correct = y_true_expanded.with_values(top_indices_expanded.values * y_true_expanded.values)
            total_correct = tf.sparse.reduce_sum(is_correct, axis=1)
            num_correct = tf.reduce_sum(total_correct, name="num_correct")

            self._correct.assign_add(num_correct)
            self._total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32, name="batch_size") * self._k)

    def result(self):
        return tf.math.divide_no_nan(self._correct.value(), self._total.value(), name="precision_at_k_result")


class SparsePrecision(Metric):
    """
    Calculates Precision, expecting the labels to be supplied as a `SparseTensor`
    """
    def __init__(self, threshold, name=None, **kwargs):
        name = name or "Precision"
        super().__init__(name=name, **kwargs)
        self._threshold = threshold
        self._correct = self.add_weight("NumCorrect", (), initializer=Zeros())       # type: tf.Variable
        self._total = self.add_weight("NumTotal", (), initializer=Zeros())           # type: tf.Variable

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v.value()))

    def update_state(self, y_true, y_pred, sample_weight=None):
        with tf.name_scope("sparse_precision"):
            assert sample_weight is None
            is_predicted = tf.greater(y_pred, self._threshold)
            is_correct = sparse_dense_cwise_op(y_true, is_predicted, lambda x, y: tf.multiply(x, tf.cast(y, tf.float32)))

            self._correct.assign_add(tf.sparse.reduce_sum(is_correct))
            self._total.assign_add(tf.cast(tf.math.count_nonzero(is_predicted), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self._correct.value(), self._total.value(), name="sparse_precision_result")


class SparseRecall(Metric):
    """
    Calculates Precision, expecting the labels to be supplied as a `SparseTensor`
    """
    def __init__(self, threshold, name=None, **kwargs):
        name = name or "Recall"
        super().__init__(name=name, **kwargs)
        self._threshold = threshold
        self._correct = self.add_weight("NumCorrect", (), initializer=Zeros())       # type: tf.Variable
        self._total = self.add_weight("NumTotal", (), initializer=Zeros())           # type: tf.Variable

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v.value()))

    def _is_correctly_predicted(self, y_true, scores):
        is_predicted = tf.cast(tf.greater(scores, self._threshold), tf.float32)
        return tf.multiply(y_true, is_predicted)

    def update_state(self, y_true, y_pred, sample_weight=None):
        with tf.name_scope("sparse_recall"):
            assert sample_weight is None
            is_correct = sparse_dense_cwise_op(y_true, y_pred, self._is_correctly_predicted)

            self._correct.assign_add(tf.sparse.reduce_sum(is_correct))
            self._total.assign_add(tf.sparse.reduce_sum(y_true))

    def result(self):
        return tf.math.divide_no_nan(self._correct.value(), self._total.value(), name="sparse_recall_result")


class SparseLossBase(Loss):
    def __init__(self, name):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        scores = tf.convert_to_tensor(y_pred)
        hinge_error = dense_sparse_cwise_op(scores, y_true, self._cwise_loss)
        return tf.reduce_mean(hinge_error, axis=-1)

    @classmethod
    def _cwise_loss(cls, score, y):
        margin = (2 * y - 1) * score
        return tf.nn.relu(1.0 - margin)


class SparseHingeLoss(SparseLossBase):
    def __init__(self, name="sparse_hinge_loss"):
        super().__init__(name=name)

    @classmethod
    def _cwise_loss(cls, score, y):
        margin = (2 * y - 1) * score
        return tf.nn.relu(1.0 - margin)


class SparseSquaredHingeLoss(SparseLossBase):
    def __init__(self, name="sparse_hinge_loss"):
        super().__init__(name=name)

    @classmethod
    def _cwise_loss(cls, score, y):
        margin = (2 * y - 1) * score
        return tf.square(tf.nn.relu(1.0 - margin))


class SparseBinaryCrossEntropyLoss(SparseLossBase):
    def __init__(self, name="sparse_bce_loss"):
        super().__init__(name=name)

    @classmethod
    def _cwise_loss(cls, score, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=score)

import tensorflow as tf
from .layer import RewireLayerBase
from .ops.fixed_fan_in_ops import fan_in_sparse_matmul, fan_in_sparse_matmul_tp


class StratifiedRewireLayerV2(RewireLayerBase):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        # int64 is much faster
        self.index_dtype = tf.uint32

    def _init_connections(self, dim):
        # generate one incoming and one outgoing index for each connection randomly, and put these together in an
        # index variable
        return tf.cast(tf.random.uniform(shape=(self.connections,), minval=0, maxval=dim, dtype=tf.int32), self.index_dtype)

    def _get_sparse_kernel(self):
        return None

    def _update_sparse_kernel(self, indices, values, locations, others):
        self.kernel_indices.assign(indices)
        self.kernel_weights.assign(values)

        outputs = []
        for other in others:
            zero = tf.zeros(tf.shape(locations)[0], other.dtype)
            new_v = tf.tensor_scatter_nd_update(other, locations, zero)
            outputs.append(new_v)

        return outputs

    def _rewire(self, locations, other_vars):
        indices = self.kernel_indices.value()
        num_changes = tf.shape(locations)[0]

        # generate random sources for the new connections, and determine their location in the index tensor
        # all sources have index (ConnectionID, 0)
        new_sources = tf.random.uniform(shape=(num_changes,), minval=0, maxval=self.input_dim, dtype=tf.int32)
        new_sources = tf.cast(new_sources, self.index_dtype)

        # perform the scatter update, and re-generate the sparse tensor
        new_indices = tf.tensor_scatter_nd_update(indices, locations, new_sources)
        return self._update_sparse_kernel(new_indices, self.kernel_weights.value(), locations, other_vars)

    def _sparse_matmul(self, inputs):
        indices = tf.reshape(self.kernel_indices.value(), (-1, self.connections // self.units))
        weights = tf.reshape(self.kernel_weights.value(), (-1, self.connections // self.units))
        tp_in = tf.transpose(inputs)
        output = fan_in_sparse_matmul_tp(tp_in, indices, weights)
        return output

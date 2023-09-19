from abc import abstractmethod
from typing import Optional

import tensorflow as tf
from tensorflow import TensorShape
from tensorflow.keras import layers, initializers, regularizers, constraints


def _ensure_positive_integer(value, name, allow_none=False):
    if value is None and allow_none:
        return None

    value_as_int = int(value)
    if value_as_int < 0 or value != value_as_int:
        raise ValueError(f'Received an invalid value for `{name}`, expected '
                         f'a positive integer, got {value}.')
    return value_as_int


def _ensure_limit(value, name: str, minimum, maximum, allow_none=False):
    if value is None and allow_none:
        return None

    if minimum <= value <= maximum:
        return value

    raise ValueError(f"Received an invalid value {value} for `{name}`, expected {minimum} <=  {name} <= {maximum}")


class RewireSelector(tf.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def select(self, weight_values: tf.Tensor) -> tf.Tensor:
        """
        Given a tensor of weight values, returns a tensor of booleans,
        which should contain true for all the weights that shall be
        rewired.
        """
        raise NotImplementedError()

    def build(self, weight_values: tf.Tensor):
        pass


class FixedMagnitudeSelector(RewireSelector):
    """
    Rewire all weights below a given, fixed magnitude.
    """
    def __init__(self, threshold):
        super().__init__()
        self.threshold = tf.Variable(threshold, trainable=False, name="threshold")

    def build(self, weight_values: tf.Tensor):
        pass

    def select(self, weight_values):
        return tf.less(tf.abs(weight_values), self.threshold)


class FixedFractionSelector(RewireSelector):
    """
    Rewire a fixed fraction of all connections on each update. Always
    the connections with the lowest weight are chosen.
    """
    def __init__(self, fraction):
        super().__init__()
        fraction = _ensure_limit(fraction, "fraction", 0.0, 1.0)
        self.fraction = tf.Variable(fraction, trainable=False, name="threshold")

    def build(self, weight_values: tf.Tensor):
        pass

    def select(self, weight_values):
        # determine how many connections to prune
        num_weights = tf.size(weight_values)
        num_selected = tf.cast(tf.math.round(self.fraction * tf.cast(num_weights, tf.float32)), tf.int32)

        # find the threshold
        flat_weights = tf.abs(tf.reshape(weight_values, (-1,)))
        val, ind = tf.nn.top_k(-flat_weights, k=num_selected)
        threshold = -val[-1]

        # and select the elements
        return tf.less(flat_weights, threshold)


class SignFlipSelector(RewireSelector):
    def __init__(self):
        super().__init__()

    def select(self, weight_values):
        # determine how many connections to prune
        num_weights = tf.size(weight_values)
        num_selected = tf.cast(tf.math.round(self.fraction * tf.cast(num_weights, tf.float32)), tf.int32)

        # find the threshold
        flat_weights = tf.abs(tf.reshape(weight_values, (-1,)))
        val, ind = tf.nn.top_k(-flat_weights, k=num_selected)
        threshold = -val[-1]

        # and select the elements
        return tf.less(flat_weights, threshold)


class RewireLayerBase(layers.Layer):
    def __init__(self, units, *, connections: Optional[int] = None, sparsity: Optional[float] = None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_bias=True,
                 rewire_selector=None,
                 **kwargs):
        super(RewireLayerBase, self).__init__(**kwargs)

        self.units = _ensure_positive_integer(units, "units")
        self._connections = _ensure_positive_integer(connections, "connections", allow_none=True)
        self._sparsity = _ensure_limit(sparsity, "sparsity", 0.0, 1.0, allow_none=True)
        if sparsity is not None and connections is not None:
            raise ValueError("The `sparsity` and `connections` arguments are mutually exclusive")

        self.input_dim = None

        self.input_spec = layers.InputSpec(min_ndim=2)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel_weights = None  # type: tf.Variable
        self.kernel_indices = None
        self._step_counter = None
        self.bias = None
        self.use_bias = use_bias

        if rewire_selector is None:
            rewire_selector = FixedFractionSelector(0.01)
        self.rewire_selector = rewire_selector

    @property
    def connections(self):
        return self._connections

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def sparse_kernel(self):
        return self._get_sparse_kernel()

    @property
    def sparse_weights(self):
        return self.kernel_weights

    def build(self, input_shape):
        input_shape = TensorShape(input_shape)
        last_dim = input_shape[-1]
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `SparseOutLayer` '
                             'should be defined. Found `None`.')

        self.input_dim = last_dim

        # figure out the number of required connections
        if self.connections is None:
            self._connections = int((self.units * last_dim) * (1.0 - self.sparsity))
        else:
            self._sparsity = 1.0 - self.connections / (self.units * last_dim)

        self.input_spec = layers.InputSpec(min_ndim=2, axes={-1: last_dim})

        # build the weights for the kernel, and assign initial values. There is one weight for each connection
        self.kernel_weights = self.add_weight(
            'kernel_weights',
            shape=[self.connections],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )

        connections = self._init_connections(last_dim)
        self.kernel_indices = tf.Variable(initial_value=connections, trainable=False, name="kernel_indices")

        # actually build the kernel. This will sort the indices
        self._update_sparse_kernel(connections, self.kernel_weights.value(), None, [])

        # add a bias weight if requested
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)

        self.built = True

        # we need a step counter in order to track when to do rewiring.
        self._step_counter = self.add_weight(name="step_count", shape=(), dtype=tf.int64, trainable=False,
                                             initializer=initializers.Zeros())  # type: tf.Variable

    def rewire(self, other_vars: list = None):
        if other_vars is None:
            other_vars = []
        locations = self._get_rewire_candidates()
        return self._rewire(locations, other_vars)

    def _get_sparse_kernel(self):
        raise NotImplementedError()

    def _update_sparse_kernel(self, indices, values, locations=None, others=None) -> list:
        raise NotImplementedError()

    def _rewire(self, locations, other_vars):
        raise NotImplementedError()

    def _init_connections(self, dim):
        raise NotImplementedError()

    def _sparse_matmul(self, inputs):
        raise NotImplementedError()

    def _get_rewire_candidates(self):
        return tf.where(self.rewire_selector.select(self.kernel_weights.value()))

    def call(self, inputs, training=None):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        rank = inputs.shape.rank
        assert rank == 2 or rank is None, rank
        outputs = self._sparse_matmul(inputs)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1] is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % (input_shape,))
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
            'connections': self.connections,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config


class RewireLayer(RewireLayerBase):
    def __init__(self, *args, **kwargs):
        super(RewireLayer, self).__init__(*args, **kwargs)

    def _init_connections(self, dim):
        # generate one incoming and one outgoing index for each connection randomly, and put these together in an
        # index variable
        connections_in = tf.random.uniform(shape=(self.connections,), minval=0, maxval=dim, dtype=tf.int64)
        connections_out = tf.random.uniform(shape=(self.connections,), minval=0, maxval=self.units, dtype=tf.int64)
        connections = tf.stack([connections_in, connections_out], -1)
        return connections

    def _update_sparse_kernel(self, indices, values, locations, others):
        sparse_tensor = tf.sparse.SparseTensor(indices, values, (self.input_dim, self.units))
        outputs = []
        for other in others:
            zero = tf.zeros(tf.shape(locations)[0], other.dtype)
            new_v = tf.tensor_scatter_nd_update(other, locations, zero)
            sparse_v = tf.sparse.SparseTensor(indices, new_v, (self.input_dim, self.units))
            reordered = tf.sparse.reorder(sparse_v)
            outputs.append(reordered.values)

        reordered = tf.sparse.reorder(sparse_tensor)

        self.kernel_indices.assign(reordered.indices)
        self.kernel_weights.assign(reordered.values)

        return outputs

    def _get_sparse_kernel(self):
        return tf.sparse.SparseTensor(self.kernel_indices.value(), self.kernel_weights.value(),
                                      (self.input_dim, self.units))

    def _rewire(self, locations, other_vars):
        indices = self.kernel_indices.value()
        num_changes = tf.shape(locations)[0]

        # generate random sources for the new connections, and determine their location in the index tensor
        # all sources have index (ConnectionID, 0)
        new_sources = tf.random.uniform(shape=(num_changes,), minval=0, maxval=self.input_dim, dtype=tf.int64)
        src_idx = tf.zeros((num_changes, 1), dtype=tf.int64)
        src_loc = tf.concat([locations, src_idx], axis=-1)

        # generate random targets for the new connections, and determine their location in the index tensor
        # all targets have index (ConnectionID, 1)
        new_targets = tf.random.uniform(shape=(num_changes,), minval=0, maxval=self.units, dtype=tf.int64)
        tgt_idx = tf.ones((num_changes, 1), dtype=tf.int64)
        tgt_loc = tf.concat([locations, tgt_idx], axis=-1)

        # generate one big list of updates
        all_locations = tf.concat([src_loc, tgt_loc], axis=0)
        all_updates = tf.concat([new_sources, new_targets], axis=0)

        # perform the scatter update, and re-generate the sparse tensor
        new_indices = tf.tensor_scatter_nd_update(indices, all_locations, all_updates)
        return self._update_sparse_kernel(new_indices, self.kernel_weights.value(), locations, other_vars)

    def _sparse_matmul(self, inputs):
        sparse_kernel = self._get_sparse_kernel()
        outputs = tf.sparse.sparse_dense_matmul(sparse_kernel, inputs, adjoint_a=True, adjoint_b=True)
        return tf.transpose(outputs)


class StratifiedRewireLayer(RewireLayerBase):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        # int64 is much faster
        self.index_dtype = tf.int64

    def _init_connections(self, dim):
        # generate one incoming and one outgoing index for each connection randomly, and put these together in an
        # index variable
        return tf.random.uniform(shape=(self.connections,), minval=0, maxval=dim, dtype=self.index_dtype)

    def _build_full_indices(self, in_indices):
        connections_out = tf.tile(tf.range(0, self.units, dtype=tf.int64)[:, None], (1, self.connections // self.units))
        connections_out = tf.reshape(connections_out, shape=(self.connections,))
        return tf.stack([connections_out, tf.cast(in_indices, tf.int64)], -1)

    def _get_sparse_kernel(self):
        return tf.sparse.SparseTensor(self._build_full_indices(self.kernel_indices.value()), self.kernel_weights.value(),
                                      (self.units, self.input_dim))

    def _update_sparse_kernel(self, indices, values, locations, others):
        indices = self._build_full_indices(indices)
        sparse_tensor = tf.sparse.SparseTensor(indices, values, (self.units, self.input_dim))
        reordered = tf.sparse.reorder(sparse_tensor)
        self.kernel_indices.assign(tf.cast(reordered.indices[:, 1], self.index_dtype))
        self.kernel_weights.assign(reordered.values)

        outputs = []
        for other in others:
            zero = tf.zeros(tf.shape(locations)[0], other.dtype)
            new_v = tf.tensor_scatter_nd_update(other, locations, zero)
            sparse_v = tf.sparse.SparseTensor(indices, new_v, (self.units, self.input_dim))
            reordered = tf.sparse.reorder(sparse_v)
            outputs.append(reordered.values)

        return outputs

    def _rewire(self, locations, other_vars):
        indices = self.kernel_indices.value()
        num_changes = tf.shape(locations)[0]

        # generate random sources for the new connections, and determine their location in the index tensor
        # all sources have index (ConnectionID, 0)
        new_sources = tf.random.uniform(shape=(num_changes,), minval=0, maxval=self.input_dim, dtype=self.index_dtype)

        # perform the scatter update, and re-generate the sparse tensor
        new_indices = tf.tensor_scatter_nd_update(indices, locations, new_sources)
        return self._update_sparse_kernel(new_indices, self.kernel_weights.value(), locations, other_vars)

    def _sparse_matmul(self, inputs):
        sparse_kernel = self._get_sparse_kernel()
        outputs = tf.sparse.sparse_dense_matmul(sparse_kernel, inputs, adjoint_a=False, adjoint_b=True)
        return tf.transpose(outputs)

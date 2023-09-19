import tensorflow as tf


def sparse_dense_cwise_op(a: tf.sparse.SparseTensor, b: tf.Tensor, op: callable):
    with tf.name_scope("sparse_dense_cwise_op"):
        tf.debugging.assert_equal(tf.shape(a), tf.shape(b))
        gathered = tf.gather_nd(b, a.indices)
        return a.with_values(op(a.values, gathered))


def dense_sparse_cwise_op(b: tf.Tensor, a: tf.sparse.SparseTensor, op: callable):
    with tf.name_scope("dense_sparse_cwise_op"):
        tf.debugging.assert_equal(tf.shape(a), tf.shape(b))
        gathered = tf.gather_nd(b, a.indices)
        sparse_part = op(gathered, a.values)
        dense_part = op(b, tf.zeros_like(b))
        return tf.tensor_scatter_nd_update(dense_part, a.indices, sparse_part)

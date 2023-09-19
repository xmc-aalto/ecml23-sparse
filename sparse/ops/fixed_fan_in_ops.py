"""Use fixed_fan_in ops in python."""

from tensorflow.python.framework import load_library, ops
from tensorflow.python.platform import resource_loader

fan_in_sparse_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('libsparseops.so'))

_fan_in_sparse_matmul_forward = fan_in_sparse_ops.fan_in_sparse_matmul
_fan_in_sparse_matmul_forward_tp = fan_in_sparse_ops.fan_in_sparse_matmul_tp
_fan_in_sparse_matmul_backward = fan_in_sparse_ops.fan_in_sparse_matmul_grad
_fan_in_sparse_matmul_backward_tp = fan_in_sparse_ops.fan_in_sparse_matmul_grad_tp


def fan_in_sparse_matmul(features, lookup, weights):
    return _fan_in_sparse_matmul_forward(features, lookup, weights)


def fan_in_sparse_matmul_tp(features, lookup, weights):
    return _fan_in_sparse_matmul_forward_tp(features, lookup, weights)


def fan_in_sparse_matmul_grad(features, lookup, weights, out_grad):
    return _fan_in_sparse_matmul_backward(features, lookup, weights, out_grad)


def fan_in_sparse_matmul_grad_tp(features, lookup, weights, out_grad):
    return _fan_in_sparse_matmul_backward_tp(features, lookup, weights, out_grad)


@ops.RegisterGradient("FanInSparseMatmul")
def _fan_in_sparse_matmul_grad(op: ops.Operation, grad: ops.Tensor):
    feature_grad, weight_grad = fan_in_sparse_matmul_grad(op.inputs[0], op.inputs[1], op.inputs[2], grad)
    return [feature_grad, None, weight_grad]


@ops.RegisterGradient("FanInSparseMatmulTp")
def _fan_in_sparse_matmul_grad(op: ops.Operation, grad: ops.Tensor):
    feature_grad, weight_grad = fan_in_sparse_matmul_grad_tp(op.inputs[0], op.inputs[1], op.inputs[2], grad)
    return [feature_grad, None, weight_grad]

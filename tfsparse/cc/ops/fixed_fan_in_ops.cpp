//
// Created by erik on 29.3.2022.
//

#ifndef TFSPARSE__REGISTER_H
#define TFSPARSE__REGISTER_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

template<bool Transposed>
Status fan_in_sparse_matmul_shape(shape_inference::InferenceContext* c) {
    using namespace shape_inference;
    ShapeHandle feature_shape;
    ShapeHandle lookup_shape;
    ShapeHandle weights_shape;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &feature_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &lookup_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &weights_shape));
    TF_RETURN_IF_ERROR(c->Merge(lookup_shape, weights_shape, &lookup_shape));

    ShapeHandle out_shape = c->Matrix(Transposed ? c->Dim(feature_shape, 1) : c->Dim(feature_shape, 0),
                                      c->Dim(weights_shape, 0));
    c->set_output(0, out_shape);
    return OkStatus();
}

template<bool Transposed>
Status fan_in_sparse_matmul_grad_shape(shape_inference::InferenceContext* c) {
    using namespace shape_inference;
    ShapeHandle feature_shape;
    ShapeHandle lookup_shape;
    ShapeHandle weights_shape;
    ShapeHandle output_shape;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &feature_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &lookup_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &weights_shape));
    // TODO re-introduce error here and make sure it is caught by tests
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &output_shape));
    TF_RETURN_IF_ERROR(c->Merge(lookup_shape, weights_shape, &lookup_shape));
    ShapeHandle expected_out_shape = c->Matrix(Transposed ? c->Dim(feature_shape, 1) : c->Dim(feature_shape, 0),
                                               c->Dim(weights_shape, 0));
    TF_RETURN_IF_ERROR(c->Merge(output_shape, expected_out_shape, &output_shape));

    c->set_output(0, feature_shape);
    c->set_output(1, weights_shape);
    return OkStatus();
}

REGISTER_OP("FanInSparseMatmul")
.Attr("T: type")
.Attr("I: type")
.Input("features: T")
.Input("lookup: I")
.Input("weights: T")
.Output("result: T")
.SetShapeFn(fan_in_sparse_matmul_shape<false>);

REGISTER_OP("FanInSparseMatmulTp")
.Attr("T: type")
.Attr("I: type")
.Input("features: T")
.Input("lookup: I")
.Input("weights: T")
.Output("result: T")
.SetShapeFn(fan_in_sparse_matmul_shape<true>);

REGISTER_OP("FanInSparseMatmulGrad")
.Attr("T: type")
.Attr("I: type")
.Input("features: T")
.Input("lookup: I")
.Input("weights: T")
.Input("output: T")
.Output("feature_grad: T")
.Output("weight_grad: T")
.SetShapeFn(fan_in_sparse_matmul_grad_shape<false>);

REGISTER_OP("FanInSparseMatmulGradTp")
.Attr("T: type")
.Attr("I: type")
.Input("features: T")
.Input("lookup: I")
.Input("weights: T")
.Input("output: T")
.Output("feature_grad: T")
.Output("weight_grad: T")
.SetShapeFn(fan_in_sparse_matmul_grad_shape<true>);

#endif //TFSPARSE__REGISTER_H

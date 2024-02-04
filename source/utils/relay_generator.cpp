#include "relay_generator.h"

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

namespace tvm_cpp {
namespace relay_generator {

Status generate_dead_code_module(tvm::IRModule& module) {
    // var gen function
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Var expression not found");
    }

    // constant gen function
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Constant not found");
    }

    // get the Conv2D relay function
    const tvm::runtime::PackedFunc* conv2d = tvm::runtime::Registry::Get("relay.op.nn._make.conv2d");
    if (!conv2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.conv2d expression not found");
    }

    // get the bias add relay function
    const tvm::runtime::PackedFunc* bias_add = tvm::runtime::Registry::Get("relay.op.nn._make.bias_add");
    if (!bias_add) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.bias_add expression not found");
    }

    // get the relu relay function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.relu expression not found");
    }

    // get the fold const relay function
    const tvm::runtime::PackedFunc* fold = tvm::runtime::Registry::Get("relay._transform.FoldConstantExpr");
    if (!fold) {
        return Status(StatusCode::RUNTIME_ERROR, "relay._transform.FoldConstantExpr expression not found");
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Function expression not found");
    }

    // the input shape and data type
    tvm::relay::TensorType input_type{{1, 3, 7, 7}, tvm::DataType::Float(32)};
    // generate the IRModule input
    tvm::relay::Expr input_expr = (*var_gen)("input", input_type, tvm::relay::Span());

    // generate the convolution weights
    tvm::runtime::NDArray weights = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8, 3, 3, 3}),
                                                                 tvm::DataType::Float(32), {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8 * 3 * 3 * 3; ++i) {
        static_cast<float*>(weights->data)[i] = 1.0f;
    }

    tvm::relay::Expr weights_expr = (*const_gen)(weights, tvm::relay::Span());

    // generate the convolution bias
    tvm::runtime::NDArray bias = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8}), tvm::DataType::Float(32),
                                                              {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8; ++i) {
        static_cast<float*>(bias->data)[i] = 1.0f;
    }
    tvm::relay::Expr bias_expr = (*const_gen)(bias, tvm::relay::Span());

    // the conv2d attributes
    tvm::runtime::Array<tvm::relay::IndexExpr> strides_exp({1, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> padding_exp({0, 0, 0, 0});
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation_exp({1, 1});
    int64_t group = 1;
    tvm::relay::IndexExpr channels(8);
    tvm::runtime::Array<tvm::relay::IndexExpr> kernel_size({3, 3});
    tvm::runtime::String data_layout = "NCHW";
    tvm::runtime::String kernel_layout = "OIHW";
    tvm::runtime::String out_layout = "";
    tvm::DataType out_dtype;

    // generate the conv2d expr
    tvm::relay::Expr conv_out_expr =
        (*conv2d)(input_expr, weights_expr, strides_exp, padding_exp, dilation_exp, group, channels, kernel_size,
                  data_layout, kernel_layout, out_layout, out_dtype);

    // add bias
    int axis = 1;
    tvm::relay::Expr conv_bias_out_expr = (*bias_add)(conv_out_expr, bias_expr, axis);

    // the convolution 2d expression
    tvm::relay::Expr conv_expr = (*fold)(conv_bias_out_expr, tvm::IRModule(), false);

    // the relu expression
    tvm::relay::Expr result_expr = (*relu)(conv_expr);
    result_expr = (*fold)(result_expr, tvm::IRModule(), false);

    // the graph inputs and outputs
    tvm::runtime::Array<tvm::relay::Expr> graph_input({input_expr});
    tvm::relay::Expr graph_output = result_expr;

    // the entire graph
    tvm::relay::Expr func =
        (*function)(graph_input, graph_output, tvm::relay::Type(), tvm::runtime::Array<tvm::relay::TypeVar>(),
                    tvm::DictAttrs(), tvm::relay::Span());
    module = tvm::IRModule::FromExpr(func);

    return Status::ok();
}

Status generate_fuse_op_module(tvm::IRModule& module) {
    // var gen function
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Var expression not found");
    }

    // constant gen function
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Constant not found");
    }

    // get the Conv2D relay function
    const tvm::runtime::PackedFunc* conv2d = tvm::runtime::Registry::Get("relay.op.nn._make.conv2d");
    if (!conv2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.conv2d expression not found");
    }

    // get the bias add relay function
    const tvm::runtime::PackedFunc* bias_add = tvm::runtime::Registry::Get("relay.op.nn._make.bias_add");
    if (!bias_add) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.bias_add expression not found");
    }

    // get the relu relay function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.relu expression not found");
    }

    // get the fold const relay function
    const tvm::runtime::PackedFunc* fold = tvm::runtime::Registry::Get("relay._transform.FoldConstantExpr");
    if (!fold) {
        return Status(StatusCode::RUNTIME_ERROR, "relay._transform.FoldConstantExpr expression not found");
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Function expression not found");
    }

    // the input shape and data type
    tvm::relay::TensorType input_type{{1, 3, 7, 7}, tvm::DataType::Float(32)};
    // generate the IRModule input
    tvm::relay::Expr input_expr = (*var_gen)("input", input_type, tvm::relay::Span());

    // generate the convolution weights
    tvm::runtime::NDArray weights = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8, 3, 3, 3}),
                                                                 tvm::DataType::Float(32), {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8 * 3 * 3 * 3; ++i) {
        static_cast<float*>(weights->data)[i] = 1.0f;
    }

    tvm::relay::Expr weights_expr = (*const_gen)(weights, tvm::relay::Span());

    // generate the convolution bias
    tvm::runtime::NDArray bias = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8}), tvm::DataType::Float(32),
                                                              {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8; ++i) {
        static_cast<float*>(bias->data)[i] = 1.0f;
    }
    tvm::relay::Expr bias_expr = (*const_gen)(bias, tvm::relay::Span());

    // the conv2d attributes
    tvm::runtime::Array<tvm::relay::IndexExpr> strides_exp({1, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> padding_exp({0, 0, 0, 0});
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation_exp({1, 1});
    int64_t group = 1;
    tvm::relay::IndexExpr channels(8);
    tvm::runtime::Array<tvm::relay::IndexExpr> kernel_size({3, 3});
    tvm::runtime::String data_layout = "NCHW";
    tvm::runtime::String kernel_layout = "OIHW";
    tvm::runtime::String out_layout = "";
    tvm::DataType out_dtype;

    // generate the conv2d expr
    tvm::relay::Expr conv_out_expr =
        (*conv2d)(input_expr, weights_expr, strides_exp, padding_exp, dilation_exp, group, channels, kernel_size,
                  data_layout, kernel_layout, out_layout, out_dtype);

    // add bias
    int axis = 1;
    tvm::relay::Expr conv_bias_out_expr = (*bias_add)(conv_out_expr, bias_expr, axis);

    // the convolution 2d expression
    tvm::relay::Expr conv_expr = (*fold)(conv_bias_out_expr, tvm::IRModule(), false);

    // the relu expression
    tvm::relay::Expr result_expr = (*relu)(conv_expr);
    result_expr = (*fold)(result_expr, tvm::IRModule(), false);

    // the graph inputs and outputs
    tvm::runtime::Array<tvm::relay::Expr> graph_input({input_expr});
    tvm::relay::Expr graph_output = result_expr;

    // the entire graph
    tvm::relay::Expr func =
        (*function)(graph_input, graph_output, tvm::relay::Type(), tvm::runtime::Array<tvm::relay::TypeVar>(),
                    tvm::DictAttrs(), tvm::relay::Span());
    module = tvm::IRModule::FromExpr(func);

    return Status::ok();
}

Status generate_common_subexp_module(tvm::IRModule& module) {
    // var gen function
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Var expression not found");
    }

    // constant gen function
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Constant not found");
    }

    // get the Conv2D relay function
    const tvm::runtime::PackedFunc* conv2d = tvm::runtime::Registry::Get("relay.op.nn._make.conv2d");
    if (!conv2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.conv2d expression not found");
    }

    // get the bias add relay function
    const tvm::runtime::PackedFunc* bias_add = tvm::runtime::Registry::Get("relay.op.nn._make.bias_add");
    if (!bias_add) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.bias_add expression not found");
    }

    // get the relu relay function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.relu expression not found");
    }

    // get the fold const relay function
    const tvm::runtime::PackedFunc* fold = tvm::runtime::Registry::Get("relay._transform.FoldConstantExpr");
    if (!fold) {
        return Status(StatusCode::RUNTIME_ERROR, "relay._transform.FoldConstantExpr expression not found");
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Function expression not found");
    }

    // the input shape and data type
    tvm::relay::TensorType input_type{{1, 3, 7, 7}, tvm::DataType::Float(32)};
    // generate the IRModule input
    tvm::relay::Expr input_expr = (*var_gen)("input", input_type, tvm::relay::Span());

    // generate the convolution weights
    tvm::runtime::NDArray weights = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8, 3, 3, 3}),
                                                                 tvm::DataType::Float(32), {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8 * 3 * 3 * 3; ++i) {
        static_cast<float*>(weights->data)[i] = 1.0f;
    }

    tvm::relay::Expr weights_expr = (*const_gen)(weights, tvm::relay::Span());

    // generate the convolution bias
    tvm::runtime::NDArray bias = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8}), tvm::DataType::Float(32),
                                                              {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8; ++i) {
        static_cast<float*>(bias->data)[i] = 1.0f;
    }
    tvm::relay::Expr bias_expr = (*const_gen)(bias, tvm::relay::Span());

    // the conv2d attributes
    tvm::runtime::Array<tvm::relay::IndexExpr> strides_exp({1, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> padding_exp({0, 0, 0, 0});
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation_exp({1, 1});
    int64_t group = 1;
    tvm::relay::IndexExpr channels(8);
    tvm::runtime::Array<tvm::relay::IndexExpr> kernel_size({3, 3});
    tvm::runtime::String data_layout = "NCHW";
    tvm::runtime::String kernel_layout = "OIHW";
    tvm::runtime::String out_layout = "";
    tvm::DataType out_dtype;

    // generate the conv2d expr
    tvm::relay::Expr conv_out_expr =
        (*conv2d)(input_expr, weights_expr, strides_exp, padding_exp, dilation_exp, group, channels, kernel_size,
                  data_layout, kernel_layout, out_layout, out_dtype);

    // add bias
    int axis = 1;
    tvm::relay::Expr conv_bias_out_expr = (*bias_add)(conv_out_expr, bias_expr, axis);

    // the convolution 2d expression
    tvm::relay::Expr conv_expr = (*fold)(conv_bias_out_expr, tvm::IRModule(), false);

    // the relu expression
    tvm::relay::Expr result_expr = (*relu)(conv_expr);
    result_expr = (*fold)(result_expr, tvm::IRModule(), false);

    // the graph inputs and outputs
    tvm::runtime::Array<tvm::relay::Expr> graph_input({input_expr});
    tvm::relay::Expr graph_output = result_expr;

    // the entire graph
    tvm::relay::Expr func =
        (*function)(graph_input, graph_output, tvm::relay::Type(), tvm::runtime::Array<tvm::relay::TypeVar>(),
                    tvm::DictAttrs(), tvm::relay::Span());
    module = tvm::IRModule::FromExpr(func);

    return Status::ok();
}

Status generate_remove_unused_fun_module(tvm::IRModule& module) {
    // var gen function
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Var expression not found");
    }

    // constant gen function
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Constant not found");
    }

    // get the Conv2D relay function
    const tvm::runtime::PackedFunc* conv2d = tvm::runtime::Registry::Get("relay.op.nn._make.conv2d");
    if (!conv2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.conv2d expression not found");
    }

    // get the bias add relay function
    const tvm::runtime::PackedFunc* bias_add = tvm::runtime::Registry::Get("relay.op.nn._make.bias_add");
    if (!bias_add) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.bias_add expression not found");
    }

    // get the relu relay function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.relu expression not found");
    }

    // get the fold const relay function
    const tvm::runtime::PackedFunc* fold = tvm::runtime::Registry::Get("relay._transform.FoldConstantExpr");
    if (!fold) {
        return Status(StatusCode::RUNTIME_ERROR, "relay._transform.FoldConstantExpr expression not found");
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Function expression not found");
    }

    // the input shape and data type
    tvm::relay::TensorType input_type{{1, 3, 7, 7}, tvm::DataType::Float(32)};
    // generate the IRModule input
    tvm::relay::Expr input_expr = (*var_gen)("input", input_type, tvm::relay::Span());

    // generate the convolution weights
    tvm::runtime::NDArray weights = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8, 3, 3, 3}),
                                                                 tvm::DataType::Float(32), {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8 * 3 * 3 * 3; ++i) {
        static_cast<float*>(weights->data)[i] = 1.0f;
    }

    tvm::relay::Expr weights_expr = (*const_gen)(weights, tvm::relay::Span());

    // generate the convolution bias
    tvm::runtime::NDArray bias = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8}), tvm::DataType::Float(32),
                                                              {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8; ++i) {
        static_cast<float*>(bias->data)[i] = 1.0f;
    }
    tvm::relay::Expr bias_expr = (*const_gen)(bias, tvm::relay::Span());

    // the conv2d attributes
    tvm::runtime::Array<tvm::relay::IndexExpr> strides_exp({1, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> padding_exp({0, 0, 0, 0});
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation_exp({1, 1});
    int64_t group = 1;
    tvm::relay::IndexExpr channels(8);
    tvm::runtime::Array<tvm::relay::IndexExpr> kernel_size({3, 3});
    tvm::runtime::String data_layout = "NCHW";
    tvm::runtime::String kernel_layout = "OIHW";
    tvm::runtime::String out_layout = "";
    tvm::DataType out_dtype;

    // generate the conv2d expr
    tvm::relay::Expr conv_out_expr =
        (*conv2d)(input_expr, weights_expr, strides_exp, padding_exp, dilation_exp, group, channels, kernel_size,
                  data_layout, kernel_layout, out_layout, out_dtype);

    // add bias
    int axis = 1;
    tvm::relay::Expr conv_bias_out_expr = (*bias_add)(conv_out_expr, bias_expr, axis);

    // the convolution 2d expression
    tvm::relay::Expr conv_expr = (*fold)(conv_bias_out_expr, tvm::IRModule(), false);

    // the relu expression
    tvm::relay::Expr result_expr = (*relu)(conv_expr);
    result_expr = (*fold)(result_expr, tvm::IRModule(), false);

    // the graph inputs and outputs
    tvm::runtime::Array<tvm::relay::Expr> graph_input({input_expr});
    tvm::relay::Expr graph_output = result_expr;

    // the entire graph
    tvm::relay::Expr func =
        (*function)(graph_input, graph_output, tvm::relay::Type(), tvm::runtime::Array<tvm::relay::TypeVar>(),
                    tvm::DictAttrs(), tvm::relay::Span());
    module = tvm::IRModule::FromExpr(func);

    // add an unused function
    tvm::relay::Expr unused_input_expr = (*var_gen)("input_unused", input_type, tvm::relay::Span());
    // the unused function inputs and outputs
    tvm::runtime::Array<tvm::relay::Expr> unused_input({unused_input_expr});
    tvm::relay::Expr unused_output = (*relu)(unused_input_expr);

    tvm::BaseFunc unused_func =
        (*function)(unused_input, unused_output, tvm::relay::Type(), tvm::runtime::Array<tvm::relay::TypeVar>(),
                    tvm::DictAttrs(), tvm::relay::Span());
    module->Add(tvm::GlobalVar("unused_func_1"), unused_func);
    return Status::ok();
}

Status generate_traverse_expr(tvm::relay::Expr& result){
    // var gen function
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Var expression not found");
    }

    // constant gen function
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Constant not found");
    }

    // get the Conv2D relay function
    const tvm::runtime::PackedFunc* conv2d = tvm::runtime::Registry::Get("relay.op.nn._make.conv2d");
    if (!conv2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.conv2d expression not found");
    }

    // get the bias add relay function
    const tvm::runtime::PackedFunc* bias_add = tvm::runtime::Registry::Get("relay.op.nn._make.bias_add");
    if (!bias_add) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.bias_add expression not found");
    }

    // get the relu relay function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.relu expression not found");
    }

    // get the fold const relay function
    const tvm::runtime::PackedFunc* fold = tvm::runtime::Registry::Get("relay._transform.FoldConstantExpr");
    if (!fold) {
        return Status(StatusCode::RUNTIME_ERROR, "relay._transform.FoldConstantExpr expression not found");
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Function expression not found");
    }

    // the input shape and data type
    tvm::relay::TensorType input_type{{1, 3, 7, 7}, tvm::DataType::Float(32)};
    // generate the IRModule input
    tvm::relay::Expr input_expr = (*var_gen)("input", input_type, tvm::relay::Span());

    // generate the convolution weights
    tvm::runtime::NDArray weights = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8, 3, 3, 3}),
                                                                 tvm::DataType::Float(32), {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8 * 3 * 3 * 3; ++i) {
        static_cast<float*>(weights->data)[i] = 1.0f;
    }

    tvm::relay::Expr weights_expr = (*const_gen)(weights, tvm::relay::Span());

    // generate the convolution bias
    tvm::runtime::NDArray bias = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple({8}), tvm::DataType::Float(32),
                                                              {DLDeviceType::kDLCPU, 0});
    for (int i = 0; i < 8; ++i) {
        static_cast<float*>(bias->data)[i] = 1.0f;
    }
    tvm::relay::Expr bias_expr = (*const_gen)(bias, tvm::relay::Span());

    // the conv2d attributes
    tvm::runtime::Array<tvm::relay::IndexExpr> strides_exp({1, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> padding_exp({0, 0, 0, 0});
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation_exp({1, 1});
    int64_t group = 1;
    tvm::relay::IndexExpr channels(8);
    tvm::runtime::Array<tvm::relay::IndexExpr> kernel_size({3, 3});
    tvm::runtime::String data_layout = "NCHW";
    tvm::runtime::String kernel_layout = "OIHW";
    tvm::runtime::String out_layout = "";
    tvm::DataType out_dtype;

    // generate the conv2d expr
    tvm::relay::Expr conv_out_expr =
        (*conv2d)(input_expr, weights_expr, strides_exp, padding_exp, dilation_exp, group, channels, kernel_size,
                  data_layout, kernel_layout, out_layout, out_dtype);

    // add bias
    int axis = 1;
    tvm::relay::Expr conv_bias_out_expr = (*bias_add)(conv_out_expr, bias_expr, axis);

    // the convolution 2d expression
    tvm::relay::Expr conv_expr = (*fold)(conv_bias_out_expr, tvm::IRModule(), false);

    // the relu expression
    tvm::relay::Expr result_expr = (*relu)(conv_expr);
    result = (*fold)(result_expr, tvm::IRModule(), false);

    return Status::ok();
}

}    // namespace relay_generator
}    // namespace tvm_cpp

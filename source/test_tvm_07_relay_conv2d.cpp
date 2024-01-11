#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // Step 1. Get the conv2d packed function
    const tvm::runtime::PackedFunc* conv2d = tvm::runtime::Registry::Get("relay.op.nn._make.conv2d");
    if (!conv2d) {
        std::cerr << "relay.op.nn._make.conv2d expression not found" << std::endl;
        return -1;
    }

    // Step 2. Generate the input data
    const tvm::runtime::PackedFunc* input_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!input_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    std::string input_name = "input_data";
    tvm::relay::Span input_span;

    // the input data type
    DLDataType input_data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the input tensor type, float32 and shape is {1, 32, 320, 320}
    tvm::relay::TensorType input_tensor_type{{1, 32, 320, 320}, tvm::DataType{input_data_type}};

    // the input var expression
    tvm::relay::Var input_var = (*input_gen)(input_name, input_tensor_type, input_span);

    // Step 3. Generate the conv2d weight
    // Constant in relay
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        std::cerr << "relay.ir.Constant expression not found" << std::endl;
        return -1;
    }

    // shape, data type, device
    tvm::runtime::NDArray weight_data =
        tvm::runtime::NDArray::Empty({64, 32, 3, 3}, {DLDataTypeCode::kDLFloat, 32, 1}, {DLDeviceType::kDLCPU, 0});
    // initialize the weight data
    for(int i = 0; i < 64 * 32 * 3 * 3; ++i){
        static_cast<float*>(weight_data->data)[i] = 1.0f;
    }

    tvm::relay::Span weight_span;
    // the weight constant expression
    tvm::relay::Constant weight_expr = (*const_gen)(weight_data, weight_span);

    // Step 4. do the convolution
    tvm::runtime::Array<tvm::PrimExpr> strides({2, 2});
    tvm::runtime::Array<tvm::PrimExpr> padding({1, 1, 1, 1});
    tvm::runtime::Array<tvm::PrimExpr> dilation({1, 1});
    int groups = 1;
    tvm::relay::IndexExpr channels{64};
    tvm::runtime::Array<tvm::PrimExpr> kernel_size({3, 3});
    tvm::runtime::String data_layout = "NCHW";
    tvm::runtime::String kernel_layout = "OIHW";
    tvm::runtime::String out_layout = "";
    tvm::DataType out_dtype;

    tvm::relay::Expr out_expr = (*conv2d)(input_var, weight_expr, strides, padding, dilation, groups, channels, kernel_size, data_layout, kernel_layout, out_layout, out_dtype);

    // Step 5. Add bias
    const tvm::runtime::PackedFunc* bias_add = tvm::runtime::Registry::Get("relay.op.nn._make.bias_add");
    if (!bias_add) {
        std::cerr << "relay.op.nn._make.bias_add expression not found" << std::endl;
        return -1;
    }

    // shape, data type, device
    tvm::runtime::NDArray bias_data =
        tvm::runtime::NDArray::Empty({64}, {DLDataTypeCode::kDLFloat, 32, 1}, {DLDeviceType::kDLCPU, 0});
    // initialize the weight data
    for(int i = 0; i < 64; ++i){
        static_cast<float*>(bias_data->data)[i] = 1.0f;
    }

    tvm::relay::Span bias_span;
    // the weight constant expression
    tvm::relay::Constant bias_expr = (*const_gen)(bias_data, bias_span);

    // add the bias
    tvm::relay::Expr result_expr = (*bias_add)(out_expr, bias_expr, 1);


    // Step 6. Get the output information
    // type infer
    const tvm::runtime::PackedFunc* type_infer = tvm::runtime::Registry::Get("relay._transform.InferType");
    if (!type_infer) {
        std::cerr << "relay._transform.InferType expression not found" << std::endl;
        return -1;
    }

    // pass run
    const tvm::runtime::PackedFunc* pass_run = tvm::runtime::Registry::Get("transform.RunPass");
    if (!pass_run) {
        std::cerr << "transform.pass_run expression not found" << std::endl;
        return -1;
    }

    tvm::IRModule mod = IRModule::FromExpr(result_expr);
    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    IRModule mod_new = (*pass_run)(infer_type_pass, mod);

    tvm::relay::Expr expr = mod_new->Lookup("main").as<FunctionNode>()->body;
    tvm::Type result_type = expr->checked_type();
    tvm::runtime::Optional<tvm::TensorType> type = result_type.as<tvm::TensorType>();
    if (type != nullptr) {
        tvm::TensorType tensor_type = type.value();

        const tvm::DataType& data_type = tensor_type->dtype;
        const tvm::runtime::Array<tvm::PrimExpr>& shape = tensor_type->shape;

        std::cout << "Conv2d Output DataType: " << data_type << ", Shape: ";
        std::cout << "{";
        for (int i = 0; i < shape.size(); ++i) {
            const tvm::PrimExpr& exp = shape[i];
            const IntImmNode* node = exp.as<IntImmNode>();
            if (i) {
                std::cout << ", ";
            }
            if (node) {
                std::cout << node->value;
            }
        }
        std::cout << "}" << std::endl;
    }

    return 0;
}
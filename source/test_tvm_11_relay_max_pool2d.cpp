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
    // Step 1. Get the max pool 2d packed function
    const tvm::runtime::PackedFunc* max_pool_2d = tvm::runtime::Registry::Get("relay.op.nn._make.max_pool2d");
    if (!max_pool_2d) {
        std::cerr << "relay.op.nn._make.max_pool2d expression not found" << std::endl;
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
    // the input tensor type, float32 and shape is {1, 256, 20, 20}
    tvm::relay::TensorType input_tensor_type{{1, 256, 20, 20}, tvm::DataType{input_data_type}};

    // the input var expression
    tvm::relay::Var input_var = (*input_gen)(input_name, input_tensor_type, input_span);

    // Step 3. do the 2d max pooling operation
    tvm::runtime::Array<tvm::relay::IndexExpr> pool_size({5, 5});
    tvm::runtime::Array<tvm::relay::IndexExpr> strides({1, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation({1, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> padding({2, 2, 2, 2});

    tvm::runtime::String layout = "NCHW";
    tvm::runtime::String out_layout = "";
    bool ceil_model = false;

    tvm::relay::Expr out_expr =
        (*max_pool_2d)(input_var, pool_size, strides, dilation, padding, layout, out_layout, ceil_model);

    // Step 4. Get the output information
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

    tvm::IRModule mod = IRModule::FromExpr(out_expr);
    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    IRModule mod_new = (*pass_run)(infer_type_pass, mod);

    tvm::relay::Expr expr = mod_new->Lookup("main").as<FunctionNode>()->body;
    tvm::Type result_type = expr->checked_type();
    tvm::runtime::Optional<tvm::TensorType> type = result_type.as<tvm::TensorType>();
    if (type != nullptr) {
        tvm::TensorType tensor_type = type.value();

        const tvm::DataType& data_type = tensor_type->dtype;
        const tvm::runtime::Array<tvm::PrimExpr>& shape = tensor_type->shape;

        std::cout << "MaxPool 2d Output DataType: " << data_type << ", Shape: ";
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
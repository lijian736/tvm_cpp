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
    // Step 1. Get the dense packed function
    const tvm::runtime::PackedFunc* dense = tvm::runtime::Registry::Get("relay.op.nn._make.dense");
    if (!dense) {
        std::cerr << "relay.op.nn._make.dense expression not found" << std::endl;
        return -1;
    }

    // Step 2. Generate the input data
    const tvm::runtime::PackedFunc* input_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!input_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    tvm::relay::Span input_span;

    // the input data type
    DLDataType input_data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the input tensor type, float32 and shape is {20, 30}, {50, 30}
    tvm::relay::TensorType input1_tensor_type{{20, 30}, tvm::DataType{input_data_type}};
    tvm::relay::TensorType input2_tensor_type{{50, 30}, tvm::DataType{input_data_type}};

    // the input var expression
    tvm::relay::Var input1_var = (*input_gen)("input1", input1_tensor_type, input_span);
    tvm::relay::Var input2_var = (*input_gen)("input2", input2_tensor_type, input_span);

    // Step 3. Compute the dense
    tvm::relay::Expr result_expr = (*dense)(input1_var, input2_var, 50, tvm::DataType());

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

        std::cout << "Dense Output DataType: " << data_type << ", Shape: ";
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
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
    // Step 1. Get the add packed function
    const tvm::runtime::PackedFunc* add = tvm::runtime::Registry::Get("relay.op._make.add");
    if (!add) {
        std::cerr << "relay.op._make.add expression not found" << std::endl;
        return -1;
    }

    // Step 2. Generate the input data
    const tvm::runtime::PackedFunc* input_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!input_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    tvm::relay::Span input_span;

    std::string input_name1 = "input_data1";
    std::string input_name2 = "input_data2";

    // the input data type
    DLDataType input_data_type = {DLDataTypeCode::kDLFloat, 32, 1};

    // the input tensor type1, float32 and shape is {1, 4, 5}
    tvm::relay::TensorType input_tensor_type1{{1, 4, 5}, tvm::DataType{input_data_type}};
    // the input tensor type2, float32 and shape is {2, 3, 1, 1}
    tvm::relay::TensorType input_tensor_type2{{2, 3, 1, 1}, tvm::DataType{input_data_type}};

    // the input var expression1
    tvm::relay::Var input_var1 = (*input_gen)(input_name1, input_tensor_type1, input_span);
    tvm::relay::Var input_var2 = (*input_gen)(input_name2, input_tensor_type2, input_span);

    // Step 3. Compute the add
    tvm::relay::Expr result_expr = (*add)(input_var1, input_var2);

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

        std::cout << "Add Output DataType: " << data_type << ", Shape: ";
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
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
    // Step 1. Get the concatenate packed function
    const tvm::runtime::PackedFunc* concatenate = tvm::runtime::Registry::Get("relay.op._make.concatenate");
    if (!concatenate) {
        std::cerr << "relay.op._make.concatenate expression not found" << std::endl;
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
    std::string input_name3 = "input_data3";

    // the input data type
    DLDataType input_data_type = {DLDataTypeCode::kDLFloat, 32, 1};

    // the input tensor type1, float32 and shape is {1, 20, 32, 32}
    tvm::relay::TensorType input_tensor_type1{{1, 20, 32, 32}, tvm::DataType{input_data_type}};
    // the input tensor type2, float32 and shape is {1, 3, 32, 32}
    tvm::relay::TensorType input_tensor_type2{{1, 3, 32, 32}, tvm::DataType{input_data_type}};
    // the input tensor type3, float32 and shape is {1, 64, 32, 32}
    tvm::relay::TensorType input_tensor_type3{{1, 64, 32, 32}, tvm::DataType{input_data_type}};

    // the input var expressions
    tvm::relay::Var input_var1 = (*input_gen)(input_name1, input_tensor_type1, input_span);
    tvm::relay::Var input_var2 = (*input_gen)(input_name2, input_tensor_type2, input_span);
    tvm::relay::Var input_var3 = (*input_gen)(input_name3, input_tensor_type3, input_span);

    tvm::runtime::Array<tvm::relay::Var> input_array{input_var1, input_var2, input_var3};

    const tvm::runtime::PackedFunc* tuple = tvm::runtime::Registry::Get("relay.ir.Tuple");
    if (!tuple) {
        std::cerr << "relay.ir.Tuple expression not found" << std::endl;
        return -1;
    }

    tvm::relay::Expr input = (*tuple)(input_array, tvm::relay::Span());

    // Step 3. Concate the input tensors by axis 1
    tvm::relay::Expr result_expr = (*concatenate)(input, 1);

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

        std::cout << "Concat Output DataType: " << data_type << ", Shape: ";
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
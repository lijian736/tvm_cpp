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
    // Constant in relay
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        std::cerr << "relay.ir.Constant expression not found" << std::endl;
        return -1;
    }

    // strided
    const tvm::runtime::PackedFunc* strided_slice = tvm::runtime::Registry::Get("relay.op._make.strided_slice");
    if (!strided_slice) {
        std::cerr << "relay.op._make.strided_slice expression not found" << std::endl;
        return -1;
    }

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

    // shape, data type, device
    tvm::runtime::NDArray initializer =
        tvm::runtime::NDArray::Empty({3, 4}, {DLDataTypeCode::kDLInt, 32, 1}, {DLDeviceType::kDLCPU, 0});
    // initialize the NDArray
    for (int i = 0; i < 12; ++i) {
        static_cast<int32_t*>(initializer->data)[i] = i + 1;
    }

    tvm::relay::Span span;
    // the constant expression, the NDArray as the Constant's content
    tvm::relay::Constant const_expr = (*const_gen)(initializer, span);

    tvm::runtime::Array<tvm::relay::IndexExpr> begin({0, 1});
    tvm::runtime::Array<tvm::relay::IndexExpr> end({2, 4});
    tvm::runtime::Array<tvm::relay::IndexExpr> strides({1, 1});

    tvm::runtime::String slice_mode = "end";
    
    tvm::relay::Expr result = (*strided_slice)(const_expr, begin, end, strides, slice_mode, nullptr);

    // infer type
    tvm::IRModule mod = IRModule::FromExpr(result);
    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    IRModule mod_new = (*pass_run)(infer_type_pass, mod);

    tvm::relay::Expr expr = mod_new->Lookup("main").as<FunctionNode>()->body;
    tvm::Type result_type = expr->checked_type();

    tvm::runtime::Optional<tvm::TensorType> type = result_type.as<tvm::TensorType>();
    if (type != nullptr) {
        tvm::TensorType tensor_type = type.value();

        const tvm::DataType& data_type = tensor_type->dtype;
        const tvm::runtime::Array<tvm::PrimExpr>& shape = tensor_type->shape;

        std::cout << "DataType: " << data_type << ", Shape: ";
        std::cout << "{";
        for (int i = 0; i < shape.size(); ++i) {
            const tvm::PrimExpr& exp = shape[i];
            const IntImmNode* node = exp.as<IntImmNode>();
            if (i) {
                std::cout << ", ";
            }
            if (node) {
                std::cout << node->value;
            } else {
                std::cout << "?";
            }
        }
        std::cout << "}" << std::endl;
    }

    return 0;
}
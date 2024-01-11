#include <tvm/relay/expr.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // Var node in relay
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    std::string name = "input var";
    tvm::relay::Span span;

    // the data type
    DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the tensor type
    tvm::relay::TensorType tensor_type{{3, 224, 224}, tvm::DataType{data_type}};

    // the var expression
    tvm::relay::Var var_expr = (*var_gen)(name, tensor_type, span);
    const tvm::relay::TensorTypeNode* type = var_expr->type_annotation.as<tvm::relay::TensorTypeNode>();

    // the basic info
    const tvm::String& name_hint = var_expr->name_hint();
    const tvm::runtime::Array<tvm::PrimExpr>& shape = type->shape;

    std::cout << "Name of Var: " << (std::string)name_hint << std::endl;
    std::cout << "Shape: ";
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

    return 0;
}
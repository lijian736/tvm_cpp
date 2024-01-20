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
    // Step 1. Get the related packed function
    // Var node in relay
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    // shape of
    const tvm::runtime::PackedFunc* shape_of = tvm::runtime::Registry::Get("relay.op._make.shape_of");
    if (!shape_of) {
        std::cerr << "relay.op._make.shape_of expression not found" << std::endl;
        return -1;
    }

    // print
    const tvm::runtime::PackedFunc* print_astext = tvm::runtime::Registry::Get("relay.ir.AsText");
    if (!print_astext) {
        std::cerr << "relay.ir.AsText expression not found" << std::endl;
        return -1;
    }

    // Step 2. create the var and mode
    std::string name = "input var";
    tvm::relay::Span span;

    // the data type
    DLDataType data_type = {DLDataTypeCode::kDLInt, 32, 1};
    // the tensor type
    tvm::relay::TensorType tensor_type{{3, 224, 224}, tvm::DataType{data_type}};

    // the var expression
    tvm::relay::Var var_expr = (*var_gen)(name, tensor_type, span);

    // Step 3. get the shape
    tvm::relay::Expr shape = (*shape_of)(var_expr, tvm::runtime::DataType());
    tvm::IRModule mod = IRModule::FromExpr(shape);

    // print the IR model
    tvm::String mod_text = (*print_astext)(mod, false, nullptr);

    std::string mode_str = (std::string)mod_text;
    std::cout << "IR Model info: " << std::endl << mode_str << std::endl;

    return 0;
}
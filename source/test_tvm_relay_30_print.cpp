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
    // Var node in relay
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    // transpose
    const tvm::runtime::PackedFunc* transpose = tvm::runtime::Registry::Get("relay.op._make.transpose");
    if (!transpose) {
        std::cerr << "relay.op._make.transpose expression not found" << std::endl;
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

    // print
    const tvm::runtime::PackedFunc* print_astext = tvm::runtime::Registry::Get("relay.ir.AsText");
    if (!print_astext) {
        std::cerr << "relay.ir.AsText expression not found" << std::endl;
        return -1;
    }

    //Pretty print
    const tvm::runtime::PackedFunc* pretty_print = tvm::runtime::Registry::Get("relay.ir.PrettyPrint");
    if (!pretty_print) {
        std::cerr << "relay.ir.PrettyPrint expression not found" << std::endl;
        return -1;
    }

    std::string name = "input var";
    tvm::relay::Span span;

    // the data type
    DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the tensor type
    tvm::relay::TensorType tensor_type{{1, 3, 85, 20, 20}, tvm::DataType{data_type}};

    // the var expression
    tvm::relay::Var var_expr = (*var_gen)(name, tensor_type, span);

    tvm::runtime::Array<tvm::Integer> axes({0, 1, 3, 4, 2});

    tvm::relay::Expr result = (*transpose)(var_expr, axes);

    // infer type
    tvm::IRModule mod = IRModule::FromExpr(result);
    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    IRModule mod_new = (*pass_run)(infer_type_pass, mod);

    // print the IR model
    tvm::String mod_text = (*print_astext)(mod, false, nullptr);

    std::string mode_str = (std::string)mod_text;
    std::cout << "IR Model info: " << std::endl << mode_str << std::endl;

    // print the type infered IR model
    tvm::String result_text = (*pretty_print)(mod_new);
    std::string result_str = (std::string)result_text;
    std::cout << "IR Model Result info: " << std::endl << result_str << std::endl;

    return 0;
}
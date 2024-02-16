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
    // get the module gen packaged function
    const tvm::runtime::PackedFunc* build_module = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
    if (!build_module) {
        std::cerr << "relay.build_module._BuildModule not found" << std::endl;
        return -1;
    }

    // get the relu packed function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        std::cerr << "relay.op.nn._make.relu expression not found" << std::endl;
        return -1;
    }

    // generate the input data
    const tvm::runtime::PackedFunc* input_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!input_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    // the input tensor type, float32 and shape is {1, 32, 320, 320}
    tvm::relay::TensorType input_tensor_type{{1, 32, 320, 320}, tvm::DataType::Float(32)};
    // the input var expression
    tvm::relay::Var input_var = (*input_gen)("input_data", input_tensor_type, tvm::relay::Span());

    // compute the relu
    tvm::relay::Expr relu_expr = (*relu)(input_var);
    tvm::IRModule ir_mod = IRModule::FromExpr(relu_expr);

    // generate the relay module
    tvm::runtime::Module module = (*build_module)();
    // get member functions
    tvm::runtime::PackedFunc get_graph_json = module->GetFunction("get_graph_json");
    tvm::runtime::PackedFunc get_module = module->GetFunction("get_module");
    tvm::runtime::PackedFunc build = module->GetFunction("build");
    tvm::runtime::PackedFunc optimize = module->GetFunction("optimize");
    tvm::runtime::PackedFunc set_params = module->GetFunction("set_params");
    tvm::runtime::PackedFunc get_params = module->GetFunction("get_params");
    tvm::runtime::PackedFunc get_function_metadata = module->GetFunction("get_function_metadata");
    tvm::runtime::PackedFunc get_executor_codegen_metadata = module->GetFunction("get_executor_codegen_metadata");
    tvm::runtime::PackedFunc get_devices = module->GetFunction("get_devices");
    tvm::runtime::PackedFunc get_irmodule = module->GetFunction("get_irmodule");

    
    return 0;
}
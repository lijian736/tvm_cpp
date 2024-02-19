#include <tvm/ir/memory_pools.h>
#include <tvm/ir/module.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/runtime.h>
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

TVM_REGISTER_GLOBAL("relay.backend.lower_call")
    .set_body_typed([](const relay::Call& call, const Array<te::Tensor>& inputs, const Target& target) {
        static auto fstrategy = Op::GetAttrMap<relay::FTVMStrategy>("FTVMStrategy");
        Op op = Downcast<Op>(call->op);
        auto out_type = call->checked_type();
        OpStrategy strategy = fstrategy[op](call->attrs, inputs, out_type, target);
        auto impl = strategy->specializations[0]->implementations[0];
        auto outs = impl.Compute(call->attrs, inputs, out_type);
        auto f = tvm::runtime::Registry::Get("relay.backend._make_LoweredOutput");
        if (!f) {
            LOG(FATAL) << "relay.backend._make_LoweredOutput is not registered";
        }

        std::cout << "successfully" << std::endl;
        return (*f)(outs, impl);
    });

int main(int argc, char** argv) {
    // get the module gen packaged function
    const tvm::runtime::PackedFunc* build_module = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
    if (!build_module) {
        std::cerr << "relay.build_module._BuildModule not found" << std::endl;
        return -1;
    }

    // create target packed function
    const tvm::runtime::PackedFunc* create_target = tvm::runtime::Registry::Get("target.Target");
    if (!create_target) {
        std::cerr << "target.Target not found" << std::endl;
        return -1;
    }

    // get the create-executor packed function
    const tvm::runtime::PackedFunc* create_executor = tvm::runtime::Registry::Get("relay.backend.CreateExecutor");
    if (!create_executor) {
        std::cerr << "relay.backend.CreateExecutor not found" << std::endl;
        return -1;
    }

    // get the create-runtime packed function
    const tvm::runtime::PackedFunc* create_runtime = tvm::runtime::Registry::Get("relay.backend.CreateRuntime");
    if (!create_runtime) {
        std::cerr << "relay.backend.CreateRuntime not found" << std::endl;
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

    // create the default pass context
    auto pass_ctx = PassContext::Create();
    pass_ctx->opt_level = 1;
    With<PassContext> scope(pass_ctx);

    // generate the relay module
    tvm::runtime::Module relay_module = (*build_module)();
    // get member functions
    tvm::runtime::PackedFunc get_graph_json = relay_module->GetFunction("get_graph_json");
    tvm::runtime::PackedFunc get_module = relay_module->GetFunction("get_module");
    tvm::runtime::PackedFunc build = relay_module->GetFunction("build");
    tvm::runtime::PackedFunc optimize = relay_module->GetFunction("optimize");
    tvm::runtime::PackedFunc set_params = relay_module->GetFunction("set_params");
    tvm::runtime::PackedFunc get_params = relay_module->GetFunction("get_params");
    tvm::runtime::PackedFunc get_function_metadata = relay_module->GetFunction("get_function_metadata");
    tvm::runtime::PackedFunc get_executor_codegen_metadata = relay_module->GetFunction("get_executor_codegen_metadata");
    tvm::runtime::PackedFunc get_devices = relay_module->GetFunction("get_devices");
    tvm::runtime::PackedFunc get_irmodule = relay_module->GetFunction("get_irmodule");

    tvm::Target target = (*create_target)("llvm");
    const Array<Target> raw_targets{target};

    // create the executor
    tvm::relay::Executor executor = (*create_executor)("graph", Map<String, ObjectRef>());
    // create the runtime
    tvm::relay::Runtime runtime = (*create_runtime)("cpp", Map<String, ObjectRef>());
    // memory pool
    WorkspaceMemoryPools mem_pool;
    ConstantMemoryPools const_mem_pool;

    build(ir_mod, raw_targets, target, executor, runtime, mem_pool, const_mem_pool, "test_module");

    return 0;
}
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <tvm/ir/memory_pools.h>
#include <tvm/ir/module.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>
#include <string>

#include "onnx.proto3.pb.h"
#include "utils/onnx_utils.h"
#include "utils/utils.h"

using namespace tvm_cpp::utils;
using namespace tvm_cpp::onnx_utils;

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // Step 0. Prepare the packed functions

    // the relu function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        std::cerr << "relay.op.nn._make.relu expression not found" << std::endl;
        return -1;
    }

    // the input generate function
    const tvm::runtime::PackedFunc* input_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!input_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        std::cerr << "relay.ir.Function expression not found" << std::endl;
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

    // Pretty print
    const tvm::runtime::PackedFunc* pretty_print = tvm::runtime::Registry::Get("relay.ir.PrettyPrint");
    if (!pretty_print) {
        std::cerr << "relay.ir.PrettyPrint expression not found" << std::endl;
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

    // get the module gen packaged function
    const tvm::runtime::PackedFunc* build_module = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
    if (!build_module) {
        std::cerr << "relay.build_module._BuildModule not found" << std::endl;
        return -1;
    }

    // Step 1. Generate the input tensor

    // the input tensor type, float32 and shape is {1, 1, 2, 2}
    tvm::relay::TensorType input_tensor_type{{1, 1, 2, 2}, tvm::DataType::Float(32)};

    // the input expression
    tvm::relay::Var input_var = (*input_gen)("input_data", input_tensor_type, tvm::relay::Span());
    // get the relu expression
    tvm::relay::Expr relu_expr = (*relu)(input_var);

    // Step 2. Generate a relay IRModule with only 1 node - relu

    // the graph input
    tvm::runtime::Array<tvm::relay::Expr> all_input;
    all_input.push_back(input_var);

    // the graph output
    tvm::relay::Expr all_output = relu_expr;

    // build the graph with its input, output
    tvm::relay::Expr relu_main_func =
        (*function)(all_input, all_output, tvm::relay::Type(), tvm::runtime::Array<tvm::relay::TypeVar>(),
                    tvm::DictAttrs(), tvm::relay::Span());

    // the graph relay ir module
    tvm::IRModule ir_module = IRModule::FromExpr(relu_main_func);

    // create the default pass context
    auto pass_ctx = PassContext::Create();
    pass_ctx->opt_level = 1;
    With<PassContext> scope(pass_ctx);

    // get the infer type pass
    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    // run the infer type pass
    ir_module = (*pass_run)(infer_type_pass, ir_module);

    tvm::String ir_module_text = (*pretty_print)(ir_module);
    std::cout << "the relu relay ir: " << std::endl << ir_module_text << std::endl;

    // Step 3. Create an Executor

    // create compilation target
    tvm::Target target = (*create_target)("llvm");
    int target_device_type = target->GetTargetDeviceType();
    const Array<Target> raw_targets{target};

    // create the executor
    tvm::relay::Executor executor = (*create_executor)("graph", Map<String, ObjectRef>());
    // create the runtime
    tvm::relay::Runtime runtime = (*create_runtime)("cpp", Map<String, ObjectRef>());
    // memory pool
    WorkspaceMemoryPools mem_pool;
    ConstantMemoryPools const_mem_pool;

    // generate the relay build module
    tvm::runtime::Module relay_build_module = (*build_module)();
    // get member functions
    tvm::runtime::PackedFunc get_graph_json = relay_build_module->GetFunction("get_graph_json");
    tvm::runtime::PackedFunc get_module = relay_build_module->GetFunction("get_module");
    tvm::runtime::PackedFunc build = relay_build_module->GetFunction("build");
    tvm::runtime::PackedFunc optimize = relay_build_module->GetFunction("optimize");
    tvm::runtime::PackedFunc set_params = relay_build_module->GetFunction("set_params");
    tvm::runtime::PackedFunc get_params = relay_build_module->GetFunction("get_params");
    tvm::runtime::PackedFunc get_function_metadata = relay_build_module->GetFunction("get_function_metadata");
    tvm::runtime::PackedFunc get_executor_codegen_metadata =
        relay_build_module->GetFunction("get_executor_codegen_metadata");
    tvm::runtime::PackedFunc get_devices = relay_build_module->GetFunction("get_devices");
    tvm::runtime::PackedFunc get_irmodule = relay_build_module->GetFunction("get_irmodule");

    // build the relay ir module
    build(ir_module, raw_targets, target, executor, runtime, mem_pool, const_mem_pool, "test_module");

    // Step 4. Do inference
    // Step 5. Output the result

    // the shape
    ShapeTuple shape = {2, 2};
    // the data type
    DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the device
    DLDevice dev{DLDeviceType::kDLCPU, 0};

    // shape, data type, device
    tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(shape, data_type, dev);

    static_cast<float*>(input->data)[0] = 1.0f;
    static_cast<float*>(input->data)[1] = -1.0f;
    static_cast<float*>(input->data)[2] = -2.0f;
    static_cast<float*>(input->data)[3] = 3.0f;

    return 0;
}
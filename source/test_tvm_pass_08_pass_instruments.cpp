#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <string>

#include "onnx.proto3.pb.h"
#include "onnx_op/op_parser.h"
#include "utils/onnx_utils.h"
#include "utils/relay_generator.h"
#include "utils/relay_utils.h"
#include "utils/status.h"
#include "utils/utils.h"

using namespace tvm_cpp::common;
using namespace tvm_cpp::utils;
using namespace tvm_cpp::onnx_utils;
using namespace tvm_cpp::onnx_op;
using namespace tvm_cpp::relay_utils;
using namespace tvm_cpp::relay_generator;

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // Pretty print
    const tvm::runtime::PackedFunc* pretty_print = tvm::runtime::Registry::Get("relay.ir.PrettyPrint");
    if (!pretty_print) {
        std::cerr << "relay.ir.PrettyPrint expression not found" << std::endl;
        return -1;
    }

    // print
    const tvm::runtime::PackedFunc* print_astext = tvm::runtime::Registry::Get("relay.ir.AsText");
    if (!print_astext) {
        std::cerr << "relay.ir.AsText expression not found" << std::endl;
        return -1;
    }

    // remove unused functions pass
    const tvm::runtime::PackedFunc* remove_unused =
        tvm::runtime::Registry::Get("relay._transform.RemoveUnusedFunctions");
    if (!remove_unused) {
        std::cerr << "relay._transform.RemoveUnusedFunctions expression not found" << std::endl;
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

    // Pass context gen functioon
    const tvm::runtime::PackedFunc* pass_ctx_gen = tvm::runtime::Registry::Get("transform.PassContext");
    if (!pass_ctx_gen) {
        std::cerr << "transform.PassContext not found" << std::endl;
        return -1;
    }

    // Pass instrument gen functioon
    const tvm::runtime::PackedFunc* pass_instrument_gen = tvm::runtime::Registry::Get("instrument.PassInstrument");
    if (!pass_instrument_gen) {
        std::cerr << "instrument.PassInstrument not found" << std::endl;
        return -1;
    }

    // generate module
    tvm::IRModule module;
    auto ret = generate_remove_unused_fun_module(module);
    if (!ret.is_ok()) {
        std::cout << ret << std::endl;
        return -1;
    }

    tvm::runtime::Array<tvm::runtime::String> entry_functions({"main"});

    runtime::TypedPackedFunc<void()> pass_ctx_enter = [] { std::cout << "------Enter pass context" << std::endl; };
    runtime::TypedPackedFunc<void()> pass_ctx_exit = [] { std::cout << "------Exit pass context" << std::endl; };
    runtime::TypedPackedFunc<bool(const IRModule&, const tvm::transform::PassInfo&)> should_run_pass = [](const tvm::IRModule& module, const tvm::transform::PassInfo& info) -> bool {
        if (info->name == "RemoveUnusedFunctions") {
            return true;
        }
        return false;
    };
    runtime::TypedPackedFunc<void(const IRModule&, const tvm::transform::PassInfo&)> before_pass = [](const tvm::IRModule& module, const tvm::transform::PassInfo& info) {
        std::cout << "------Run before pass [" << (std::string)info->name << "]" << std::endl;
    };
    runtime::TypedPackedFunc<void(const IRModule&, const tvm::transform::PassInfo&)> after_pass = [](const tvm::IRModule& module, const tvm::transform::PassInfo& info) {
        std::cout << "------Run after pass [" << (std::string)info->name << "]" << std::endl;
    };

    tvm::instrument::PassInstrument instrument = (*pass_instrument_gen)(
        "test_instrument", pass_ctx_enter, pass_ctx_exit, should_run_pass, before_pass, after_pass);
    // the optimizataion level
    int opt_level = 4;
    // the list of required passes
    tvm::runtime::Array<tvm::String> required;
    // the list of disabled passes
    tvm::runtime::Array<tvm::String> disabled;
    // a list of pass instrument implementations
    tvm::runtime::Array<tvm::instrument::PassInstrument> instruments({instrument});
    // pass specific configurations
    tvm::runtime::Optional<Map<String, ObjectRef>> configs;

    // generate the pass context
    PassContext pass_ctx = (*pass_ctx_gen)(opt_level, required, disabled, instruments, configs);
    tvm::With<PassContext> scope(pass_ctx);

    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    tvm::relay::transform::Pass ruf_pass = (*remove_unused)(entry_functions);

    // print the type infered IR module
    tvm::String before_text = (*pretty_print)(module);
    std::string before_str = (std::string)before_text;
    std::cout << "the ONNX relay IR Model info(before removing unused functions): " << std::endl
              << before_str << std::endl;

    module = (*pass_run)(infer_type_pass, module);
    module = (*pass_run)(ruf_pass, module);

    // print the new IR module after removing unused functions
    tvm::String after_text = (*pretty_print)(module);
    std::string after_str = (std::string)after_text;
    std::cout << "the ONNX relay IR Model info(after removing unused functions): " << std::endl
              << after_str << std::endl;

    return 0;
}
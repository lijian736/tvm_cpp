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

    // common subexpression elimination pass
    const tvm::runtime::PackedFunc* common_subexp =
        tvm::runtime::Registry::Get("relay._transform.EliminateCommonSubexpr");
    if (!common_subexp) {
        std::cerr << "relay._transform.EliminateCommonSubexpr expression not found" << std::endl;
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

    // generate module
    tvm::IRModule module;
    auto ret = generate_common_subexp_module(module);
    if (!ret.is_ok()) {
        std::cout << ret << std::endl;
        return -1;
    }

    // get common subexpr elimination pass
    auto pass_ctx = PassContext::Create();
    pass_ctx->opt_level = 4;
    With<PassContext> scope(pass_ctx);

    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    tvm::relay::transform::Pass cse_pass = (*common_subexp)(nullptr);

    // print the type infered IR module
    tvm::String before_text = (*pretty_print)(module);
    std::string before_str = (std::string)before_text;
    std::cout << "the ONNX relay IR Model info(before common subexpression elimination): " << std::endl
              << before_str << std::endl;

    module = (*pass_run)(infer_type_pass, module);
    module = (*pass_run)(cse_pass, module);

    // print the new IR module after common subexpression elimination
    tvm::String after_text = (*pretty_print)(module);
    std::string after_str = (std::string)after_text;
    std::cout << "the ONNX relay IR Model info(after common subexpression elimination): " << std::endl
              << after_str << std::endl;

    return 0;
}
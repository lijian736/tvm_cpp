#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <string>

#include "onnx.proto3.pb.h"
#include "onnx_op/op_parser.h"
#include "tvm/relay/expr_functor.h"
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

class TraversePass : public tvm::relay::ExprVisitor {
public:
    TraversePass() = default;
    virtual ~TraversePass() = default;

    void VisitExpr_(const OpNode* op) override { std::cout << "----Traverse op node: " << op->name << std::endl; }
};

int main(int argc, char** argv) {
    // generate relay expr
    tvm::relay::Expr expr;
    auto ret = generate_traverse_expr(expr);
    if (!ret.is_ok()) {
        std::cout << ret << std::endl;
        return -1;
    }

    // generate the pass context
    auto pass_ctx = PassContext::Create();
    pass_ctx->opt_level = 0;
    With<PassContext> scope(pass_ctx);

    TraversePass traverse_pass;
    traverse_pass(expr);

    return 0;
}
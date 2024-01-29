#include "softmax.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
Status SoftmaxParser::parse_op(const onnx::NodeProto& proto_node,
                               std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                               tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Softmax") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Softmax parameter");
    }

    return Status::ok();
}

std::string SoftmaxParser::get_name() { return "Softmax"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
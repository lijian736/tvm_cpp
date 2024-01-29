#include "squeeze.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze
Status SqueezeParser::parse_op(const onnx::NodeProto& proto_node,
                               std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                               tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Squeeze") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Squeeze parameter");
    }

    return Status::ok();
}

std::string SqueezeParser::get_name() { return "Squeeze"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
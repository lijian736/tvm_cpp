#include "sqrt.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
Status SqrtParser::parse_op(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Sqrt") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Sqrt parameter");
    }

    return Status::ok();
}

std::string SqrtParser::get_name() { return "Sqrt"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
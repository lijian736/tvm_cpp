#include "mul.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
Status MulParser::parse_op(const onnx::NodeProto& proto_node,
                           std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                           tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Mul") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Mul parameter");
    }

    // get the Mul relay function
    const tvm::runtime::PackedFunc* mul = tvm::runtime::Registry::Get("relay.op._make.multiply");
    if (!mul) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.multiply expression not found");
    }

    // get the inputs
    int input_size = proto_node.input_size();
    for (int i = 0; i < input_size; ++i) {
        const auto& input = proto_node.input(i);
    }

    // get the outputs
    int output_size = proto_node.output_size();
    for (int i = 0; i < output_size; ++i) {
        const auto& output = proto_node.output(i);
    }

    return Status::ok();
}

std::string MulParser::get_name() { return "Mul"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
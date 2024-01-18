#include "relu.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
Status ReluParser::parse_op(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                            tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Relu") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Relu parameter");
    }

    // get the relu relay function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.relu expression not found");
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

std::string ReluParser::get_name() { return "Relu"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
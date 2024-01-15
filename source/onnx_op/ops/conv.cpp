#include "conv.h"

namespace tvm_cpp {
namespace onnx_op {

Status Conv2DParser::parse_op(const onnx::NodeProto& proto_node,
                              const std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                              tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Conv") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Concat parameter");
    }

    // get the Conv2D relay function
    const tvm::runtime::PackedFunc* conv2d = tvm::runtime::Registry::Get("relay.op.nn._make.conv2d");
    if (!conv2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.conv2d expression not found");
    }

    // get the attributes for Conv2D op
    auto attr_size = proto_node.attribute_size();
    for (decltype(attr_size) i = 0; i < attr_size; ++i) {
        const auto& attr = proto_node.attribute(i);
    }

    // get the inputs
    auto input_size = proto_node.input_size();
    for (decltype(input_size) i = 0; i < input_size; ++i) {
        const auto& input = proto_node.input(i);
    }

    // get the outputs
    auto output_size = proto_node.output_size();
    for (decltype(output_size) i = 0; i < output_size; ++i) {
        const auto& output = proto_node.output(i);
    }

    return Status::ok();
}

}    // namespace onnx_op
}    // namespace tvm_cpp
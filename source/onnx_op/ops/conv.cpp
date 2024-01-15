#include "conv.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
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
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    std::string auto_pad = get_attr_or_default<std::string>("auto_pad", "NOTSET", attrs_map);
    std::vector<int64_t> dilations = get_attrs_or_default<int64_t>("dilations", {}, attrs_map);
    int64_t group = get_attr_or_default<int64_t>("group", 1, attrs_map);
    std::vector<int64_t> kernel_shape = get_attrs_or_default<int64_t>("kernel_shape", {}, attrs_map);
    std::vector<int64_t> pads = get_attrs_or_default<int64_t>("pads", {}, attrs_map);
    std::vector<int64_t> strides = get_attrs_or_default<int64_t>("strides", {}, attrs_map);

    if (auto_pad != "NOTSET") {
        std::ostringstream oss;
        oss << "Node: Conv[" << proto_node.name()
            << "], auto_pad attribute is not supported now. auto_pad value: " << auto_pad;

        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
    }

    if (group <= 1) {
        std::ostringstream oss;
        oss << "Node: Conv[" << proto_node.name()
            << "], group convolution is not supported now. group attribute: " << group;

        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
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

}    // namespace onnx_op
}    // namespace tvm_cpp
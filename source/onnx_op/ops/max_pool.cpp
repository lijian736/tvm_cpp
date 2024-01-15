#include "max_pool.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
Status MaxPool2DParser::parse_op(const onnx::NodeProto& proto_node,
                                 const std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                                 tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "MaxPool") {
        return Status(StatusCode::INVALID_PARAM, "Invalid MaxPool parameter");
    }

    // get the MaxPool 2d relay function
    const tvm::runtime::PackedFunc* max_pool2d = tvm::runtime::Registry::Get("relay.op.nn._make.max_pool2d");
    if (!max_pool2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.max_pool2d expression not found");
    }

    // get the attributes for MaxPool 2d op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    std::string auto_pad = get_attr_or_default<std::string>("auto_pad", "NOTSET", attrs_map);
    int64_t ceil_mode = get_attr_or_default<int64_t>("ceil_mode", 0, attrs_map);
    std::vector<int64_t> dilations = get_attrs_or_default<int64_t>("dilations", {}, attrs_map);
    std::vector<int64_t> kernel_shape = get_attrs_or_default<int64_t>("kernel_shape", {}, attrs_map);
    std::vector<int64_t> pads = get_attrs_or_default<int64_t>("pads", {}, attrs_map);
    int64_t storage_order = get_attr_or_default<int64_t>("storage_order", 0, attrs_map);
    std::vector<int64_t> strides = get_attrs_or_default<int64_t>("strides", {}, attrs_map);

    // auto_pad is a DEPRECATED attribute, skip it
    if (auto_pad != "NOTSET") {
        std::ostringstream oss;
        oss << "Node: MaxPool[" << proto_node.name()
            << "], auto_pad is a DEPRECATED attribute, not supported now. auto_pad value: " << auto_pad;

        return Status(StatusCode::INVALID_PARAM, oss.str());
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
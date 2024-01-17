#include "resize.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
Status Resize2DParser::parse_op(const onnx::NodeProto& proto_node,
                                std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                                tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Resize") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Resize parameter");
    }

    // get the Resize2D relay function
    const tvm::runtime::PackedFunc* resize2d = tvm::runtime::Registry::Get("relay.op.image._make.resize2d");
    if (!resize2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.image._make.resize2d expression not found");
    }

    // get the attributes for Resize 2d op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    int64_t antialias = get_attr_or_default<int64_t>("antialias", 0, attrs_map);
    std::vector<int64_t> axes = get_attrs_or_default<int64_t>("axes", {}, attrs_map);
    std::string coordinate =
        get_attr_or_default<std::string>("coordinate_transformation_mode", "half_pixel", attrs_map);
    float cubic = get_attr_or_default<float>("cubic_coeff_a ", -0.75f, attrs_map);
    int64_t exclude = get_attr_or_default<int64_t>("exclude_outside", 0, attrs_map);
    float extrapolation = get_attr_or_default<float>("extrapolation_value ", -0.75f, attrs_map);
    std::string aspect = get_attr_or_default<std::string>("keep_aspect_ratio_policy", "stretch", attrs_map);
    std::string mode = get_attr_or_default<std::string>("mode", "nearest", attrs_map);
    std::string nearest_mode = get_attr_or_default<std::string>("nearest_mode", "round_prefer_floor", attrs_map);

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
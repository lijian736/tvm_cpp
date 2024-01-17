#include "conv.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
Status Conv2DParser::parse_op(const onnx::NodeProto& proto_node,
                              std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
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
    if (input_size < 2) {
        std::ostringstream oss;
        oss << "Invalid inputs of Conv: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Conv: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& weight = proto_node.input(1);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Conv: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    auto weight_iter = expressions.find(weight);
    if (weight_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Conv: " << proto_node.name() << " input: " << weight;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    tvm::runtime::Array<tvm::relay::IndexExpr> strides_exp;
    tvm::runtime::Array<tvm::relay::IndexExpr> padding_exp;
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation_exp;
    int groups = 1;

    std::vector<int64_t> weight_shape;
    tvm_cpp::relay_utils::infer_relay_shape(weight_iter->second, weight_shape);
    if (weight_shape.size() < 1) {
        return Status(StatusCode::INVALID_MODEL, "Invalid weight shape");
    }

    tvm::relay::IndexExpr channels((int32_t)weight_shape[0]);
    tvm::runtime::Array<tvm::relay::IndexExpr> kernel_size;
    tvm::runtime::String data_layout = "NCHW";
    tvm::runtime::String kernel_layout = "OIHW";
    tvm::runtime::String out_layout = "";
    tvm::DataType out_dtype;

    std::for_each(strides.begin(), strides.end(), [&](int64_t val) { strides_exp.push_back((int32_t)val); });

    std::for_each(pads.begin(), pads.end(), [&](int64_t val) { padding_exp.push_back((int32_t)val); });

    std::for_each(dilations.begin(), dilations.end(), [&](int64_t val) { dilation_exp.push_back((int32_t)val); });

    std::for_each(kernel_shape.begin(), kernel_shape.end(), [&](int64_t val) { kernel_size.push_back((int32_t)val); });

    tvm::relay::Expr out_expr =
        (*conv2d)(input_iter->second, weight_iter->second, strides_exp, padding_exp, dilation_exp, groups, channels,
                  kernel_size, data_layout, kernel_layout, out_layout, out_dtype);

    if (input_size > 2) {
        const std::string& bias = proto_node.input(2);

    } else {
        auto ret = expressions.emplace(output, out_expr);
        if (!ret.second) {
            ret.first->second = std::move(out_expr);
        }
    }

    return Status::ok();
}

}    // namespace onnx_op
}    // namespace tvm_cpp
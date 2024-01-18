#include "max_pool.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
Status MaxPool2DParser::parse_op(const onnx::NodeProto& proto_node,
                                 std::unordered_map<std::string, tvm::relay::Expr>& expressions,
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
    if (input_size != 1) {
        std::ostringstream oss;
        oss << "Invalid inputs of MaxPool: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of MaxPool: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, MaxPool: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    std::vector<int64_t> input_shape;
    tvm_cpp::relay_utils::infer_relay_shape(input_iter->second, input_shape);

    int dims = input_shape.size() - 2;
    if (dims != 2) {
        return Status(StatusCode::INVALID_MODEL, "Only support 2D MaxPool");
    }

    // set default value for dilations
    if (dilations.size() == 0) {
        // count, value
        dilations.assign(2, 1);
    }

    // check the dilations for 2d maxpool
    if (dilations.size() != 2) {
        std::ostringstream oss;
        oss << "Invalid dilations, MaxPool: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // set the default value for padding
    if (pads.size() == 0) {
        pads.assign(4, 0);
    }

    // check the padding for 2d maxpool
    if (pads.size() != 4) {
        std::ostringstream oss;
        oss << "Invalid padding, MaxPool: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // set the default value for strides
    if (strides.size() == 0) {
        strides.assign(2, 1);
    }

    // check the strides for 2d maxpool
    if (strides.size() != 2) {
        std::ostringstream oss;
        oss << "Invalid strides, MaxPool: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // do the 2d max pooling
    tvm::runtime::Array<tvm::relay::IndexExpr> pool_size;
    tvm::runtime::Array<tvm::relay::IndexExpr> strides_exp;
    tvm::runtime::Array<tvm::relay::IndexExpr> dilation_exp;
    tvm::runtime::Array<tvm::relay::IndexExpr> padding_exp;

    tvm::runtime::String layout = "NCHW";
    tvm::runtime::String out_layout = "";

    std::for_each(strides.begin(), strides.end(), [&](int64_t val) { strides_exp.push_back((int32_t)val); });

    std::for_each(pads.begin(), pads.end(), [&](int64_t val) { padding_exp.push_back((int32_t)val); });

    std::for_each(dilations.begin(), dilations.end(), [&](int64_t val) { dilation_exp.push_back((int32_t)val); });

    std::for_each(kernel_shape.begin(), kernel_shape.end(), [&](int64_t val) { pool_size.push_back((int32_t)val); });

    tvm::relay::Expr out_expr = (*max_pool2d)(input_iter->second, pool_size, strides_exp, dilation_exp, padding_exp,
                                              layout, out_layout, (ceil_mode ? true : false));

    auto status = fold_const(out_expr);
    if (!status.is_ok()) {
        return status;
    }

    // add to expressions
    auto ret = expressions.emplace(output, out_expr);
    if (!ret.second) {
        ret.first->second = out_expr;
    }
    relay = out_expr;

    return Status::ok();
}

std::string MaxPool2DParser::get_name() { return "MaxPool"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
#include "softmax.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
Status SoftmaxParser::parse_op(const onnx::NodeProto& proto_node,
                               std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                               tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Softmax") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Softmax parameter");
    }

    // get the softmax relay function
    const tvm::runtime::PackedFunc* softmax = tvm::runtime::Registry::Get("relay.op.nn._make.softmax");
    if (!softmax) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.softmax expression not found");
    }

    // get the attributes for concat op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    int64_t axis = get_attr_or_default<int64_t>("axis", -1, attrs_map);

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 1) {
        std::ostringstream oss;
        oss << "Invalid inputs of Softmax: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Softmax: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Softmax: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    std::vector<int64_t> axes_shape;
    tvm::DataType axes_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(input_iter->second, axes_shape, axes_dtype);

    int64_t input_rank = (int64_t)axes_shape.size();
    if (axis < -input_rank || axis > input_rank - 1) {
        std::ostringstream oss;
        oss << "Invalid Softmax axis: " << proto_node.name() << " axis: " << input_rank;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    if (axis < 0) {
        axis += input_rank;
    }

    // Compute the Softmax
    tvm::relay::Expr result_expr = (*softmax)(input_iter->second, (int)axis);

    auto status = fold_const(result_expr);
    if (!status.is_ok()) {
        return status;
    }

    // add to expressions
    auto ret = expressions.emplace(output, result_expr);
    if (!ret.second) {
        ret.first->second = result_expr;
    }
    relay = result_expr;

    return Status::ok();
}

std::string SoftmaxParser::get_name() { return "Softmax"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
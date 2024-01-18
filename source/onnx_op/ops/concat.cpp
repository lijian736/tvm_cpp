#include "concat.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
Status ConcatParser::parse_op(const onnx::NodeProto& proto_node,
                              std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Concat") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Concat parameter");
    }

    // get the Concat relay function
    const tvm::runtime::PackedFunc* concat = tvm::runtime::Registry::Get("relay.op._make.concatenate");
    if (!concat) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.concatenate expression not found");
    }

    // get the tuple relay
    const tvm::runtime::PackedFunc* tuple = tvm::runtime::Registry::Get("relay.ir.Tuple");
    if (!tuple) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Tuple expression not found");
    }

    // get the attributes for concat op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    int64_t axis;
    auto status = get_attr<int64_t>("axis", &axis, attrs_map);
    if (!status.is_ok()) {
        return status;
    }

    tvm::runtime::Array<tvm::relay::Expr> input_array;
    // get the inputs
    int input_size = proto_node.input_size();
    for (int i = 0; i < input_size; ++i) {
        const auto& input_name = proto_node.input(i);
        auto input_iter = expressions.find(input_name);
        if (input_iter == expressions.end()) {
            std::ostringstream oss;
            oss << "Input not found, Concat: " << proto_node.name() << " input: " << input_name;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        input_array.push_back(input_iter->second);
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Concat: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    tvm::relay::Expr all_input = (*tuple)(input_array, tvm::relay::Span());

    // Concate the input tensors by axis
    tvm::relay::Expr result_expr = (*concat)(all_input, axis);

    status = fold_const(result_expr);
    if (!status.is_ok()) {
        return status;
    }

    // add to expressions
    const std::string& output = proto_node.output(0);
    auto ret = expressions.emplace(output, result_expr);
    if (!ret.second) {
        ret.first->second = result_expr;
    }
    relay = result_expr;

    return Status::ok();
}

std::string ConcatParser::get_name() { return "Concat"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
#include "transpose.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
Status TransposeParser::parse_op(const onnx::NodeProto& proto_node,
                                 std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                                 tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Transpose") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Transpose parameter");
    }

    // get the Transpose relay function
    const tvm::runtime::PackedFunc* transpose = tvm::runtime::Registry::Get("relay.op._make.transpose");
    if (!transpose) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.transpose expression not found");
    }

    // get the attributes for Transpose op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    std::vector<int64_t> perm = get_attrs_or_default<int64_t>("perm", {}, attrs_map);

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 1) {
        std::ostringstream oss;
        oss << "Invalid inputs of Transpose: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Transpose: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Transpose: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    tvm::runtime::Array<tvm::Integer> axes;
    std::for_each(perm.begin(), perm.end(), [&](int64_t val) { axes.push_back((int32_t)val); });

    tvm::relay::Expr result_expr = (*transpose)(input_iter->second, axes);

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

std::string TransposeParser::get_name() { return "Transpose"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
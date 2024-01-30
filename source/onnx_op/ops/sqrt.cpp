#include "sqrt.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
Status SqrtParser::parse_op(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Sqrt") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Sqrt parameter");
    }

    // get the sqrt relay function
    const tvm::runtime::PackedFunc* sqrt_relay = tvm::runtime::Registry::Get("relay.op._make.sqrt");
    if (!sqrt_relay) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.sqrt expression not found");
    }

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 1) {
        std::ostringstream oss;
        oss << "Invalid inputs of Sqrt: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Sqrt: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Sqrt: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // Compute the sqrt
    tvm::relay::Expr result_expr = (*sqrt_relay)(input_iter->second);

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

std::string SqrtParser::get_name() { return "Sqrt"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
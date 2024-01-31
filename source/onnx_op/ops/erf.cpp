#include "erf.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Erf
Status ErfParser::parse_op(const onnx::NodeProto& proto_node,
                           std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Erf") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Erf parameter");
    }

    // get the Erf relay function
    const tvm::runtime::PackedFunc* erf_relay = tvm::runtime::Registry::Get("relay.op._make.erf");
    if (!erf_relay) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.erf expression not found");
    }

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 1) {
        std::ostringstream oss;
        oss << "Invalid inputs of Erf: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Erf: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Erf: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // Compute the erf
    tvm::relay::Expr result_expr = (*erf_relay)(input_iter->second);

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

std::string ErfParser::get_name() { return "Erf"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
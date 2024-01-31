#include "pow.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow
Status PowParser::parse_op(const onnx::NodeProto& proto_node,
                           std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Pow") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Pow parameter");
    }

    // get the Pow relay function
    const tvm::runtime::PackedFunc* power_relay = tvm::runtime::Registry::Get("relay.op._make.power");
    if (!power_relay) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.power expression not found");
    }

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 2) {
        std::ostringstream oss;
        oss << "Invalid inputs of Pow: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Pow: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input0 = proto_node.input(0);
    const std::string& input1 = proto_node.input(1);
    const std::string& output = proto_node.output(0);

    auto input0_iter = expressions.find(input0);
    if (input0_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Pow: " << proto_node.name() << " input: " << input0;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    auto input1_iter = expressions.find(input1);
    if (input1_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Pow: " << proto_node.name() << " input: " << input1;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    tvm::relay::Expr result_expr = (*power_relay)(input0_iter->second, input1_iter->second);

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

std::string PowParser::get_name() { return "Pow"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
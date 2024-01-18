#include "add.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
Status AddParser::parse_op(const onnx::NodeProto& proto_node,
                           std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Add") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Add parameter");
    }

    // get the Add relay function
    const tvm::runtime::PackedFunc* add = tvm::runtime::Registry::Get("relay.op._make.add");
    if (!add) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.add expression not found");
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

std::string AddParser::get_name() { return "Add"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
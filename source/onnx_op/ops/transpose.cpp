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
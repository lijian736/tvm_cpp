#include "concat.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
Status ConcatParser::parse_op(const onnx::NodeProto& proto_node,
                              const std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                              tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Concat") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Concat parameter");
    }

    // get the Concat relay function
    const tvm::runtime::PackedFunc* concat = tvm::runtime::Registry::Get("relay.op._make.concatenate");
    if (!concat) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.concatenate expression not found");
    }

    // get the attributes for concat op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    int64_t axis;
    auto ret = get_attr<int64_t>("axis", &axis, attrs_map);
    if(!ret.is_ok()){
        return ret;
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

}    // namespace onnx_op
}    // namespace tvm_cpp
#include "reshape.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
Status ReshapeParser::parse_op(const onnx::NodeProto& proto_node,
                               std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                               tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Reshape") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Reshape parameter");
    }

    // get the Reshape relay function
    const tvm::runtime::PackedFunc* reshape = tvm::runtime::Registry::Get("relay.op._make.reshape");
    if (!reshape) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.reshape expression not found");
    }

    // get the attributes for Reshape op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    int64_t allowzero = get_attr_or_default<int64_t>("allowzero ", 0, attrs_map);

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 2) {
        std::ostringstream oss;
        oss << "Invalid inputs of Reshape: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Reshape: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& new_shape = proto_node.input(1);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Reshape: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    auto new_shape_iter = expressions.find(new_shape);
    if (new_shape_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Reshape: " << proto_node.name() << " input: " << new_shape;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    std::vector<int64_t> new_shape_shape;
    tvm::DataType new_shape_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(new_shape_iter->second, new_shape_shape, new_shape_dtype);
    int64_t scale_ele_nums = 1;
    for (auto& dim : new_shape_shape) {
        scale_ele_nums *= dim;
    }

    const tvm::relay::ConstantNode* const_expr = new_shape_iter->second.as<tvm::relay::ConstantNode>();
    if (!const_expr) {
        return Status(StatusCode::RUNTIME_ERROR, "cast new shape to const node fails for REshape");
    }

    tvm::runtime::Array<tvm::Integer> new_shape_arr;

    tvm::runtime::NDArray new_shape_data = const_expr->data;

    if (new_shape_dtype.is_int()) {
        if (new_shape_dtype.bits() == 64) {
            for (int i = 0; i < scale_ele_nums; ++i) {
                auto val = static_cast<int64_t*>(new_shape_data->data)[i];
                new_shape_arr.push_back((int)val);
            }
        } else {
            return Status(StatusCode::NOT_IMPLEMENTED, "unsupported new shape data type for Reshape");
        }
    } else {
        return Status(StatusCode::NOT_IMPLEMENTED, "unsupported new shape data type for Reshape");
    }

    tvm::relay::Expr result_expr = (*reshape)(input_iter->second, new_shape_arr, (allowzero ? true : false));

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

std::string ReshapeParser::get_name() { return "Reshape"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
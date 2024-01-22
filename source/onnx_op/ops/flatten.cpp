#include "flatten.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten
Status FlattenParser::parse_op(const onnx::NodeProto& proto_node,
                               std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                               tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Flatten") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Flatten parameter");
    }

    // get the attributes for Flatten op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    int64_t axis = get_attr_or_default<int64_t>("axis", 1, attrs_map);

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 1) {
        std::ostringstream oss;
        oss << "Invalid inputs of Flatten: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Flatten: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Flatten: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the input shape and data type
    std::vector<int64_t> input_shape;
    tvm::DataType input_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(input_iter->second, input_shape, input_dtype);

    int input_rank = (int)input_shape.size();

    // the axis must be in range[-rank, rank]
    if (axis < -input_rank || axis > input_rank) {
        std::ostringstream oss;
        oss << "Invalid axis value, Flatten: " << proto_node.name() << " axis: " << axis;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    if (axis < 0) {
        axis += input_rank;
    }

    tvm::relay::Expr result_expr;
    if (axis == 1) {
        // get the batch flatten relay function
        const tvm::runtime::PackedFunc* batch_flatten = tvm::runtime::Registry::Get("relay.op.nn._make.batch_flatten");
        if (!batch_flatten) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.batch_flatten expression not found");
        }
        result_expr = (*batch_flatten)(input_iter->second);
    } else {
        int d0 = 1;
        int d1 = 1;

        for (int i = 0; i < axis; ++i) {
            d0 *= input_shape[i];
        }

        for (int i = axis; i < input_rank; ++i) {
            d1 *= input_shape[i];
        }

        tvm::runtime::Array<tvm::Integer> shape_arr({d0, d1});
        // get the Reshape relay function
        const tvm::runtime::PackedFunc* reshape = tvm::runtime::Registry::Get("relay.op._make.reshape");
        if (!reshape) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.reshape expression not found");
        }

        result_expr = (*reshape)(input_iter->second, shape_arr, false);
    }

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

std::string FlattenParser::get_name() { return "Flatten"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
#include "squeeze.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze
Status SqueezeParser::parse_op(const onnx::NodeProto& proto_node,
                               std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                               tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Squeeze") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Squeeze parameter");
    }

    // get the squeeze relay function
    const tvm::runtime::PackedFunc* squeeze = tvm::runtime::Registry::Get("relay.op._make.squeeze");
    if (!squeeze) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.squeeze expression not found");
    }

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 1 && input_size != 2) {
        std::ostringstream oss;
        oss << "Invalid inputs of Squeeze: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Squeeze: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input0 = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input0_iter = expressions.find(input0);
    if (input0_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Squeeze: " << proto_node.name() << " input: " << input0;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    tvm::relay::Expr result_expr;
    
    if (input_size == 1) {
        result_expr = (*squeeze)(input0_iter->second, nullptr);
    } else if (input_size == 2) {
        tvm::runtime::Array<tvm::Integer> axes;

        const std::string& input1 = proto_node.input(1);
        auto input1_iter = expressions.find(input1);
        if (input1_iter == expressions.end()) {
            std::ostringstream oss;
            oss << "Input not found, Squeeze: " << proto_node.name() << " input: " << input1;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        const tvm::relay::ConstantNode* const_expr = input1_iter->second.as<tvm::relay::ConstantNode>();
        if (!const_expr) {
            return Status(StatusCode::RUNTIME_ERROR, "cast axes to const node fails for Squeeze");
        }

        std::vector<int64_t> axes_shape;
        tvm::DataType axes_dtype;
        tvm_cpp::relay_utils::infer_relay_shape_dtype(input1_iter->second, axes_shape, axes_dtype);

        int64_t axes_ele_nums = 1;
        for (auto& dim : axes_shape) {
            axes_ele_nums *= dim;
        }

        tvm::runtime::NDArray axes_data = const_expr->data;

        if (axes_dtype.is_int()) {
            if (axes_dtype.bits() == 64) {
                for (int i = 0; i < axes_ele_nums; ++i) {
                    auto val = static_cast<int64_t*>(axes_data->data)[i];
                    axes.push_back((int)val);
                }
            } else {
                return Status(StatusCode::NOT_IMPLEMENTED, "unsupported axes data type for Squeeze");
            }
        } else {
            return Status(StatusCode::NOT_IMPLEMENTED, "unsupported axes data type for Squeeze");
        }

        result_expr = (*squeeze)(input0_iter->second, axes);
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

std::string SqueezeParser::get_name() { return "Squeeze"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
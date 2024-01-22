#include "global_avg_pool.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
Status GlobalAveragePoolParser::parse_op(const onnx::NodeProto& proto_node,
                                         std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                                         tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "GlobalAveragePool") {
        return Status(StatusCode::INVALID_PARAM, "Invalid GlobalAveragePool parameter");
    }

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 1) {
        std::ostringstream oss;
        oss << "Invalid inputs of GlobalAveragePool: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of GlobalAveragePool: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, GlobalAveragePool: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    std::vector<int64_t> input_shape;
    tvm::DataType input_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(input_iter->second, input_shape, input_dtype);

    tvm::relay::Expr result_expr;
    int input_rank = (int)input_shape.size();
    if (input_rank == 3) {
        // get the 1d global avg pool relay function
        const tvm::runtime::PackedFunc* global_avg_pool =
            tvm::runtime::Registry::Get("relay.op.nn._make.adaptive_avg_pool1d");
        if (!global_avg_pool) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.adaptive_avg_pool1d expression not found");
        }

        tvm::runtime::Array<tvm::relay::IndexExpr> output_size({1});
        result_expr = (*global_avg_pool)(input_iter->second, output_size, "NCW", "");
    } else if (input_rank == 4) {
        // get the 2d global avg pool relay function
        const tvm::runtime::PackedFunc* global_avg_pool =
            tvm::runtime::Registry::Get("relay.op.nn._make.global_avg_pool2d");
        if (!global_avg_pool) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.global_avg_pool2d expression not found");
        }

        result_expr = (*global_avg_pool)(input_iter->second, output_size, "NCHW", "");
    } else if (input_rank == 5) {
        // get the 3d global avg pool relay function
        const tvm::runtime::PackedFunc* global_avg_pool =
            tvm::runtime::Registry::Get("relay.op.nn._make.adaptive_avg_pool3d");
        if (!global_avg_pool) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.adaptive_avg_pool3d expression not found");
        }

        tvm::runtime::Array<tvm::relay::IndexExpr> output_size({1, 1, 1});
        result_expr = (*global_avg_pool)(input_iter->second, output_size, "NCDHW", "");
    } else {
        std::ostringstream oss;
        oss << "unsupported input rank for GlobalAveragePool: " << proto_node.name() << " input rank: " << input_rank;
        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
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

std::string GlobalAveragePoolParser::get_name() { return "GlobalAveragePool"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
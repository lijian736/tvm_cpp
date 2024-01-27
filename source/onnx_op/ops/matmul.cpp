#include "matmul.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
Status MatMulParser::parse_op(const onnx::NodeProto& proto_node,
                              std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "MatMul") {
        return Status(StatusCode::INVALID_PARAM, "Invalid MatMul parameter");
    }

    // transpose
    const tvm::runtime::PackedFunc* transpose = tvm::runtime::Registry::Get("relay.op._make.transpose");
    if (!transpose) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.transpose expression not found");
    }

    // get the dense function
    const tvm::runtime::PackedFunc* dense = tvm::runtime::Registry::Get("relay.op.nn._make.dense");
    if (!dense) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.dense expression not found");
    }

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size != 2) {
        std::ostringstream oss;
        oss << "Invalid inputs of MatMul: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of MatMul: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& matrixA_name = proto_node.input(0);
    const std::string& matrixB_name = proto_node.input(1);
    const std::string& output = proto_node.output(0);

    auto matrixA_iter = expressions.find(matrixA_name);
    if (matrixA_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, MatMul: " << proto_node.name() << " input: " << matrixA_name;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    auto matrixB_iter = expressions.find(matrixB_name);
    if (matrixB_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, MatMul: " << proto_node.name() << " input: " << matrixB_name;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the matrix A shape
    std::vector<int64_t> matrixA_shape;
    tvm::DataType matrixA_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(matrixA_iter->second, matrixA_shape, matrixA_dtype);

    // get the matrix B shape
    std::vector<int64_t> matrixB_shape;
    tvm::DataType matrixB_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(matrixB_iter->second, matrixB_shape, matrixB_dtype);

    if (matrixA_shape.size() != 2 || matrixB_shape.size() != 2) {
        std::ostringstream oss;
        oss << "MatMul [" << proto_node.name() << "] is not implemented with shape rank ["
            << " matrix A rank: " << matrixA_shape.size() << ", matrix B rank: " << matrixB_shape.size() << "]";
        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
    }

    tvm::runtime::Array<tvm::Integer> axes({1, 0});
    // matrix B
    tvm::relay::Expr matrixB = matrixB_iter->second;
    // transpose matrix B
    matrixB = (*transpose)(matrixB, axes);

    tvm::relay::Expr result_expr;
    result_expr = (*dense)(matrixA_iter->second, matrixB, matrixB_shape[1], matrixA_dtype);

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

std::string MatMulParser::get_name() { return "MatMul"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
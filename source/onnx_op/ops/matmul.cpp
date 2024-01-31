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

    return parse_method_1(proto_node, expressions, relay);
    // return parse_method_2(proto_node, expressions, relay);
}

Status MatMulParser::parse_method_1(const onnx::NodeProto& proto_node,
                                    std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                                    tvm::relay::Expr& relay) {
    // get the transpose relay function
    const tvm::runtime::PackedFunc* transpose = tvm::runtime::Registry::Get("relay.op._make.transpose");
    if (!transpose) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.transpose expression not found");
    }

    // get the dense relay function
    const tvm::runtime::PackedFunc* dense = tvm::runtime::Registry::Get("relay.op.nn._make.dense");
    if (!dense) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.dense expression not found");
    }

    // get the Reshape relay function
    const tvm::runtime::PackedFunc* reshape = tvm::runtime::Registry::Get("relay.op._make.reshape");
    if (!reshape) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.reshape expression not found");
    }

    // get the boradcast_to relay function
    const tvm::runtime::PackedFunc* broadcast_to = tvm::runtime::Registry::Get("relay.op._make.broadcast_to");
    if (!broadcast_to) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.broadcast_to expression not found");
    }

    // get the batch matmul relay function
    const tvm::runtime::PackedFunc* batch_matmul = tvm::runtime::Registry::Get("relay.op.nn._make.batch_matmul");
    if (!batch_matmul) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.batch_matmul expression not found");
    }

    // get the expand_dims relay function
    const tvm::runtime::PackedFunc* expand_dims = tvm::runtime::Registry::Get("relay.op._make.expand_dims");
    if (!expand_dims) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.expand_dims expression not found");
    }

    // get the squeeze relay function
    const tvm::runtime::PackedFunc* squeeze = tvm::runtime::Registry::Get("relay.op._make.squeeze");
    if (!squeeze) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.squeeze expression not found");
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

    if (matrixA_shape.size() > 2 || matrixB_shape.size() > 2) {
        std::vector<int64_t> output_batch;
        std::vector<int64_t> new_shape_A(matrixA_shape);
        std::vector<int64_t> new_shape_B(matrixB_shape);
        if (matrixA_shape.size() > matrixB_shape.size()) {
            // position, count, value
            new_shape_B.insert(new_shape_B.begin(), matrixA_shape.size() - matrixB_shape.size(), 1);
        } else if (matrixA_shape.size() < matrixB_shape.size()) {
            // position, count, value
            new_shape_A.insert(new_shape_A.begin(), matrixB_shape.size() - matrixA_shape.size(), 1);
        }

        int batch_loop = (int)(std::max(matrixA_shape.size(), matrixB_shape.size()) - 2);
        for (int i = 0; i < batch_loop; ++i) {
            int64_t max_val = std::max(new_shape_A[i], new_shape_B[i]);
            output_batch.emplace_back(max_val);
        }

        std::vector<int64_t> broadcast_shape_A(output_batch);
        std::vector<int64_t> broadcast_shape_B(output_batch);
        broadcast_shape_A.insert(broadcast_shape_A.end(), matrixA_shape.end() - 2, matrixA_shape.end());
        broadcast_shape_B.insert(broadcast_shape_B.end(), matrixB_shape.end() - 2, matrixB_shape.end());

        tvm::relay::Expr output_result;
        // reshape matrix A to rank 2 and do dense operation
        if (matrixB_shape.size() == 2) {
            tvm::runtime::Array<tvm::Integer> reshape_shape_A({-1, (int)matrixA_shape[matrixA_shape.size() - 1]});
            tvm::relay::Expr reshape_A = (*reshape)(matrixA_iter->second, reshape_shape_A, true);

            tvm::runtime::Array<tvm::Integer> axes({1, 0});
            tvm::relay::Expr transpose_B = (*transpose)(matrixB_iter->second, axes);
            output_result = (*dense)(reshape_A, transpose_B, matrixB_shape[1], matrixA_dtype);
        } else {
            tvm::relay::Expr A;
            tvm::relay::Expr B;
            tvm::runtime::Array<tvm::Integer> broadcast_shape_A_relay;
            tvm::runtime::Array<tvm::Integer> broadcast_shape_B_relay;

            if (broadcast_shape_A != matrixA_shape) {
                std::for_each(broadcast_shape_A.begin(), broadcast_shape_A.end(),
                              [&](int64_t val) { broadcast_shape_A_relay.push_back((int32_t)val); });
                A = (*broadcast_to)(matrixA_iter->second, broadcast_shape_A_relay);
            }

            if (broadcast_shape_B != matrixB_shape) {
                std::for_each(broadcast_shape_B.begin(), broadcast_shape_B.end(),
                              [&](int64_t val) { broadcast_shape_B_relay.push_back((int32_t)val); });
                B = (*broadcast_to)(matrixB_iter->second, broadcast_shape_B_relay);
            }

            tvm::runtime::Array<tvm::Integer> reshape_shape_A(
                {-1, broadcast_shape_A_relay[broadcast_shape_A_relay.size() - 2],
                 broadcast_shape_A_relay[broadcast_shape_A_relay.size() - 1]});
            tvm::runtime::Array<tvm::Integer> reshape_shape_B(
                {-1, broadcast_shape_B_relay[broadcast_shape_B_relay.size() - 2],
                 broadcast_shape_B_relay[broadcast_shape_B_relay.size() - 1]});

            tvm::relay::Expr reshape_A = (*reshape)(A, reshape_shape_A, true);
            tvm::relay::Expr reshape_B = (*reshape)(B, reshape_shape_B, true);

            output_result = (*batch_matmul)(reshape_A, reshape_B, matrixA_dtype, false, false);
        }

        tvm::runtime::Array<tvm::Integer> final_shape;
        std::for_each(output_batch.begin(), output_batch.end(),
                      [&](int64_t val) { final_shape.push_back((int32_t)val); });
        final_shape.push_back(matrixA_shape[matrixA_shape.size() - 2]);
        final_shape.push_back(matrixB_shape[matrixB_shape.size() - 1]);

        tvm::relay::Expr result_expr;
        result_expr = (*reshape)(output_result, final_shape, true);

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

    if (matrixA_shape.size() == 1 || matrixB_shape.size() == 1) {
        tvm::runtime::Array<tvm::Integer> axis;
        tvm::relay::Expr lhs;
        tvm::relay::Expr rhs;
        if (matrixA_shape.size() == 1) {
            // relay, position, number of new axes
            lhs = (*expand_dims)(matrixA_iter->second, 0, 1);
            axis.push_back(0);
        } else {
            lhs = matrixA_iter->second;
        }

        tvm::relay::Expr tmp;
        if (matrixB_shape.size() == 1) {
            // relay, position, number of new axes
            rhs = (*expand_dims)(matrixB_iter->second, 1, 1);
            axis.push_back(-1);

            tmp = (*dense)(lhs, rhs, 1, matrixA_dtype);
        } else {
            rhs = matrixB_iter->second;
            tmp = (*dense)(lhs, rhs, matrixB_shape[matrixB_shape.size() - 1], matrixA_dtype);
        }

        tvm::relay::Expr result_expr;
        result_expr = (*squeeze)(tmp, axis);

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

Status MatMulParser::parse_method_2(const onnx::NodeProto& proto_node,
                                    std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                                    tvm::relay::Expr& relay) {
    // get the matmul relay function
    const tvm::runtime::PackedFunc* matmul = tvm::runtime::Registry::Get("relay.op.nn._make.matmul");
    if (!matmul) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.matmul expression not found");
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

    tvm::relay::Expr result_expr;
    result_expr = (*matmul)(matrixA_iter->second, matrixB_iter->second, matrixB_shape[matrixB_shape.size() - 1],
                            matrixA_dtype, false, false);

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
#include "gemm.h"

#include "utils/relay_utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
Status GemmParser::parse_op(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Gemm") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Gemm parameter");
    }

    // get the const generate function
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Constant expression not found");
    }

    // transpose
    const tvm::runtime::PackedFunc* transpose = tvm::runtime::Registry::Get("relay.op._make.transpose");
    if (!transpose) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.transpose expression not found");
    }

    // get the mutiply function
    const tvm::runtime::PackedFunc* mutiply = tvm::runtime::Registry::Get("relay.op._make.multiply");
    if (!mutiply) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.multiply expression not found");
    }

    // get the add function
    const tvm::runtime::PackedFunc* add = tvm::runtime::Registry::Get("relay.op._make.add");
    if (!add) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.add expression not found");
    }

    // get the dense function
    const tvm::runtime::PackedFunc* dense = tvm::runtime::Registry::Get("relay.op.nn._make.dense");
    if (!dense) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.nn._make.dense expression not found");
    }

    // get the attributes for Gemm op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    float alpha = get_attr_or_default<float>("alpha", 1.0f, attrs_map);
    float beta = get_attr_or_default<float>("beta", 1.0f, attrs_map);
    int64_t transA = get_attr_or_default<int64_t>("transA", 0, attrs_map);
    int64_t transB = get_attr_or_default<int64_t>("transB", 0, attrs_map);

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size < 2) {
        std::ostringstream oss;
        oss << "Invalid inputs of Gemm: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Gemm: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& inputA = proto_node.input(0);
    const std::string& inputB = proto_node.input(1);
    const std::string& output = proto_node.output(0);

    auto inputA_iter = expressions.find(inputA);
    if (inputA_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Gemm: " << proto_node.name() << " input: " << inputA;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    auto inputB_iter = expressions.find(inputB);
    if (inputB_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Gemm: " << proto_node.name() << " input: " << inputB;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the matrix B shape
    std::vector<int64_t> matrixB_shape;
    tvm::DataType matrixB_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(inputB_iter->second, matrixB_shape, matrixB_dtype);
    int channels = transB ? matrixB_shape[0] : matrixB_shape[1];

    tvm::runtime::Array<tvm::Integer> axes({1, 0});

    // matrix A
    tvm::relay::Expr matrixA = inputA_iter->second;
    // transpose matrix A
    if (transA) {
        matrixA = (*transpose)(matrixA, axes);
    }

    // matrix B
    tvm::relay::Expr matrixB = inputB_iter->second;
    // transpose matrix B
    if (!transB) {
        matrixB = (*transpose)(matrixB, axes);
    }

    // A = alpha * A
    if (std::abs(alpha - 1.0f) > 1e-6) {
        tvm::runtime::NDArray alpha_nd =
            tvm::runtime::NDArray::Empty({}, {DLDataTypeCode::kDLFloat, 32, 1}, {DLDeviceType::kDLCPU, 0});
        static_cast<float*>(alpha_nd->data)[0] = alpha;

        // generate the const
        tvm::relay::Constant alpha_expr = (*const_gen)(alpha_nd, tvm::relay::Span());

        matrixA = (*mutiply)(matrixA, alpha_expr);
    }

    tvm::relay::Expr result_expr;
    // out = A * B
    result_expr = (*dense)(matrixA, matrixB, channels, tvm::DataType());

    // C exists
    if (input_size == 3) {
        const std::string& inputC = proto_node.input(2);
        auto inputC_iter = expressions.find(inputC);
        if (inputC_iter == expressions.end()) {
            std::ostringstream oss;
            oss << "Input not found, Gemm: " << proto_node.name() << " input: " << inputC;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }
        // matrix C
        tvm::relay::Expr matrixC = inputC_iter->second;

        // out += beta * C
        if (std::abs(beta - 1.0f) > 1e-6) {
            tvm::runtime::NDArray beta_nd =
                tvm::runtime::NDArray::Empty({}, {DLDataTypeCode::kDLFloat, 32, 1}, {DLDeviceType::kDLCPU, 0});
            static_cast<float*>(beta_nd->data)[0] = beta;

            // generate the const
            tvm::relay::Constant beta_expr = (*const_gen)(beta_nd, tvm::relay::Span());

            // C = beta * C
            matrixC = (*mutiply)(matrixC, beta_expr);
        }

        // out += C
        result_expr = (*add)(result_expr, matrixC);
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

std::string GemmParser::get_name() { return "Gemm"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
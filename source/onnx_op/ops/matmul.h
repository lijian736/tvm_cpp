#ifndef _H_TVM_CPP_ONNX_OP_MATMUL_PARSER_H_
#define _H_TVM_CPP_ONNX_OP_MATMUL_PARSER_H_

#include "onnx_op/op_parser.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
class MatMulParser : public IOnnxOpParser {
public:
    MatMulParser() = default;
    virtual ~MatMulParser() = default;

    virtual std::string get_name() override;
    virtual Status parse_op(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                            tvm::relay::Expr& relay) override;

private:
    Status parse_method_1(const onnx::NodeProto& proto_node, std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                    tvm::relay::Expr& relay);

    Status parse_method_2(const onnx::NodeProto& proto_node, std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                    tvm::relay::Expr& relay);          
};

}    // namespace onnx_op
}    // namespace tvm_cpp

#endif
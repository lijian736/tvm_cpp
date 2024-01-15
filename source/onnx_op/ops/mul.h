#ifndef _H_TVM_CPP_ONNX_OP_MUL_PARSER_H_
#define _H_TVM_CPP_ONNX_OP_MUL_PARSER_H_

#include "onnx_op/op_parser.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
class MulParser : public IOnnxOpParser {
public:
    MulParser() = default;
    virtual ~MulParser() = default;

    virtual Status parse_op(const onnx::NodeProto& proto_node,
                            const std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                            tvm::relay::Expr& relay) override;
};

}    // namespace onnx_op
}    // namespace tvm_cpp

#endif
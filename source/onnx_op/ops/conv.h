#ifndef _H_TVM_CPP_ONNX_OP_CONV_PARSER_H_
#define _H_TVM_CPP_ONNX_OP_CONV_PARSER_H_

#include "onnx_op/op_parser.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
class ConvParser : public IOnnxOpParser {
public:
    ConvParser() = default;
    virtual ~ConvParser() = default;

    virtual std::string get_name() override;
    virtual Status parse_op(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                            tvm::relay::Expr& relay) override;
};

}    // namespace onnx_op
}    // namespace tvm_cpp

#endif
#ifndef _H_TVM_CPP_ONNX_OP_MAX_POOL_PARSER_H_
#define _H_TVM_CPP_ONNX_OP_MAX_POOL_PARSER_H_

#include "onnx_op/op_parser.h"

namespace tvm_cpp {
namespace onnx_op {

class MaxPool2DParser : public IOnnxOpParser {
public:
    MaxPool2DParser() = default;
    virtual ~MaxPool2DParser() = default;

    virtual Status parse_op(const onnx::NodeProto& proto_node,
                            const std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                            tvm::relay::Expr& relay) override;
};

}    // namespace onnx_op
}    // namespace tvm_cpp

#endif
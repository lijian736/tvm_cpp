#ifndef _H_TVM_CPP_ONNX_OP_OP_PARSER_H_
#define _H_TVM_CPP_ONNX_OP_OP_PARSER_H_

#include <tvm/relay/expr.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>

#include "onnx.proto3.pb.h"
#include "utils/status.h"

namespace tvm_cpp {
namespace onnx_op {

/**
 * @brief The interface for parsing ONNX ops
 * 
 */
class IOnnxOpParser {
public:
    IOnnxOpParser() = default;
    virtual ~IOnnxOpParser() = default;

    /**
     * @brief Parse the ONNX op to relay expression
     * 
     * @param proto_node the ONNX op proto
     * @param relay output parameter. the relay expression
     * @return Status 
     */
    virtual Status parse_op(const onnx::NodeProto& proto_node, tvm::relay::Expr& relay) = 0;
};

}
}    // namespace tvm_cpp

#endif
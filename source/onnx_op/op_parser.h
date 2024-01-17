#ifndef _H_TVM_CPP_ONNX_OP_OP_PARSER_H_
#define _H_TVM_CPP_ONNX_OP_OP_PARSER_H_

#include <tvm/relay/expr.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>

#include "onnx.proto3.pb.h"
#include "utils/onnx_utils.h"
#include "utils/status.h"

using namespace tvm_cpp::onnx_utils;

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
     * @param expressions input/output parameter. the relay expression map. key: the relay expression name. value: the relay expression
     * @param relay output parameter. the relay expression
     * @return Status
     */
    virtual Status parse_op(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                            tvm::relay::Expr& relay) = 0;

    /**
     * @brief Get the proto node attributes map. 
     * If there are multiple attributes with the same name, the last one will be inserted into the map
     * 
     * @param proto_node the proto node
     * @param attrs_map the attributes map. key: the attribute name. value: the attribute pointer
     */
    void get_attributes_map(const onnx::NodeProto& proto_node, std::unordered_map<std::string, const onnx::AttributeProto*>& attrs_map);
};

}    // namespace onnx_op
}    // namespace tvm_cpp

#endif
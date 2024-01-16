#ifndef _H_TVM_CPP_UTILS_RELAY_UTILS_H_
#define _H_TVM_CPP_UTILS_RELAY_UTILS_H_

#include <tvm/relay/expr.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>

#include "onnx.proto3.pb.h"
#include "status.h"

namespace tvm_cpp {
namespace relay_utils {

/**
 * @brief Convert the ONNX initializer to TVM relay constant expression
 *
 * @param gen_func the TVM Constant generator function
 * @param proto_tensor the ONNX tensor proto
 * @param relay output parameter. the generated relay expression
 * @return Status
 */
Status convert_initializer_to_relay(const tvm::runtime::PackedFunc* gen_func, const onnx::TensorProto& proto_tensor,
                                    tvm::relay::Expr& relay);

/**
 * @brief Parse the graph proto initializers to TVM relay expressions
 *
 * @param onnx_graph onnx graph proto
 * @param relays the relay map. key: the initializer name, value: the relay expression
 * @return Status
 */
Status parse_graph_initializers_to_relays(const onnx::GraphProto& onnx_graph,
                                          std::unordered_map<std::string, tvm::relay::Expr>& relays);

/**
 * @brief Parse the graph proto inputs to TVM relay expressions
 *
 * @param onnx_graph onnx graph proto
 * @param relays the relay map. key: the input name, value: the relay expression
 * @return Status
 */
Status parse_graph_inputs_to_relays(const onnx::GraphProto& onnx_graph,
                                    std::unordered_map<std::string, tvm::relay::Expr>& relays);

/**
 * @brief Convert the ONNX node proto to TVM relay expressions
 *
 * @param proto_node the ONNX node proto
 * @param relay output parameter. the generated relay expression
 * @return Status
 */
Status convert_node_to_relay(const onnx::NodeProto& proto_node, tvm::relay::Expr& relay);

}    // namespace relay_utils
}    // namespace tvm_cpp

#endif
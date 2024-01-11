#ifndef _H_TVM_CPP_UTILS_ONNX_UTILS_H_
#define _H_TVM_CPP_UTILS_ONNX_UTILS_H_

#include <string>
#include <unordered_map>

#include "onnx.proto3.pb.h"

namespace tvm_cpp {
namespace onnx_utils {

/**
 * @brief validate the onnx proto model
 *
 * @param model the onnx proto model
 * @return true
 * @return false
 */
bool validate_onnx_proto(const onnx::ModelProto& model);

/**
 * @brief print onnx proto model info
 *
 * @param model the onnx proto model
 */
void print_onnx_model_info(const onnx::ModelProto& model);

/**
 * @brief Retrieve onnx node types map. map key: the op type, value: the op type counter
 *
 * @param model the onnx proto model
 * @param types_map output parameter. the onnx node type map
 */
void retrieve_onnx_node_types(const onnx::ModelProto& model, std::unordered_map<std::string, int>& types_map);

}    // namespace onnx_utils
}    // namespace tvm_cpp

#endif
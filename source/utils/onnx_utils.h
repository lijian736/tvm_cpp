#ifndef _H_TVM_CPP_UTILS_ONNX_UTILS_H_
#define _H_TVM_CPP_UTILS_ONNX_UTILS_H_

#include <string>
#include <unordered_map>

#include "onnx.proto3.pb.h"
#include "status.h"

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

/**
 * @brief Get a single attribute
 *
 * @tparam T
 * @param name the attribute name
 * @param value output parameter. the returned attribute value
 * @param attributes the attributes map. key: attribute name, value: attribute proto
 * @return Status if the attribute does NOT exist in the `attributes` or the attribute data type mismatch, return fail.
 */
template <typename T>
Status get_attr(const std::string& name, T* value,
                const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes);

template <>
inline Status get_attr<float>(const std::string& name, float* value,
                              const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != onnx::AttributeProto_AttributeType_FLOAT) {
        return Status(StatusCode::FAIL);
    }

    *value = iter->second->f();
    return Status::ok();
}

template <>
inline Status get_attr<int64_t>(const std::string& name, int64_t* value,
                                const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != onnx::AttributeProto_AttributeType_INT) {
        return Status(StatusCode::FAIL);
    }

    *value = iter->second->i();
    return Status::ok();
}

template <>
inline Status get_attr<std::string>(const std::string& name, std::string* value,
                                    const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != onnx::AttributeProto_AttributeType_STRING) {
        return Status(StatusCode::FAIL);
    }

    *value = iter->second->s();
    return Status::ok();
}

/**
 * @brief Get the attrs vector
 *
 * @tparam T
 * @param name the attribute name
 * @param values output parameter. the returned attributes
 * @param attributes the attributes
 * @return Status
 */
template <typename T>
Status get_attrs(const std::string& name, std::vector<T>& values,
                 const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes);

template <>
inline Status get_attrs<float>(const std::string& name, std::vector<float>& values,
                               const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != onnx::AttributeProto_AttributeType_FLOATS) {
        return Status(StatusCode::FAIL);
    }

    values.assign(iter->second->floats().begin(), iter->second->floats().end());
    return Status::ok();
}

template <>
inline Status get_attrs<int64_t>(const std::string& name, std::vector<int64_t>& values,
                                 const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != onnx::AttributeProto_AttributeType_INTS) {
        return Status(StatusCode::FAIL);
    }

    values.assign(iter->second->ints().begin(), iter->second->ints().end());
    return Status::ok();
}

template <>
inline Status get_attrs<std::string>(const std::string& name, std::vector<std::string>& values,
                                     const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != onnx::AttributeProto_AttributeType_STRINGS) {
        return Status(StatusCode::FAIL);
    }

    values.assign(iter->second->strings().begin(), iter->second->strings().end());
    return Status::ok();
}

/**
 * @brief Get the attr or default value
 *
 * @tparam T
 * @param name the attribute name
 * @param default_value the default attribute value
 * @param attributes the attributes
 * @return T if the attribute does NOT exist in the `attributes` or the attribute data type mismatch, return the default
 * value. otherwise return the attribute value
 */
template <typename T>
inline T get_attr_or_default(const std::string& name, const T& default_value,
                             const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    T tmp;
    return get_attr<T>(name, &tmp, attributes).is_ok() ? tmp : default_value;
}

/**
 * @brief Get the attr or default value
 *
 * @tparam T
 * @param name the attribute name
 * @param value output parameter. the attribute value
 * @param default_value the default attribute value
 * @param attributes the attributes
 */
template <typename T>
inline void get_attr_or_default(const std::string& name, T* value, const T& default_value,
                                const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    if (!get_attr<T>(name, value, attributes).is_ok()) {
        *value = default_value;
    }
}

template <typename T>
inline std::vector<T> get_attrs_or_default(
    const std::string& name, const std::vector<T>& default_value,
    const std::unordered_map<std::string, const onnx::AttributeProto*>& attributes) {
    std::vector<T> tmp;
    return get_attrs<T>(name, tmp, attributes).is_ok() ? tmp : default_value;
}

}    // namespace onnx_utils
}    // namespace tvm_cpp

#endif
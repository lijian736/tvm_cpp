#ifndef _H_TVM_CPP_UTILS_UTILS_H_
#define _H_TVM_CPP_UTILS_UTILS_H_

#include <string>
#include <unordered_map>

#include "onnx.proto3.pb.h"

namespace tvm_cpp {
namespace utils {

/**
 * @brief trim the string start
 *
 * @param str input/output parameter
 */
void trim_start(std::string& str);

/**
 * @brief trim the string end
 *
 * @param str input/output parameter
 */
void trim_end(std::string& str);

/**
 * @brief trim the string
 *
 * @param str input/output parameter
 */
void trim(std::string& str);

/**
 * @brief string ends with the suffix
 *
 * @param str the string
 * @param suffix suffix
 * @return true
 * @return false
 */
bool ends_with(const std::string& str, const std::string& suffix);

/**
 * @brief check if the file path exists
 *
 * @param file_path the file path
 * @return true
 * @return false
 */
bool file_exist(const std::string& file_path);

}    // namespace utils
}    // namespace tvm_cpp

#endif
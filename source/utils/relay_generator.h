
#ifndef _H_TVM_CPP_UTILS_RELAY_GENERATOR_H_
#define _H_TVM_CPP_UTILS_RELAY_GENERATOR_H_

#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>

#include "status.h"

namespace tvm_cpp {
namespace relay_generator {

/**
 * @brief Generate an IRModule with dead codes
 *
 * @param module output parameter. the generated IRModule with dead codes
 * @return Status
 */
Status generate_dead_code_module(tvm::IRModule& module);

/**
 * @brief Generate an IRModule for op fusing
 *
 * @param module output parameter. the generated IRModule
 * @return Status
 */
Status generate_fuse_op_module(tvm::IRModule& module);

/**
 * @brief Generate an IRModule for common sub expressions
 *
 * @param module output parameter. the generated IRModule
 * @return Status
 */
Status generate_common_subexp_module(tvm::IRModule& module);

/**
 * @brief Generate an IRModule for remove-ununsed-functions pass
 *
 * @param module output parameter. the generated IRModule
 * @return Status
 */
Status generate_remove_unused_fun_module(tvm::IRModule& module);

/**
 * @brief Generate an relay for traverse pass
 * 
 * @param expr output parameter. the generated expr
 * @return Status
 */
Status generate_traverse_expr(tvm::relay::Expr& expr);

}    // namespace relay_generator
}    // namespace tvm_cpp

#endif
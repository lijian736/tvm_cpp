#ifndef _H_TVM_CPP_ONNX_OP_OP_PARSER_H_
#define _H_TVM_CPP_ONNX_OP_OP_PARSER_H_

#include <tvm/relay/expr.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <type_traits>
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
     * @brief Get the op type name
     *
     * @return std::string
     */
    virtual std::string get_name() = 0;

    /**
     * @brief Parse the ONNX op to relay expression
     *
     * @param proto_node the ONNX op proto
     * @param expressions input/output parameter. the relay expression map. key: the relay expression name. value: the
     * relay expression
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
    void get_attributes_map(const onnx::NodeProto& proto_node,
                            std::unordered_map<std::string, const onnx::AttributeProto*>& attrs_map);

    /**
     * @brief fold const relay expressions
     *
     * @param expr input/output parameter. the relay
     * @return Status
     */
    virtual Status fold_const(tvm::relay::Expr& expr);
};

/**
 * @brief the ONNX ops parser register center. A singleton class
 *
 */
class OnnxOpParserRegister {
public:
    ~OnnxOpParserRegister() = default;

    static OnnxOpParserRegister* get_instance();

    /**
     * @brief Get the op parser pointer
     *
     * @param op_type the op type, e.g. "Conv"
     * @return IOnnxOpParser* the op parser pointer or nullptr
     */
    IOnnxOpParser* get_op_parser(const std::string& op_type);

private:
    OnnxOpParserRegister();

    /**
     * @brief Register all supported ops
     */
    void register_all_supported_ops();

    template <typename T, typename = std::enable_if_t<std::is_base_of_v<tvm_cpp::onnx_op::IOnnxOpParser, T>>>
    void register_op() {
        auto val = std::make_unique<T>();
        auto ret = m_op_parsers_map.emplace(val->get_name(), nullptr);
        ret.first->second = std::move(val);
    }

private:
    // key: the op type, value: the op parser ptr
    std::unordered_map<std::string, std::unique_ptr<IOnnxOpParser>> m_op_parsers_map;
};

}    // namespace onnx_op
}    // namespace tvm_cpp

#endif
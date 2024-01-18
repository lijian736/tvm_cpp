#include "op_parser.h"

#include "ops/add.h"
#include "ops/concat.h"
#include "ops/conv.h"
#include "ops/max_pool.h"
#include "ops/mul.h"
#include "ops/relu.h"
#include "ops/reshape.h"
#include "ops/resize.h"
#include "ops/transpose.h"

namespace tvm_cpp {
namespace onnx_op {

void IOnnxOpParser::get_attributes_map(const onnx::NodeProto& proto_node,
                                       std::unordered_map<std::string, const onnx::AttributeProto*>& attrs_map) {
    int attr_size = proto_node.attribute_size();
    for (int i = 0; i < attr_size; ++i) {
        const auto& attr = proto_node.attribute(i);
        const auto& attr_name = attr.name();
        auto ret = attrs_map.emplace(attr_name, &attr);
        if (!ret.second) {
            ret.first->second = &attr;
        }
    }
}

Status IOnnxOpParser::fold_const(tvm::relay::Expr& expr) {
    // get the fold const relay function
    const tvm::runtime::PackedFunc* fold = tvm::runtime::Registry::Get("relay._transform.FoldConstantExpr");
    if (!fold) {
        return Status(StatusCode::RUNTIME_ERROR, "relay._transform.FoldConstantExpr expression not found");
    }

    expr = (*fold)(expr, tvm::IRModule(), false);

    return Status::ok();
}

OnnxOpParserRegister* OnnxOpParserRegister::get_instance() {
    static OnnxOpParserRegister instance;
    return &instance;
}

IOnnxOpParser* OnnxOpParserRegister::get_op_parser(const std::string& op_type) {
    auto iter = m_op_parsers_map.find(op_type);
    if (iter != m_op_parsers_map.end()) {
        return iter->second.get();
    }

    return nullptr;
}

void OnnxOpParserRegister::register_all_supported_ops() {
    this->register_op<Conv2DParser>();
    this->register_op<AddParser>();
    this->register_op<ConcatParser>();
    this->register_op<MaxPool2DParser>();
    this->register_op<MulParser>();
    this->register_op<ReluParser>();
    this->register_op<ReshapeParser>();
    this->register_op<Resize2DParser>();
    this->register_op<TransposeParser>();
}

}    // namespace onnx_op
}    // namespace tvm_cpp
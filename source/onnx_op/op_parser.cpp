#include "op_parser.h"

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

}    // namespace onnx_op
}    // namespace tvm_cpp
#include "onnx_utils.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

#include "utils.h"

namespace tvm_cpp {
namespace onnx_utils {

bool validate_onnx_proto(const onnx::ModelProto& model) {
    bool has_graph = model.has_graph();
    if (!has_graph) {
        std::cerr << "no graph in onnx model" << std::endl;
        return false;
    }

    if (model.opset_import_size() == 0) {
        std::cerr << "opset_import missed in onnx model" << std::endl;
        return false;
    }

    if (!onnx::Version_IsValid(model.ir_version())) {
        std::cerr << "unsupported model IR version: " << model.ir_version() << std::endl;
        return false;
    }

    if (model.ir_version() < 4) {
        std::cerr << "Too old ir version: " << model.ir_version() << ", not supported now" << std::endl;
        return false;
    }

    return true;
}

void print_onnx_model_info(const onnx::ModelProto& model) {
    // print model basic info
    std::cout << "model basic info:" << std::endl;

    std::cout << "ir version: " << model.ir_version() << std::endl;
    std::cout << "producer name: " << model.producer_name() << std::endl;
    std::cout << "producer version: " << model.producer_version() << std::endl;
    std::cout << "domain: " << model.domain() << std::endl;
    std::cout << "model version: " << model.model_version() << std::endl;
    std::cout << "model doc: " << model.doc_string() << std::endl;

    // print metadata props
    std::cout << "model metadata:" << std::endl;
    for (auto& prop : model.metadata_props()) {
        std::cout << prop.key() << " : " << prop.value() << std::endl;
    }

    // print opset
    std::cout << "opset" << std::endl;
    for (auto& opset : model.opset_import()) {
        const auto& domain = opset.domain();
        const auto version = opset.version();
        std::cout << "domain: " << domain << " version: " << version << std::endl;
    }
}

void retrieve_onnx_node_types(const onnx::ModelProto& model, std::unordered_map<std::string, int>& types_map) {
    const auto& graph = model.graph();
    for (const auto& node : graph.node()) {
        auto ret = types_map.emplace(node.op_type(), 1);
        if (!ret.second) {
            ret.first->second = ret.first->second + 1;
        }
    }
}

}    // namespace onnx_utils
}    // namespace tvm_cpp
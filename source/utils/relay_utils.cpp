#include "relay_utils.h"

#include <vector>

namespace tvm_cpp {
namespace relay_utils {

Status convert_initializer_to_relay(const tvm::runtime::PackedFunc* gen_func, const onnx::TensorProto& proto_tensor,
                                    tvm::relay::Expr& relay) {
    if (!gen_func) {
        return Status(StatusCode::INVALID_PARAM, "gen_func is nullptr");
    }

    // check the tensor data location
    if (proto_tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
        std::ostringstream oss;
        oss << "external tensor data is not supported yet, tensor name: " << proto_tensor.name();
        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
    }

    // the tensor shape
    std::vector<int64_t> tensor_shape;
    // the tensor element number
    int64_t element_num = 1;

    // add dims to the tensor shape
    int dim_size = proto_tensor.dims_size();
    for (int i = 0; i < dim_size; ++i) {
        int64_t dim = proto_tensor.dims(i);
        tensor_shape.emplace_back(dim);
        element_num *= (dim > 0 ? dim : 1);
    }

    // It's a good idea to generate empty NDArray using PackedFunc 'runtime.TVMArrayAllocWithScope'
    // shape, data type, device, only float data type supported now
    tvm::runtime::NDArray initializer = tvm::runtime::NDArray::Empty(
        tvm::runtime::ShapeTuple(tensor_shape), {DLDataTypeCode::kDLFloat, 32, 1}, {DLDeviceType::kDLCPU, 0});

    switch (proto_tensor.data_type()) {
        case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: {
            if (proto_tensor.raw_data().length() > 0) {
                // TODO, check if the current CPU bytes order is little endian
                if (proto_tensor.raw_data().length() == sizeof(float) * element_num) {
                    const char* tensor_data = proto_tensor.raw_data().data();

                    // copy the data
                    initializer.CopyFromBytes(tensor_data, proto_tensor.raw_data().length());
                } else {
                    std::ostringstream oss;
                    oss << "Invalid tensor float data length with its dims, tensor name: " << proto_tensor.name();
                    return Status(StatusCode::INVALID_MODEL, oss.str());
                }
            } else {
                std::vector<float> float_data(element_num, 0);
                if (proto_tensor.float_data_size() == element_num) {
                    for (int i = 0; i < proto_tensor.float_data_size(); ++i) {
                        float_data[i] = proto_tensor.float_data(i);
                    }

                    // copy the data
                    initializer.CopyFromBytes(float_data.data(), proto_tensor.float_data_size() * sizeof(float));
                } else {
                    std::ostringstream oss;
                    oss << "Invalid tensor float data length with its dims, tensor name: " << proto_tensor.name();
                    return Status(StatusCode::INVALID_MODEL, oss.str());
                }
            }
        }

        default: {
            std::ostringstream oss;
            oss << "not support data type for proto tensor, tensor name: " << proto_tensor.name();
            return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
        }
    }

    tvm::relay::Span span;
    relay = (*gen_func)(initializer, span);

    return Status::ok();
}

Status parse_graph_initializers_to_relays(const onnx::GraphProto& onnx_graph,
                                          std::unordered_map<std::string, tvm::relay::Expr>& relays) {
    // Constant in relay
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Constant not found");
    }

    for (auto& initializer : onnx_graph.initializer()) {
        tvm::relay::Expr relay_const;
        auto ret = convert_initializer_to_relay(const_gen, initializer, relay_const);
        if (!ret.is_ok()) {
            return ret;
        }

        // add. if the initializer has already existed in the graph, replace it.
        auto insert_ret = relays.emplace(initializer.name(), relay_const);
        if (!insert_ret.second) {
            insert_ret.first->second = std::move(relay_const);
        }
    }

    return Status::ok();
}

Status parse_graph_inputs_to_relays(const onnx::GraphProto& onnx_graph,
                                    std::unordered_map<std::string, tvm::relay::Expr>& relays) {
    // Var node in relay
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Var expression not found");
    }

    // iterate the graph inputs
    for (const auto& input : onnx_graph.input()) {
        // the graph input name
        const auto& input_name = input.name();
        if (!input_name.empty()) {
            // graph input type
            const onnx::TypeProto& type = input.type();

            if (type.value_case() == onnx::TypeProto::ValueCase::kTensorType) {
                const onnx::TypeProto_Tensor tensor_type = type.tensor_type();
                // element type
                auto elem_type = tensor_type.elem_type();
                if (elem_type != onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
                    return Status(StatusCode::NOT_IMPLEMENTED, "Graph float input is not implemented");
                }

                const onnx::TensorShapeProto shape_proto = tensor_type.shape();

                tvm::runtime::Array<tvm::PrimExpr> shape;
                for (const auto& dim : shape_proto.dim()) {
                    if (dim.value_case() == onnx::TensorShapeProto_Dimension::ValueCase::kDimValue) {
                        int64_t dim_val = dim.dim_value();
                        shape.push_back((int32_t)dim_val);
                    } else {
                        shape.push_back(-1);
                    }
                }

                // generate the var expression
                // the data type
                DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
                // the tensor type
                tvm::relay::TensorType var_type{shape, tvm::DataType{data_type}};

                // the var expression
                tvm::relay::Expr var_expr = (*var_gen)(input_name, var_type, tvm::relay::Span());
                auto ret = relays.emplace(input_name, var_expr);
                if (!ret.second) {
                    ret.first->second = std::move(var_expr);
                }
            } else {
                std::ostringstream oss;
                oss << "Graph input [" << input_name << "] has no type or has an unsupported type";
                return Status(StatusCode::INVALID_MODEL, oss.str());
            }
        } else {
            return Status(StatusCode::INVALID_MODEL, "Graph input name is empty");
        }
    }

    return Status::ok();
}

Status convert_node_to_relay(const onnx::NodeProto& proto_node, tvm::relay::Expr& relay) { return Status::ok(); }

}    // namespace relay_utils
}    // namespace tvm_cpp
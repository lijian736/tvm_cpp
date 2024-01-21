#include "relay_utils.h"

#include <vector>

#include "onnx_op/op_parser.h"
#include "utils.h"

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
    tvm::runtime::NDArray initializer;

    switch (proto_tensor.data_type()) {
        case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: {
            initializer = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple(tensor_shape),
                                                       {DLDataTypeCode::kDLFloat, 32, 1}, {DLDeviceType::kDLCPU, 0});

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

            break;
        }

        case onnx::TensorProto_DataType::TensorProto_DataType_INT64: {
            initializer = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple(tensor_shape),
                                                       {DLDataTypeCode::kDLInt, 64, 1}, {DLDeviceType::kDLCPU, 0});

            if (proto_tensor.raw_data().length() > 0) {
                if (proto_tensor.raw_data().length() == sizeof(int64_t) * element_num) {
                    const char* tensor_data = proto_tensor.raw_data().data();

                    // copy the data
                    initializer.CopyFromBytes(tensor_data, proto_tensor.raw_data().length());
                } else {
                    std::ostringstream oss;
                    oss << "Invalid tensor int64 data length with its dims, tensor name: " << proto_tensor.name();
                    return Status(StatusCode::INVALID_MODEL, oss.str());
                }
            } else {
                std::vector<int64_t> int64_data(element_num, 0);
                if (proto_tensor.int64_data_size() == element_num) {
                    for (int i = 0; i < proto_tensor.int64_data_size(); ++i) {
                        int64_data[i] = proto_tensor.int64_data(i);
                    }

                    // copy the data
                    initializer.CopyFromBytes(int64_data.data(), proto_tensor.int64_data_size() * sizeof(int64_t));
                } else {
                    std::ostringstream oss;
                    oss << "Invalid tensor int64 data length with its dims, tensor name: " << proto_tensor.name();
                    return Status(StatusCode::INVALID_MODEL, oss.str());
                }
            }
            break;
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

    for (const auto& initializer : onnx_graph.initializer()) {
        tvm::relay::Expr relay_const;
        auto ret = convert_initializer_to_relay(const_gen, initializer, relay_const);
        if (!ret.is_ok()) {
            return ret;
        }

        std::string initializer_name = initializer.name();
        tvm_cpp::utils::trim(initializer_name);

        // add. if the initializer has already existed in the graph, replace it.
        auto insert_ret = relays.emplace(initializer_name, relay_const);
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
        auto input_name = input.name();
        tvm_cpp::utils::trim(input_name);

        if (!input_name.empty()) {
            // graph input type
            const onnx::TypeProto& type = input.type();

            if (type.value_case() == onnx::TypeProto::ValueCase::kTensorType) {
                const onnx::TypeProto_Tensor tensor_type = type.tensor_type();
                // element type
                auto elem_type = tensor_type.elem_type();
                if (elem_type != onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
                    return Status(StatusCode::NOT_IMPLEMENTED, "Only float graph input is supporetd now.");
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

Status infer_relay_shape_dtype(const tvm::relay::Expr& expr, std::vector<int64_t>& shape, tvm::DataType& dtype) {
    // type infer
    const tvm::runtime::PackedFunc* type_infer = tvm::runtime::Registry::Get("relay._transform.InferType");
    if (!type_infer) {
        return Status(StatusCode::RUNTIME_ERROR, "relay._transform.InferType expression not found");
    }

    // pass run
    const tvm::runtime::PackedFunc* pass_run = tvm::runtime::Registry::Get("transform.RunPass");
    if (!pass_run) {
        return Status(StatusCode::RUNTIME_ERROR, "transform.pass_run expression not found");
    }

    tvm::IRModule mod = tvm::IRModule::FromExpr(expr);

    // get the type-infer pass
    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    // run the pass
    tvm::IRModule mod_new = (*pass_run)(infer_type_pass, mod);

    tvm::relay::Expr main_expr = mod_new->Lookup("main").as<tvm::relay::FunctionNode>()->body;
    tvm::Type result_type = main_expr->checked_type();
    tvm::runtime::Optional<tvm::TensorType> type = result_type.as<tvm::TensorType>();
    if (type != nullptr) {
        tvm::TensorType tensor_type = type.value();

        const tvm::DataType& data_type = tensor_type->dtype;
        dtype = data_type;
        const tvm::runtime::Array<tvm::PrimExpr>& expr_shape = tensor_type->shape;

        for (int i = 0; i < expr_shape.size(); ++i) {
            const tvm::PrimExpr& exp = expr_shape[i];
            const tvm::IntImmNode* node = exp.as<tvm::IntImmNode>();
            if (node) {
                shape.emplace_back(node->value);
            } else {
                return Status(StatusCode::RUNTIME_ERROR, "Invalid shape");
            }
        }
    }

    return Status::ok();
}

Status infer_relay_shape(const tvm::relay::Expr& expr, tvm::relay::Expr& relay) {
    // shape of
    const tvm::runtime::PackedFunc* shape_of = tvm::runtime::Registry::Get("relay.op._make.shape_of");
    if (!shape_of) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.shape_of expression not found");
    }

    relay = (*shape_of)(expr, tvm::runtime::DataType());
    return Status::ok();
}

Status parse_graph_nodes_to_relays(const onnx::GraphProto& onnx_graph,
                                   std::unordered_map<std::string, tvm::relay::Expr>& relays) {
    tvm::relay::Expr expr;
    // iterate the graph nodes
    for (const auto& node_prot : onnx_graph.node()) {
        // the graph node name
        auto& node_name = node_prot.name();
        auto& node_type = node_prot.op_type();

        auto regist = tvm_cpp::onnx_op::OnnxOpParserRegister::get_instance();
        auto parser = regist->get_op_parser(node_type);

        if (!parser) {
            std::ostringstream oss;
            oss << "Graph node type [" << node_type << "] has no registered parsers";
            return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
        }

        auto ret = parser->parse_op(node_prot, relays, expr);
        if (!ret.is_ok()) {
            return ret;
        }
    }

    return Status::ok();
}

Status parse_graph_to_irmodule(const onnx::GraphProto& onnx_graph, tvm::IRModule& module) {
    std::unordered_map<std::string, tvm::relay::Expr> input_relays;
    std::unordered_map<std::string, tvm::relay::Expr> initializer_relays;

    // get graph inputs relays
    Status status = parse_graph_inputs_to_relays(onnx_graph, input_relays);
    if (!status.is_ok()) {
        return status;
    }

    // get graph initializers relays
    status = parse_graph_initializers_to_relays(onnx_graph, initializer_relays);
    if (!status.is_ok()) {
        return status;
    }

    // merge the initializers and inputs. if some input is an initializer of the graph,
    // keep the initializer
    std::unordered_map<std::string, tvm::relay::Expr> all_relays;
    all_relays.merge(initializer_relays);
    all_relays.merge(input_relays);

    // some input exists as an intializers in the graph
    if (input_relays.size() > 0) {
        // LOG
    }

    status = parse_graph_nodes_to_relays(onnx_graph, all_relays);
    if (!status.is_ok()) {
        return status;
    }

    tvm::relay::Expr all_output;
    int output_size = onnx_graph.output_size();
    if (output_size > 1) {
        // get the tuple relay
        const tvm::runtime::PackedFunc* tuple = tvm::runtime::Registry::Get("relay.ir.Tuple");
        if (!tuple) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Tuple expression not found");
        }

        // the output relays
        tvm::runtime::Array<tvm::relay::Expr> output_array;
        for (int i = 0; i < output_size; ++i) {
            const auto& output = onnx_graph.output(i);
            const auto& output_name = output.name();
            auto relay_iter = all_relays.find(output_name);
            if (relay_iter == all_relays.end()) {
                std::ostringstream oss;
                oss << "Graph output [" << output_name << "] has no related relay";
                return Status(StatusCode::INVALID_MODEL, oss.str());
            }

            output_array.push_back(relay_iter->second);
        }

        all_output = (*tuple)(output_array, tvm::relay::Span());
    } else {
        const auto& output_name = onnx_graph.output(0).name();
        auto relay_iter = all_relays.find(output_name);
        if (relay_iter == all_relays.end()) {
            std::ostringstream oss;
            oss << "Graph output [" << output_name << "] has no related relay";
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        all_output = relay_iter->second;
    }

    tvm::runtime::Array<tvm::relay::Expr> all_input;
    int input_size = onnx_graph.input_size();
    for (int i = 0; i < input_size; ++i) {
        const auto& input = onnx_graph.input(i);
        const auto& input_name = input.name();
        auto relay_iter = all_relays.find(input_name);
        if (relay_iter == all_relays.end()) {
            std::ostringstream oss;
            oss << "Graph output [" << input_name << "] has no related relay";
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        all_input.push_back(relay_iter->second);
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Function expression not found");
    }

    tvm::relay::Expr func = (*function)(all_input, all_output);
    module = tvm::IRModule::FromExpr(func);

    return Status::ok();
}

}    // namespace relay_utils
}    // namespace tvm_cpp
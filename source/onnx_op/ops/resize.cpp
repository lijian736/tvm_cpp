#include "resize.h"

#include "utils/relay_utils.h"
#include "utils/utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
Status ResizeParser::parse_op(const onnx::NodeProto& proto_node,
                              std::unordered_map<std::string, tvm::relay::Expr>& expressions, tvm::relay::Expr& relay) {
    // check the op type
    if (proto_node.op_type() != "Resize") {
        return Status(StatusCode::INVALID_PARAM, "Invalid Resize parameter");
    }

    // get the Resize2D relay function
    const tvm::runtime::PackedFunc* resize2d = tvm::runtime::Registry::Get("relay.op.image._make.resize2d");
    if (!resize2d) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op.image._make.resize2d expression not found");
    }

    // strided
    const tvm::runtime::PackedFunc* strided_slice = tvm::runtime::Registry::Get("relay.op._make.strided_slice");
    if (!strided_slice) {
        return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.strided_slice expression not found");
    }

    // get the attributes for Resize 2d op
    std::unordered_map<std::string, const onnx::AttributeProto*> attrs_map;
    get_attributes_map(proto_node, attrs_map);

    int64_t antialias = get_attr_or_default<int64_t>("antialias", 0, attrs_map);
    std::vector<int64_t> axes = get_attrs_or_default<int64_t>("axes", {}, attrs_map);
    std::string coordinate =
        get_attr_or_default<std::string>("coordinate_transformation_mode", "half_pixel", attrs_map);
    float cubic = get_attr_or_default<float>("cubic_coeff_a", -0.75f, attrs_map);
    int64_t exclude = get_attr_or_default<int64_t>("exclude_outside", 0, attrs_map);
    float extrapolation = get_attr_or_default<float>("extrapolation_value", 0.0f, attrs_map);
    std::string aspect = get_attr_or_default<std::string>("keep_aspect_ratio_policy", "stretch", attrs_map);
    std::string mode = get_attr_or_default<std::string>("mode", "nearest", attrs_map);
    std::string nearest_mode = get_attr_or_default<std::string>("nearest_mode", "round_prefer_floor", attrs_map);

    // get the inputs
    int input_size = proto_node.input_size();
    if (input_size < 3) {
        std::ostringstream oss;
        oss << "Invalid inputs of Resize: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the outputs
    int output_size = proto_node.output_size();
    if (output_size != 1) {
        std::ostringstream oss;
        oss << "Invalid outputs of Resize: " << proto_node.name();
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& input = proto_node.input(0);
    const std::string& output = proto_node.output(0);

    auto input_iter = expressions.find(input);
    if (input_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Resize: " << proto_node.name() << " input: " << input;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // get the input shape and data type
    std::vector<int64_t> input_shape;
    tvm::DataType input_dtype;
    tvm_cpp::relay_utils::infer_relay_shape_dtype(input_iter->second, input_shape, input_dtype);

    // get the scale relay
    const std::string& scale_name = proto_node.input(2);
    auto scale_iter = expressions.find(scale_name);
    if (scale_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Resize: " << proto_node.name() << " input: " << scale_name;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    int dims = (int)input_shape.size();

    // the output size
    tvm::runtime::Array<tvm::relay::IndexExpr> output_size_array;
    // get the sizes
    if (input_size == 4) {
        const std::string& size_name = proto_node.input(3);
        auto size_iter = expressions.find(size_name);
        if (size_iter == expressions.end()) {
            std::ostringstream oss;
            oss << "Input not found, Resize: " << proto_node.name() << " input: " << size_name;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        std::vector<int64_t> size_shape;
        tvm::DataType size_dtype;
        tvm_cpp::relay_utils::infer_relay_shape_dtype(size_iter->second, size_shape, size_dtype);

        int64_t size_ele_nums = 1;
        for (auto& dim : size_shape) {
            size_ele_nums *= dim;
        }

        const tvm::relay::ConstantNode* const_expr = size_iter->second.as<tvm::relay::ConstantNode>();
        if (!const_expr) {
            return Status(StatusCode::RUNTIME_ERROR, "cast size to const node fails for Reize");
        }

        tvm::runtime::NDArray size_data = const_expr->data;

        if (size_dtype.is_int()) {
            if (size_dtype.bits() == 64) {
                for (int i = 0; i < size_ele_nums; ++i) {
                    auto val = static_cast<int64_t*>(size_data->data)[i] * input_shape[i];
                    output_size_array.push_back({(int32_t)val});
                }
            } else {
                return Status(StatusCode::NOT_IMPLEMENTED, "unsupported size data type for Reize");
            }
        } else {
            return Status(StatusCode::NOT_IMPLEMENTED, "unsupported size data type for Reize");
        }

    } else {
        std::vector<int64_t> scale_shape;
        tvm::DataType scale_dtype;
        tvm_cpp::relay_utils::infer_relay_shape_dtype(scale_iter->second, scale_shape, scale_dtype);

        int64_t scale_ele_nums = 1;
        for (auto& dim : scale_shape) {
            scale_ele_nums *= dim;
        }

        if (scale_ele_nums != dims) {
            return Status(StatusCode::INVALID_MODEL, "scale element num is not equal to inptu dims for Reize");
        }

        const tvm::relay::ConstantNode* const_expr = scale_iter->second.as<tvm::relay::ConstantNode>();
        if (!const_expr) {
            return Status(StatusCode::RUNTIME_ERROR, "cast scale to const node fails for Reize");
        }

        tvm::runtime::NDArray scale_data = const_expr->data;

        if (scale_dtype.is_float()) {
            for (int i = 2; i < scale_ele_nums; ++i) {
                auto val = static_cast<float*>(scale_data->data)[i] * input_shape[i];
                output_size_array.push_back({(int32_t)val});
            }
        } else if (scale_dtype.is_int()) {
            if (scale_dtype.bits() == 32) {
                for (int i = 2; i < scale_ele_nums; ++i) {
                    auto val = static_cast<int32_t*>(scale_data->data)[i] * input_shape[i];
                    output_size_array.push_back({(int32_t)val});
                }
            } else if (scale_dtype.bits() == 64) {
                for (int i = 2; i < scale_ele_nums; ++i) {
                    auto val = static_cast<int64_t*>(scale_data->data)[i] * input_shape[i];
                    output_size_array.push_back({(int32_t)val});
                }
            } else {
                return Status(StatusCode::NOT_IMPLEMENTED, "unsupported scale data type for Reize");
            }
        } else {
            return Status(StatusCode::NOT_IMPLEMENTED, "unsupported scale data type for Reize");
        }
    }

    tvm::runtime::Array<tvm::FloatImm> roi_arr;

    std::string roi_name = proto_node.input(1);
    tvm_cpp::utils::trim(roi_name);
    if (!roi_name.empty()) {
        auto roi_iter = expressions.find(roi_name);
        if (roi_iter == expressions.end()) {
            std::ostringstream oss;
            oss << "Input not found, Resize: " << proto_node.name() << " input: " << roi_name;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        const tvm::relay::ConstantNode* const_expr = roi_iter->second.as<tvm::relay::ConstantNode>();
        if (!const_expr) {
            return Status(StatusCode::RUNTIME_ERROR, "cast roi to const node fails for Reize");
        }

        DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
        const tvm::runtime::NDArray roi_data = const_expr->data;
        for (int i = 2; i < dims; ++i) {
            tvm::FloatImm roi_v(tvm::runtime::DataType{data_type},
                                static_cast<float*>(roi_data->data)[i] + static_cast<float*>(roi_data->data)[i + dims]);
            roi_arr.push_back(roi_v);
        }
    } else {
        DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
        tvm::FloatImm roi_v(tvm::runtime::DataType{data_type}, 0);
        for (int i = 0; i < dims; ++i) {
            roi_arr.push_back(roi_v);
        }
    }

    if (mode == "nearest") {
        mode = "nearest_neighbor";
    } else if (mode == "linear") {
        // do nothing
    } else if (mode == "cubic") {
        // do nothing
    } else {
        return Status(StatusCode::INVALID_MODEL, "unsupported mode for resize");
    }

    if (dims == 4) {
        tvm::DataType out_type;
        tvm::relay::Expr result_expr = (*resize2d)(input_iter->second, output_size_array, roi_arr, "NCHW", mode,
                                                   coordinate, nearest_mode, cubic, exclude, extrapolation, out_type);
        auto status = fold_const(result_expr);
        if (!status.is_ok()) {
            return status;
        }

        // add to expressions
        auto ret = expressions.emplace(output, result_expr);
        if (!ret.second) {
            ret.first->second = result_expr;
        }
        relay = result_expr;
    } else {
        return Status(StatusCode::RUNTIME_ERROR, "Unsupported input shape of Resize");
    }

    return Status::ok();
}

std::string ResizeParser::get_name() { return "Resize"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
#include "resize.h"

#include "utils/relay_utils.h"
#include "utils/utils.h"

namespace tvm_cpp {
namespace onnx_op {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
Status Resize2DParser::parse_op(const onnx::NodeProto& proto_node,
                                std::unordered_map<std::string, tvm::relay::Expr>& expressions,
                                tvm::relay::Expr& relay) {
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
    // the inptu shape expr
    tvm::relay::Expr input_shape_expr;
    tvm_cpp::relay_utils::infer_relay_shape(input_iter->second, input_shape_expr);

    // get the scale relay
    const std::string& scale_name = proto_node.input(2);
    auto scale_iter = expressions.find(scale_name);
    if (scale_iter == expressions.end()) {
        std::ostringstream oss;
        oss << "Input not found, Resize: " << proto_node.name() << " input: " << scale_name;
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    tvm::relay::Expr size_exp;
    // get the sizes
    if (input_size == 4) {
        const std::string& size_name = proto_node.input(3);
        auto size_iter = expressions.find(size_name);
        if (size_iter == expressions.end()) {
            std::ostringstream oss;
            oss << "Input not found, Resize: " << proto_node.name() << " input: " << size_name;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        size_exp = size_iter->second;
    } else {
        // data type cast
        const tvm::runtime::PackedFunc* cast = tvm::runtime::Registry::Get("relay.ir.cast");
        if (!cast) {
            return Status(StatusCode::INVALID_PARAM, "relay.ir.cast expression not found");
        }

        // get the scale shape and data type
        std::vector<int64_t> scale_shape;
        tvm::DataType scale_dtype;
        tvm_cpp::relay_utils::infer_relay_shape_dtype(scale_iter->second, scale_shape, scale_dtype);

        if (scale_shape.size() == 0) {
            return Status(StatusCode::INVALID_PARAM, "Invalid scale shape");
        }

        // cast to the scale data type
        tvm::relay::Expr exp = (*cast)(input_shape_expr, scale_dtype);

        const tvm::runtime::PackedFunc* multiply = tvm::runtime::Registry::Get("relay.op._make.multiply");
        if (!multiply) {
            return Status(StatusCode::INVALID_PARAM, "relay.op._make.multiply expression not found");
        }

        size_exp = (*multiply)(exp, scale_iter->second);
    }

    int dims = (int)input_shape.size();

    tvm::relay::Expr roi_exp;
    std::string roi_name = proto_node.input(1);
    tvm_cpp::utils::trim(roi_name);
    if (!roi_name.empty()) {
        auto roi_iter = expressions.find(roi_name);
        if (roi_iter == expressions.end()) {
            std::ostringstream oss;
            oss << "Input not found, Resize: " << proto_node.name() << " input: " << roi_name;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        roi_exp = roi_iter->second;

        tvm::runtime::Array<tvm::relay::IndexExpr> begin1({2});
        tvm::runtime::Array<tvm::relay::IndexExpr> end1({dims});

        tvm::runtime::Array<tvm::relay::IndexExpr> begin2({2 + dims});
        tvm::runtime::Array<tvm::relay::IndexExpr> end2({2 * dims});

        tvm::runtime::Array<tvm::relay::IndexExpr> strides({1});
        tvm::runtime::String slice_mode = "end";

        tvm::relay::Expr s1 = (*strided_slice)(roi_exp, begin1, end1, strides, slice_mode, nullptr);
        tvm::relay::Expr s2 = (*strided_slice)(roi_exp, begin2, end2, strides, slice_mode, nullptr);

        const tvm::runtime::PackedFunc* concat = tvm::runtime::Registry::Get("relay.op._make.concatenate");
        if (!concat) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.op._make.concatenate expression not found");
        }

        // get the tuple relay
        const tvm::runtime::PackedFunc* tuple = tvm::runtime::Registry::Get("relay.ir.Tuple");
        if (!tuple) {
            return Status(StatusCode::RUNTIME_ERROR, "relay.ir.Tuple expression not found");
        }

        tvm::runtime::Array<tvm::relay::Expr> input_array{s1, s2};
        tvm::relay::Expr all_input = (*tuple)(input_array, tvm::relay::Span());

        // Concate the input tensors by axis 0
        tvm::relay::Expr s = (*concat)(all_input, 0);
        auto ret = fold_const(s);
        if (!ret.is_ok()) {
            return ret;
        }
        roi_exp = s;
    }

    tvm::runtime::Array<tvm::relay::IndexExpr> begin({2});
    tvm::runtime::Array<tvm::relay::IndexExpr> end({dims});
    tvm::runtime::Array<tvm::relay::IndexExpr> strides({1});
    tvm::runtime::String slice_mode = "end";

    tvm::relay::Expr os = (*strided_slice)(size_exp, begin, end, strides, slice_mode, nullptr);

    auto ret = fold_const(os);
    if (!ret.is_ok()) {
        return ret;
    }
    tvm::relay::Expr output_size_exp = os;

    if (mode == "nearest") {
        mode = "nearest_neighbor";
    }

    if (dims == 4) {
        tvm::DataType out_type;
        tvm::relay::Expr result_expr = (*resize2d)(input_iter->second, output_size_exp, roi_exp, "NCHW", mode,
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

std::string Resize2DParser::get_name() { return "Resize"; }

}    // namespace onnx_op
}    // namespace tvm_cpp
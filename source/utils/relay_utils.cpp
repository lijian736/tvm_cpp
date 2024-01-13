#include "relay_utils.h"

#include <vector>

namespace tvm_cpp {
namespace relay_utils {

Status convert_initializer_to_relay(tvm::runtime::PackedFunc* gen_func, const onnx::TensorProto& proto_tensor,
                                    tvm::relay::Constant& relay) {
    if (!gen_func) {
        return Status(StatusCode::INVALID_PARAM, "gen_func is nullptr");
    }

    // check the tensor data location
    if (proto_tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
        return Status(StatusCode::NOT_IMPLEMENTED, "external tensor data is not supported now");
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
        element_num *= dim;
    }

    // It's a good idea to generate empty NDArray using PackedFunc 'runtime.TVMArrayAllocWithScope'
    // shape, data type, device
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
                    return Status(StatusCode::INVALID_MODEL, "Invalid tensor raw data length with its dims");
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
                    return Status(StatusCode::INVALID_MODEL, "Invalid tensor float data length with its dims");
                }
            }
        }

        default: {
            std::ostringstream oss;
            oss << "not support data type for proto tensor";
            return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
        }
    }

    tvm::relay::Span span;
    relay = (*gen_func)(initializer, span);

    return Status::ok();
}

}    // namespace relay_utils
}    // namespace tvm_cpp
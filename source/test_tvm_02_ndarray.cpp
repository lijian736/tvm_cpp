#include <dlpack/dlpack.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // the shape
    ShapeTuple shape = {2, 2};
    // the data type
    DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the device
    DLDevice dev{DLDeviceType::kDLCPU, 0};

    // shape, data type, device
    tvm::runtime::NDArray array = tvm::runtime::NDArray::Empty(shape, data_type, dev);

    // initialize the x NDArray
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            static_cast<float*>(array->data)[i * 2 + j] = i * 2 + j;
        }
    }

    tvm::runtime::NDArray dst_array = tvm::runtime::NDArray::Empty(shape, data_type, dev);
    dst_array.CopyFrom(array);

    std::cout << std::boolalpha;
    std::cout << "source is contiguous: " << array.IsContiguous() << std::endl;
    std::cout << "dest   is contiguous: " << dst_array.IsContiguous() << std::endl;

    std::cout << "Contents of NDArray y: " << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            auto value = static_cast<float*>(dst_array->data)[i * 2 + j];
            std::cout << i << ", " << j << " : " << value << std::endl;
        }
    }

    //print the NDArray shape
    ShapeTuple dst_shape = dst_array.Shape();
    std::cout << "y shape: " << std::endl;
    std::cout << "{ ";
    for (int i = 0; i < dst_shape.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }

        std::cout << dst_shape[i];
    }

    std::cout << " }" << std::endl;

    return 0;
}
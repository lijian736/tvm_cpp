#include <tvm/relay/expr.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // Constant in relay
    const tvm::runtime::PackedFunc* const_gen = tvm::runtime::Registry::Get("relay.ir.Constant");
    if (!const_gen) {
        std::cerr << "relay.ir.Constant expression not found" << std::endl;
        return -1;
    }

    // shape, data type, device
    tvm::runtime::NDArray initializer =
        tvm::runtime::NDArray::Empty({2, 2}, {DLDataTypeCode::kDLFloat, 32, 1}, {DLDeviceType::kDLCPU, 0});
    // initialize the NDArray
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            static_cast<float*>(initializer->data)[i * 2 + j] = i * 2 + j;
        }
    }

    tvm::relay::Span span;
    // the constant expression, the NDArray as the Constant's content
    tvm::relay::Constant const_expr = (*const_gen)(initializer, span);

    bool is_scalar = const_expr->is_scalar();
    std::cout << std::boolalpha;
    std::cout << "Const expression is scalar: " << is_scalar << std::endl;

    // the data
    tvm::runtime::NDArray data = const_expr->data;

    ShapeTuple shape = data.Shape();
    std::cout << "shape: " << std::endl;

    std::cout << "{ ";
    for (int i = 0; i < shape.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }

        std::cout << shape[i];
    }
    std::cout << " }" << std::endl;

    return 0;
}
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>
#include <string>

#include "onnx.proto3.pb.h"
#include "utils/onnx_utils.h"
#include "utils/utils.h"

using namespace tvm_cpp::utils;
using namespace tvm_cpp::onnx_utils;

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // Step 0. Prepare the packed functions

    // the relu function
    const tvm::runtime::PackedFunc* relu = tvm::runtime::Registry::Get("relay.op.nn._make.relu");
    if (!relu) {
        std::cerr << "relay.op.nn._make.relu expression not found" << std::endl;
        return -1;
    }

    // the input generate function
    const tvm::runtime::PackedFunc* input_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!input_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    // get the function relay
    const tvm::runtime::PackedFunc* function = tvm::runtime::Registry::Get("relay.ir.Function");
    if (!function) {
        std::cerr << "relay.ir.Function expression not found" << std::endl;
        return -1;
    }

    // Step 1. Generate the input tensor

    // the input tensor type, float32 and shape is {1, 1, 2, 2}
    tvm::relay::TensorType input_tensor_type{{1, 1, 2, 2}, tvm::DataType::Float(32)};

    // the input expression
    tvm::relay::Var input_var = (*input_gen)("input_data", input_tensor_type, tvm::relay::Span());
    // get the relu expression
    tvm::relay::Expr relu_expr = (*relu)(input_var);

    // Step 2. Generate a relay IRModule with only 1 node - relu

    // the graph input
    tvm::runtime::Array<tvm::relay::Expr> all_input;
    all_input.push_back(input_var);

    // the graph output
    tvm::relay::Expr all_output = relu_expr;

    // build the graph with its input, output
    tvm::relay::Expr relu_main_func =
        (*function)(all_input, all_output, tvm::relay::Type(), tvm::runtime::Array<tvm::relay::TypeVar>(),
                    tvm::DictAttrs(), tvm::relay::Span());

    // the graph relay ir module
    tvm::IRModule ir_module = IRModule::FromExpr(relu_main_func);

    // Step 3. Create an Executor
    // Step 4. Do inference
    // Step 5. Output the result

    // the shape
    ShapeTuple shape = {2, 2};
    // the data type
    DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the device
    DLDevice dev{DLDeviceType::kDLCPU, 0};

    // shape, data type, device
    tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(shape, data_type, dev);

    static_cast<float*>(input->data)[0] = 1.0f;
    static_cast<float*>(input->data)[1] = -1.0f;
    static_cast<float*>(input->data)[2] = -2.0f;
    static_cast<float*>(input->data)[3] = 3.0f;

    return 0;
}
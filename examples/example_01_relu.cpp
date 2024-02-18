#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <string>

#include "onnx.proto3.pb.h"
#include "utils/onnx_utils.h"
#include "utils/utils.h"

using namespace tvm_cpp::utils;
using namespace tvm_cpp::onnx_utils;

int main(int argc, char** argv) {
    std::cout << "example_01_relu:" << std::endl;

    return 0;
}
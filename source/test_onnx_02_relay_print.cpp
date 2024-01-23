#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>
#include <string>

#include "onnx.proto3.pb.h"
#include "onnx_op/op_parser.h"
#include "utils/onnx_utils.h"
#include "utils/relay_utils.h"
#include "utils/utils.h"

using namespace tvm_cpp::utils;
using namespace tvm_cpp::onnx_utils;
using namespace tvm_cpp::onnx_op;
using namespace tvm_cpp::relay_utils;

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    OnnxOpParserRegister::get_instance()->register_all_supported_ops();

    if (argc <= 1) {
        std::cerr << "Invalid arguments" << std::endl;
        return -1;
    }

    // Get input file path
    std::string file_name(argv[1]);
    std::cout << "The input file name is: " << file_name << std::endl;

    // Check if the input file exist
    trim(file_name);
    bool exist = file_exist(file_name);
    if (!exist) {
        std::cerr << "The input file does NOT exist, please check the file path" << std::endl;
        return -2;
    }

    // Read onnx file
    std::ifstream ifs(file_name, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Open file failed: " << file_name << std::endl;
        return -3;
    }

    onnx::ModelProto onnx_model;

    google::protobuf::io::IstreamInputStream input_stream(&ifs);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);
    bool parsed = onnx_model.ParseFromCodedStream(&coded_input);
    ifs.close();
    if (!parsed) {
        std::cerr << "Parse onnx model failed: " << file_name << std::endl;
        return -4;
    }

    tvm::IRModule mod;
    auto ret = parse_graph_to_irmodule(onnx_model.graph(), mod);
    if (!ret.is_ok()) {
        std::cout << ret << std::endl;
        return -1;
    }

    // Pretty print
    const tvm::runtime::PackedFunc* pretty_print = tvm::runtime::Registry::Get("relay.ir.PrettyPrint");
    if (!pretty_print) {
        std::cerr << "relay.ir.PrettyPrint expression not found" << std::endl;
        return -1;
    }

    // print the type infered IR model
    tvm::String result_text = (*pretty_print)(mod);
    std::string result_str = (std::string)result_text;
    std::cout << "the ONNX relay IR Model Result info: " << std::endl << result_str << std::endl;

    return 0;
}
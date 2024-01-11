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

    // print the model info
    std::cout << "------------------------------------" << std::endl;
    print_onnx_model_info(onnx_model);

    // get the all nodes types
    std::unordered_map<std::string, int> node_types_map;
    retrieve_onnx_node_types(onnx_model, node_types_map);

    std::cout << "------------------------------------" << std::endl;
    std::cout << "node types list: " << std::endl;
    for (auto& pair : node_types_map) {
        std::cout << pair.first << " : " << pair.second << std::endl;
    }

    return 0;
}
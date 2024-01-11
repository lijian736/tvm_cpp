#!/bin/bash

set -e
ROOT_DIR=$(realpath $(dirname $(realpath "$0"))"/../")
DEST_DIR=${ROOT_DIR}/third_party/onnx_proto

#generate the CPP source for onnx.proto3
protoc --proto_path=${DEST_DIR} --cpp_out=${DEST_DIR} ${DEST_DIR}/onnx.proto3
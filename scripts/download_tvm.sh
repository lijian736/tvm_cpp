#!/bin/bash

set -e
ROOT_DIR=$(realpath $(dirname $(realpath "$0"))"/../")
DEST_DIR=${ROOT_DIR}/third_party/tvm

git clone --recursive --depth=1 --branch v0.14.0 https://github.com/apache/tvm ${DEST_DIR}
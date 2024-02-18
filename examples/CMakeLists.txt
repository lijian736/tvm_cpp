find_package(Protobuf 3 REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

#add include folder
include_directories("${CMAKE_SOURCE_DIR}/third_party/onnx_proto")
include_directories("${CMAKE_SOURCE_DIR}/third_party/tvm/include")
include_directories("${CMAKE_SOURCE_DIR}/source")

include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/third_party/tvm/3rdparty/dlpack/include")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/third_party/tvm/3rdparty/dmlc-core/include")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/third_party/tvm/3rdparty/rang/include")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/third_party/tvm/3rdparty/compiler-rt")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/third_party/tvm/3rdparty/picojson")

find_library(TVM_LIBRARY         NAMES tvm         PATHS ${CMAKE_SOURCE_DIR}/third_party/tvm/build/ PATH_SUFFIXES lib)
find_library(TVM_RUNTIME_LIBRARY NAMES tvm_runtime PATHS ${CMAKE_SOURCE_DIR}/third_party/tvm/build/ PATH_SUFFIXES lib)

function(GENERATE_EXAMPLE name)
    add_executable(${name} ${name}.cpp ${CMAKE_SOURCE_DIR}/third_party/onnx_proto/onnx.proto3.pb.cc)
    target_link_libraries(${name} PRIVATE ${Protobuf_LIBRARIES} ${TVM_LIBRARY} utils)
    target_compile_definitions(${name} PRIVATE DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
endfunction()

GENERATE_EXAMPLE(example_01_relu)
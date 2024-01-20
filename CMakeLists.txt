# CMake minimal version
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Project name
project(tvm_cpp C CXX)

# C++ 17
set(CMAKE_CXX_STANDARD 17)
# C++ 17 required
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Generate compile_commands.json
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# Add source folder
add_subdirectory(source)

# Add examples folder
add_subdirectory(examples)
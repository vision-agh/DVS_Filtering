cmake_minimum_required(VERSION 3.15)
project(noise_generator_py)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create the pybind11 module
pybind11_add_module(noise_generator_py pybind_module.cpp)

# Link any necessary libraries if needed
# target_link_libraries(noise_generator_py PRIVATE ...)
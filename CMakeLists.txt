cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project("cuda_spblas" LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set (CMAKE_CXX_STANDARD 17)

set (CUDA_ARCH 70)

set(CMAKE_BUILD_TYPE Debug)

string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
string(APPEND CMAKE_CUDA_FLAGS "  -Xcompiler -Wall -D_FORCE_INLINES --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")

add_subdirectory(src)

cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project("cuda_spblas" LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set (CMAKE_CXX_STANDARD 17)

# set(BOOST_ROOT /home/guo.chengxin/boost_1_82_0)
# set(Boost_NO_SYSTEM_PATHS ON)

# find_package(Boost COMPONENTS regex system REQUIRED)
# include_directories(${Boost_INCLUDE_DIRS})

set (CUDA_ARCH 70)

set(CMAKE_BUILD_TYPE Release)

string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
string(APPEND CMAKE_CUDA_FLAGS "  -Xcompiler -Wall -D_FORCE_INLINES --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -I/home/guochengxin/cusplibrary-cuda10")

add_subdirectory(src)

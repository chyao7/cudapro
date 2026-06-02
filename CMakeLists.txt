cmake_minimum_required(VERSION 3.18)

# CUDA 13+ 须在 project() 前设定架构
if(NOT DEFINED CUDAPRO_CUDA_ARCH)
  set(CUDAPRO_CUDA_ARCH "86" CACHE STRING "CUDA compute capability without sm_ prefix")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${CUDAPRO_CUDA_ARCH}")
set(CMAKE_CUDA_FLAGS "-arch=sm_${CUDAPRO_CUDA_ARCH}" CACHE STRING "CUDA compiler flags" FORCE)

project(cudapro LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug/Release/RelWithDebInfo)" FORCE)
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "" FORCE)

find_package(CUDAToolkit REQUIRED)

if(NOT CUTLASS_DIR)
  if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR "$ENV{CUTLASS_DIR}" CACHE PATH "CUTLASS repository root")
  else()
    set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/../cutlass" CACHE PATH "CUTLASS repository root")
  endif()
endif()

set(CUDAPRO_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(CUDAPRO_BENCH_DIR "${CMAKE_SOURCE_DIR}/benchmarks")
set(CUDAPRO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

include("${CMAKE_SOURCE_DIR}/cmake/CudaLessons.cmake")

cudapro_register_cuda_sources("${CUDAPRO_SOURCE_DIR}")
cudapro_register_cuda_sources("${CUDAPRO_BENCH_DIR}")
cudapro_add_all_cuda_target()

message(STATUS "Project:    cudapro")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA arch:  sm_${CUDAPRO_CUDA_ARCH}")
message(STATUS "CUTLASS:    ${CUTLASS_DIR}")
message(STATUS "Sources:    ${CUDAPRO_SOURCE_DIR}")
message(STATUS "Benchmarks: ${CUDAPRO_BENCH_DIR}")
message(STATUS "Binaries:   ${CMAKE_BINARY_DIR}/bin/")

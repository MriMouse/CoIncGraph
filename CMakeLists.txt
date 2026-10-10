cmake_minimum_required(VERSION 2.8)
add_definitions(-w)
project(CoIncGraph LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
############## DEPENDENCIES ##############
find_package(CUDA 9 REQUIRED)

message("-- CUDA Version: ${CUDA_VERSION_STRING}")

# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

# CUB

# JSON
include_directories(${PROJECT_SOURCE_DIR}/deps/json)

# # # STDGPU
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")

add_subdirectory(deps/stdgpu)
include_directories(${PROJECT_BINARY_DIR}/deps/stdgpu/bin/include)
link_directories(${PROJECT_BINARY_DIR}/deps/stdgpu/bin/lib)
set_property(TARGET stdgpu PROPERTY CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET stdgpu PROPERTY CUDA_ARCHITECTURES 70)
############## BUILD ##############

set(EXTRA_LIBS stdgpu gflags pthread ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_70,code=sm_70;-g;-lineinfo;-Xcompiler;-ggdb;-std=c++17 -O3)
else ()
     set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_70,code=sm_70;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG;-std=c++17 -O3)
endif ()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG")
message("-- Host Compiler: ${CUDA_HOST_COMPILER}")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

# hybrid apps
cuda_add_executable(hybrid_pr src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp
       samples/hybrid_pr/hybrid_pr.cu
       samples/hybrid_pr/hybrid_pr_host.cu
       samples/hybrid_pr/main.cu)
target_link_libraries(hybrid_pr ${EXTRA_LIBS} pthread)
target_sources(hybrid_pr PRIVATE src/utils/communication_meter.cpp)
target_include_directories(hybrid_pr PRIVATE ${CUDA_INCLUDE_DIRS})
set_property(TARGET hybrid_pr APPEND_STRING PROPERTY LINK_FLAGS
    " -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMemcpyAsync")

cuda_add_executable(hybrid_bfs src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp
        samples/hybrid_bfs/hybrid_bfs.cu
        samples/hybrid_bfs/hybrid_bfs_host.cu
        samples/hybrid_bfs/main.cu)
target_link_libraries(hybrid_bfs ${EXTRA_LIBS} pthread)
target_sources(hybrid_bfs PRIVATE src/utils/communication_meter.cpp)
target_include_directories(hybrid_bfs PRIVATE ${CUDA_INCLUDE_DIRS})
set_property(TARGET hybrid_bfs APPEND_STRING PROPERTY LINK_FLAGS
    " -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMemcpyAsync")

cuda_add_executable(hybrid_sssp src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp
        samples/hybrid_sssp/hybrid_sssp.cu
       samples/hybrid_sssp/hybrid_sssp_host.cu
        samples/hybrid_sssp/main.cu)
target_link_libraries(hybrid_sssp ${EXTRA_LIBS} pthread)
target_sources(hybrid_sssp PRIVATE src/utils/communication_meter.cpp)
target_include_directories(hybrid_sssp PRIVATE ${CUDA_INCLUDE_DIRS})
set_property(TARGET hybrid_sssp APPEND_STRING PROPERTY LINK_FLAGS
    " -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMemcpyAsync")

cuda_add_executable(hybrid_cc src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp
        samples/hybrid_cc/hybrid_cc.cu
        samples/hybrid_cc/hybrid_cc_host.cu
        samples/hybrid_cc/main.cu)
target_link_libraries(hybrid_cc ${EXTRA_LIBS} pthread)
target_sources(hybrid_cc PRIVATE src/utils/communication_meter.cpp)
target_include_directories(hybrid_cc PRIVATE ${CUDA_INCLUDE_DIRS})
set_property(TARGET hybrid_cc APPEND_STRING PROPERTY LINK_FLAGS
    " -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaMemcpyAsync")

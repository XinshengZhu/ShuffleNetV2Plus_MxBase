# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.10.1)

# project information
project(ShuffleNetV2Plus_MxBase)
set(TARGET ShuffleNetV2Plus_MxBase)

# Compile options
add_definitions(-DENABLE_DVPP_INTERFACE)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wall)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -s -pie")
set(CMAKE_SKIP_RPATH TRUE)

set(MXBASE_ROOT_DIR $ENV{MX_SDK_HOME})
set(MXBASE_INC ${MXBASE_ROOT_DIR}/include)
set(MXBASE_LIB_DIR ${MXBASE_ROOT_DIR}/lib)
set(MXBASE_POST_LIB_DIR ${MXBASE_ROOT_DIR}/lib/modelpostprocessors)
set(MXBASE_POST_PROCESS_DIR ${MXBASE_ROOT_DIR}/include/MxBase/postprocess/include)

if(DEFINED ENV{MXSDK_OPENSOURCE_DIR})
    set(OPENSOURCE_DIR $ENV{MXSDK_OPENSOURCE_DIR})
else()
    set(OPENSOURCE_DIR ${MXBASE_ROOT_DIR}/opensource)
endif()
include_directories(${MXBASE_POST_LIB_DIR})
include_directories(${OPENSOURCE_DIR}/include)

include_directories(${MXBASE_INC})
include_directories(${MXBASE_POST_PROCESS_DIR})

link_directories(${OPENSOURCE_DIR}/lib)
link_directories(${MXBASE_LIB_DIR})
link_directories(${MXBASE_POST_LIB_DIR})

add_executable(${TARGET} main.cpp Shufflenetv2plusClassify.cpp)
target_link_libraries(${TARGET} glog cpprest mxbase resnet50postprocess opencv_world stdc++fs)

install(TARGETS ${TARGET} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../config/logging.conf DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../out/config)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../config/sdk.conf DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../out/config)



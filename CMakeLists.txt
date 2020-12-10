cmake_minimum_required(VERSION 2.8.3)
project(yolov5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "Opencv library status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "> libraries: ${OpenCV_LIBS}")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS} ")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ncnn)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)


add_executable(yolov5.test yolov5.cpp)

target_link_libraries(yolov5.test ${OpenCV_LIBS} ncnn)
cmake_minimum_required(VERSION 3.20)
project(lv2cv)

#add_executable(lv2cv main.cpp)
add_executable(lv2cv asinx+b.cpp)
#add_library(lv2cv SHARED lv2cv.cpp lv2cv.h)

set(CMAKE_CXX_STANDARD 14)

#添加2行代码, 注意OpenCV_DIR的目录是自己编译生成的文件夹
set(OpenCV_DIR "D:/opencv/opencv/build")
#include_directories(D:/opencv/opencv/build/x64/vc15/bin)


# 使用conan时需要添加这两行
#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
#添加1行代码
target_link_libraries(lv2cv ${OpenCV_LIBS})

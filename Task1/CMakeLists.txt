cmake_minimum_required(VERSION 3.10)
project(Task1)

set(CMAKE_CXX_STANDARD 20)

if(NOT DEFINED TYPE)
    set(TYPE "double")
endif()

add_compile_definitions(-DTYPE=${TYPE})

add_executable(task1 task1.cpp)
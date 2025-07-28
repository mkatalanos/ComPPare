cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "manual_timing_demo")

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -DHAVE_GOOGLE_BENCHMARK")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(benchmark REQUIRED PATHS "${CMAKE_SOURCE_DIR}/../../../benchmark/build")

include_directories(
    "${CMAKE_SOURCE_DIR}/../../../include"
)

add_executable(${PROJECT_NAME} manual_timing_demo.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE benchmark::benchmark stdc++)

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

set(PROJECT_NAME "google_benchmark_demo")

project(${PROJECT_NAME} LANGUAGES CXX)

find_package(benchmark REQUIRED PATHS "${CMAKE_SOURCE_DIR}/../../../benchmark/build")
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

include_directories(
    "${CMAKE_SOURCE_DIR}/../../../include"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -DHAVE_GOOGLE_BENCHMARK")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CPP_SOURCES
    google_benchmark_demo.cpp
)

add_executable(${PROJECT_NAME} ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    benchmark::benchmark
    stdc++
    OpenMP::OpenMP_CXX
    ${CMAKE_THREAD_LIBS_INIT}
    TBB::tbb
    m
)

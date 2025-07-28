cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

set(PROJECT_NAME "custom_error_policy_demo")

project(${PROJECT_NAME} LANGUAGES CXX)

include_directories("${CMAKE_SOURCE_DIR}/../../../include")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -fsanitize=address")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CPP_SOURCES
  custom_error_policy_demo.cpp
)

add_executable(${PROJECT_NAME} ${CPP_SOURCES})
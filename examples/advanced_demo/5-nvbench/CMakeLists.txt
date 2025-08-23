cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(PROJECT_NAME "saxpy_nvbench")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(nvbench REQUIRED PATHS "${CMAKE_SOURCE_DIR}/../../../nvbench/build")
find_package(CUDAToolkit)

include_directories(
    "${CMAKE_SOURCE_DIR}/../../../include"
)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "-Wall -Wextra -std=c++20")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set(CUDA_SOURCES
    saxpy_nvbench.cu
)

add_executable(${PROJECT_NAME} ${CUDA_SOURCES})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    nvbench::nvbench
    stdc++
    CUDA::cudart
    m
)


# Detect CUDA architectures supported by nvcc 
execute_process(
  COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
  OUTPUT_VARIABLE raw_archs
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCHALL "compute_([0-9]+)" sm_list ${raw_archs})
set(arch_list "")
foreach(sm ${sm_list})
  string(REGEX REPLACE "compute_" "" num ${sm})
  list(APPEND arch_list ${num})
endforeach()
list(REMOVE_DUPLICATES arch_list)
message(STATUS "Cuda Architectures supported by compiler: sm_${arch_list}")

# Verify if user defined CUDA_ARCH is valid for compiler
if(CUDA_ARCH)

  set(_real_archs)

  foreach(a IN LISTS CUDA_ARCH)
    if(NOT "${a}" IN_LIST arch_list)
      message(FATAL_ERROR "Invalid CUDA_ARCH specified: ${CUDA_ARCH}. Valid options are: ${arch_list}")
    endif()
    list(APPEND _real_archs "${a}-real")
  endforeach()

  set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${_real_archs}"
  )

else()

  set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES all
  )

endif()

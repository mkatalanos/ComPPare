cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(PROJECT_NAME "saxpy")

project(${PROJECT_NAME} LANGUAGES CXX) # CUDA will be added below if needed
option(USE_CUDA "Enable CUDA support" OFF) # default is OFF


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Include the directory for ComPPare.hpp
include_directories(
    "${CMAKE_SOURCE_DIR}/../../include"
)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

set(CPP_SOURCES
    main.cpp
    saxpy_cpu.cpp
    init_saxpy.cpp
)

set(CUDA_SOURCES
    saxpy_gpu.cu
)

# CPU ONLY
if (NOT USE_CUDA)
  message(STATUS "-- CUDA support: OFF --")

  set(CMAKE_CXX_STANDARD        20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  set(CMAKE_CXX_FLAGS        "-Wall -Wextra")
  set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O3")

  add_executable(${PROJECT_NAME} ${CPP_SOURCES})
  target_link_libraries(${PROJECT_NAME}
      PRIVATE
      ${CMAKE_THREAD_LIBS_INIT}
      stdc++
      m
  )
  return()
endif()

# GPU 
message(STATUS "-- CUDA support: ON --")

enable_language(CUDA)
find_package(CUDAToolkit)
if (NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "CUDAToolkit not found.")
endif()

include_directories(
    "${CUDAToolkit_INCLUDE_DIRS}"
)

set(CMAKE_CXX_STANDARD        20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS        "-Wall -Wextra -DHAVE_CUDA")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3")

set(CMAKE_CUDA_STANDARD       20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS        "-Wall -Wextra -DHAVE_CUDA")
set(CMAKE_CUDA_FLAGS_DEBUG         "-O0 -G -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE        "-O3")

find_package(CUDAToolkit REQUIRED)


# link CUDA sources into shared library (not sure why but double free error when static linking)
add_library(saxpy_cuda SHARED ${CUDA_SOURCES})
target_link_libraries(saxpy_cuda 
                      PRIVATE 
                      CUDA::cudart 
                      ${CMAKE_THREAD_LIBS_INIT}
                      stdc++
                      m
)
set_target_properties(saxpy_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

# final executable
add_executable(${PROJECT_NAME} ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME}
                      PRIVATE
                      saxpy_cuda
                      CUDA::cudart
                      ${CMAKE_THREAD_LIBS_INIT}
                      stdc++
                      m
                      OpenMP::OpenMP_CXX
                      TBB::tbb
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
if (CUDA_ARCH)

    set(_real_archs)

    foreach(a IN LISTS CUDA_ARCH)
      if (NOT "${a}" IN_LIST arch_list)
        message(FATAL_ERROR "Invalid CUDA_ARCH specified: ${CUDA_ARCH}. Valid options are: ${arch_list}")
      endif()
      list(APPEND _real_archs "${a}-real")
    endforeach()

    set_target_properties(
      saxpy_cuda PROPERTIES
      CUDA_ARCHITECTURES "${_real_archs}"
    )

    set_target_properties(
      ${PROJECT_NAME} PROPERTIES
      CUDA_ARCHITECTURES "${_real_archs}"
    )

else()

    message(STATUS "No CUDA_ARCH specified, compiling for all architectures supported by the compiler")
    set_target_properties(
      saxpy_cuda PROPERTIES
      CUDA_ARCHITECTURES all
    )

    set_target_properties(
      ${PROJECT_NAME} PROPERTIES
      CUDA_ARCHITECTURES all
    )

endif()
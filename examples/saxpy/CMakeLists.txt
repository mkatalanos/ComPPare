cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

set(PROJECT_NAME "saxpy")

project(${PROJECT_NAME} LANGUAGES CXX) # CUDA will be added below if needed
option(CUDA "Enable CUDA support" OFF) # default is OFF


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Include the directory for ComPPare.hpp
include_directories(
    "${CMAKE_SOURCE_DIR}/../../include"
)

find_package(Threads REQUIRED)

set(CPP_SOURCES
    main.cpp
    saxpy_cpu.cpp
    init_saxpy.cpp
)

set(CUDA_SOURCES
    saxpy_gpu.cu
)

# CPU ONLY
if (NOT CUDA)
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

# NVHPC compiler supports std::par on GPU with -stdpar flag
# This is not supported by other compilers, so we check the compiler ID
# and set the flag accordingly.
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(STATUS "Using NVHPC compiler, enabling std::par on GPU")
    set(STDPARFLAG "-stdpar=gpu")
    set(HAVE_NVHPC ON)
    set(HAVE_NVHPC_FLAG "-DHAVE_NVHPC") # preprocessor flag to indicate NVHPC compiler to enable std::par on GPU
else()
    message(STATUS "Non-NVHPC compiler skipping using std::par on CPU")
    set(STDPARFLAG "")
    set(HAVE_NVHPC_FLAG "")
endif()

set(CMAKE_CXX_STANDARD        20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS        "-Wall -Wextra -DHAVE_CUDA ${HAVE_NVHPC_FLAG} ${STDPARFLAG}")
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
  list(APPEND arch_list ${num}-real)
endforeach()
list(REMOVE_DUPLICATES arch_list)
list(SORT arch_list)
message(STATUS "Auto-detected CUDA ARCHS: sm_${arch_list}")

# Verify if user defined CUDA_ARCH is valid for compiler
if (CUDA_ARCH)
  if ("${CUDA_ARCH}-real" IN_LIST arch_list)
    message(STATUS "Using user-specified CUDA ARCH: sm_${CUDA_ARCH}-real")
    set(arch_list "${CUDA_ARCH}-real")
  else()
    message(FATAL_ERROR "Invalid CUDA architecture specified: ${CUDA_ARCH}. "
      "Valid options are: ${arch_list}")
  endif()
endif()
# set the CUDA architectures for the project
set(CMAKE_CUDA_ARCHITECTURES ${arch_list} CACHE STRING "" FORCE)

# link CUDA sources into shared library (not sure why but double free error when static linking)
# could be due to the use of std::par on GPU with NVHPC
add_library(saxpy_cuda SHARED ${CUDA_SOURCES})
target_link_libraries(saxpy_cuda 
                      PRIVATE 
                      CUDA::cudart 
                      ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(saxpy_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

# final executable
add_executable(${PROJECT_NAME} ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME}
                      PRIVATE
                      saxpy_cuda
                      CUDA::cudart
                      ${CMAKE_THREAD_LIBS_INIT}
)

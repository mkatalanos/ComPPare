cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "DoNotOptimize_demo")

project(${PROJECT_NAME} LANGUAGES CXX)

option(USE_GOOGLE_BENCHMARK "Use Google Benchmark for performance comparison" OFF)
if(USE_GOOGLE_BENCHMARK)
    find_package(benchmark REQUIRED PATHS "${CMAKE_SOURCE_DIR}/../../../benchmark/build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_GOOGLE_BENCHMARK")
else()
    message(STATUS "Google Benchmark is not used. Set USE_GOOGLE_BENCHMARK to ON to enable it.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(
    "${CMAKE_SOURCE_DIR}/../../../include"
)

add_executable(${PROJECT_NAME} DoNotOptimize_demo.cpp)

if(USE_GOOGLE_BENCHMARK)
    target_link_libraries(${PROJECT_NAME} PRIVATE benchmark::benchmark stdc++)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++)
endif()

# === Generate ASM files === 
set(ASM_FLAGS
    "-O3" # optimize to show the effect of comppare::DoNotOptimize()
    "-Wall"
    "-Wextra"
    "-Wconversion"
    "-fno-omit-frame-pointer"
    "-std=c++${CMAKE_CXX_STANDARD}"
    "-I${CMAKE_SOURCE_DIR}/../../../include"
    "-S"
)

if(USE_GOOGLE_BENCHMARK)
  list(APPEND ASM_FLAGS 
    "-DHAVE_GOOGLE_BENCHMARK"
    "-I${CMAKE_SOURCE_DIR}/../../../benchmark/include"
)
endif()

# helper function to make assembly files
function(make_asm cpp_file)
  get_filename_component(name_we ${cpp_file} NAME_WE)
  set(outdir "${CMAKE_CURRENT_BINARY_DIR}/asm")
  file(MAKE_DIRECTORY ${outdir})
  set(asm_out "${outdir}/${name_we}.s")

  add_custom_command(
    OUTPUT ${asm_out}
    COMMAND_EXPAND_LISTS
    COMMAND ${CMAKE_CXX_COMPILER}
            ${ASM_FLAGS}
            -o ${asm_out}
            ${CMAKE_SOURCE_DIR}/${cpp_file}
    DEPENDS ${CMAKE_SOURCE_DIR}/${cpp_file}
    COMMENT "Generating assembly listing: ${asm_out}"
    VERBATIM
  )

  set(_ALL_ASM_DUMPS ${_ALL_ASM_DUMPS} ${asm_out} PARENT_SCOPE)
endfunction()

make_asm(saxpy.cpp)
make_asm(saxpy_DoNotOptimize.cpp)
make_asm(DoNotOptimize_demo.cpp)

add_custom_target(
    asm
    ALL
    DEPENDS
    ${_ALL_ASM_DUMPS}
)
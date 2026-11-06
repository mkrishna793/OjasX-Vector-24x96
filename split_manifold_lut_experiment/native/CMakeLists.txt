cmake_minimum_required(VERSION 3.16)
project(ojasx_lut VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# Include directories
include_directories(include)

# Core Library Sources
set(OJASX_SOURCES
    src/ojasx_lut_cpu.cpp
    src/ojasx_lut_kv_cache.cpp
    src/ojasx_lut_c_api.cpp
    src/ojasx_ggml_adapter.cpp
)

# Shared Library
add_library(ojasx_lut SHARED ${OJASX_SOURCES})
set_target_properties(ojasx_lut PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "ojasx_lut"
)

# Test Executable
add_executable(test_native_lut tests/test_native_lut.cpp ${OJASX_SOURCES})

# Standalone CLI Runner & Benchmark
add_executable(ojasx_run src/ojasx_run.cpp ${OJASX_SOURCES})

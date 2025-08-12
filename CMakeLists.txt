cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0042 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(ccl LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect source files
set(SOURCE_LIB
    src/libccl.cxx
    src/CCParticle.cxx
    src/EnergyUtils.cxx
    src/CoalescenceUtils.cxx
    src/MSTUtils.cxx
    src/Tracking.cxx
)

# Header files to install
set(PUBLIC_HEADERS
    src/libccl.h
    src/CCParticle.h
    src/EnergyUtils.h
    src/SkyrmeFormula.h
)

# Create shared library
add_library(ccl SHARED ${SOURCE_LIB})

set_target_properties(ccl PROPERTIES
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
    OUTPUT_NAME "ccl"
)


# Install targets
install(TARGETS ccl
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Tests
option(BUILD_TESTS "Build the test suite" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

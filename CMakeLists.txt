cmake_minimum_required(VERSION 3.15)
project(spheni VERSION 0.1.0 LANGUAGES CXX)

cmake_policy(SET CMP0148 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

add_library(spheni STATIC
    src/core/engine.cpp
    src/core/factory.cpp
    src/core/flat_index.cpp
    src/core/ivf_index.cpp
    src/utils/kernels.cpp
    src/utils/kmeans.cpp
    src/utils/topk.cpp
)
set_target_properties(spheni PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(SPHENI_BUILD_PYTHON "Build Python bindings" ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spheni PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(spheni PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(spheni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(SPHENI_BUILD_PYTHON)
    find_package(pybind11 REQUIRED CONFIG)
    pybind11_add_module(spheni_py MODULE
        src/python/bindings.cpp
    )
    target_link_libraries(spheni_py PRIVATE spheni)
    set_target_properties(spheni_py PROPERTIES OUTPUT_NAME spheni)
endif()

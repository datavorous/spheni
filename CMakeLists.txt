cmake_minimum_required(VERSION 3.15)
project(spheni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(spheni STATIC
    src/indexes/flat.cpp
    src/indexes/ivf.cpp
    src/math/kmeans.cpp
)

target_include_directories(spheni
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(spheni PRIVATE
    -O3 -march=native -ffast-math -fno-exceptions -fno-rtti
)

foreach(ex flat ivf)
    add_executable(example_${ex} examples/${ex}.cpp)
    target_link_libraries(example_${ex} PRIVATE spheni)
endforeach()
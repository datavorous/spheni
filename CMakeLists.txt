cmake_minimum_required(VERSION 3.15)
project(spheni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /O2)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

add_library(spheni STATIC
  src/core/engine.cpp
  src/core/factory.cpp
  src/indexes/flat/flat_index.cpp
  src/indexes/ivf/ivf_index.cpp
  src/math/kernels.cpp
  src/math/kmeans.cpp
  src/math/topk.cpp
)

target_include_directories(spheni
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(spheni PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(example_flat examples/flat/example.cpp)
target_link_libraries(example_flat PRIVATE spheni)

add_executable(example_ivf examples/ivf/example.cpp)
target_link_libraries(example_ivf PRIVATE spheni)

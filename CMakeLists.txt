cmake_minimum_required(VERSION 3.15)
project(spheni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(spheni STATIC
    src/indexes/flat.cpp
    src/indexes/ivf.cpp
    src/math/kmeans.cpp
    src/indexes/pq_flat.cpp
    src/indexes/ivf_pq.cpp
)

target_include_directories(spheni
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(spheni PRIVATE
    -O3 -march=native -ffast-math -fno-exceptions -fno-rtti
)

# foreach(ex flat ivf pq_flat ivf_pq)
#     add_executable(example_${ex} examples/${ex}.cpp)
#     target_link_libraries(example_${ex} PRIVATE spheni)
# endforeach()

foreach(ex ivf_pq)
    add_executable(benchmark_${ex} benchmarks/${ex}.cpp)
    target_link_libraries(benchmark_${ex} PRIVATE spheni)
endforeach()

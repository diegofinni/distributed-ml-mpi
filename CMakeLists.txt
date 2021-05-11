cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
SET(GCC_COMPILE_FLAGS "--std=c++11")
add_definitions(${GCC_COMPILE_FLAGS})

project(dml)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(dml serverTest.cpp master_node.hpp master_node.cpp worker_node.hpp worker_node.cpp)
add_executable(lr decentralized.cpp decentralized.hpp lr.cpp lr.hpp main.cpp)
add_executable(lr_seq lr.cpp lr.hpp lr_main.cpp)

target_link_libraries(dml ${MPI_LIBRARIES})
target_link_libraries(lr ${MPI_LIBRARIES})


if(MPI_COMPILE_FLAGS)
    set_target_properties(dml PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(lr PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(dml PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
    set_target_properties(lr PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

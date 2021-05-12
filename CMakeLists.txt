cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
SET(GCC_COMPILE_FLAGS "--std=c++11")
add_definitions(${GCC_COMPILE_FLAGS})

project(dml)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(lr_centralized centralized_main.cpp master_node.hpp master_node.cpp worker_node.hpp worker_node.cpp lr.hpp lr.cpp)
add_executable(lr_decentralized decentralized_main.cpp decentralized.cpp decentralized.hpp lr.cpp lr.hpp)
add_executable(lr_seq sequential_main.cpp lr.cpp lr.hpp)

target_link_libraries(lr_centralized ${MPI_LIBRARIES})
target_link_libraries(lr_decentralized ${MPI_LIBRARIES})


if(MPI_COMPILE_FLAGS)
    set_target_properties(lr_centralized PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(lr_decentralized PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(lr_centralized PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
    set_target_properties(lr_decentralized PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

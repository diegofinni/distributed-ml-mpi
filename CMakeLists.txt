cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)

project(dml)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
include_directories(/Usr/local/include)
include_directories(/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi)

add_executable(dml main.cpp decentralized.hpp decentralized.cpp)
target_link_libraries(dml ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(dml PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(dml PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

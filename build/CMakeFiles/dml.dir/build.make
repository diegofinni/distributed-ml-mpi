# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.20.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.20.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build"

# Include any dependencies generated for this target.
include CMakeFiles/dml.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/dml.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dml.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dml.dir/flags.make

CMakeFiles/dml.dir/serverTest.cpp.o: CMakeFiles/dml.dir/flags.make
CMakeFiles/dml.dir/serverTest.cpp.o: ../serverTest.cpp
CMakeFiles/dml.dir/serverTest.cpp.o: CMakeFiles/dml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dml.dir/serverTest.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dml.dir/serverTest.cpp.o -MF CMakeFiles/dml.dir/serverTest.cpp.o.d -o CMakeFiles/dml.dir/serverTest.cpp.o -c "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/serverTest.cpp"

CMakeFiles/dml.dir/serverTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dml.dir/serverTest.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/serverTest.cpp" > CMakeFiles/dml.dir/serverTest.cpp.i

CMakeFiles/dml.dir/serverTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dml.dir/serverTest.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/serverTest.cpp" -o CMakeFiles/dml.dir/serverTest.cpp.s

CMakeFiles/dml.dir/master_node.cpp.o: CMakeFiles/dml.dir/flags.make
CMakeFiles/dml.dir/master_node.cpp.o: ../master_node.cpp
CMakeFiles/dml.dir/master_node.cpp.o: CMakeFiles/dml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/dml.dir/master_node.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dml.dir/master_node.cpp.o -MF CMakeFiles/dml.dir/master_node.cpp.o.d -o CMakeFiles/dml.dir/master_node.cpp.o -c "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/master_node.cpp"

CMakeFiles/dml.dir/master_node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dml.dir/master_node.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/master_node.cpp" > CMakeFiles/dml.dir/master_node.cpp.i

CMakeFiles/dml.dir/master_node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dml.dir/master_node.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/master_node.cpp" -o CMakeFiles/dml.dir/master_node.cpp.s

CMakeFiles/dml.dir/worker_node.cpp.o: CMakeFiles/dml.dir/flags.make
CMakeFiles/dml.dir/worker_node.cpp.o: ../worker_node.cpp
CMakeFiles/dml.dir/worker_node.cpp.o: CMakeFiles/dml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/dml.dir/worker_node.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dml.dir/worker_node.cpp.o -MF CMakeFiles/dml.dir/worker_node.cpp.o.d -o CMakeFiles/dml.dir/worker_node.cpp.o -c "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/worker_node.cpp"

CMakeFiles/dml.dir/worker_node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dml.dir/worker_node.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/worker_node.cpp" > CMakeFiles/dml.dir/worker_node.cpp.i

CMakeFiles/dml.dir/worker_node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dml.dir/worker_node.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/worker_node.cpp" -o CMakeFiles/dml.dir/worker_node.cpp.s

# Object files for target dml
dml_OBJECTS = \
"CMakeFiles/dml.dir/serverTest.cpp.o" \
"CMakeFiles/dml.dir/master_node.cpp.o" \
"CMakeFiles/dml.dir/worker_node.cpp.o"

# External object files for target dml
dml_EXTERNAL_OBJECTS =

dml: CMakeFiles/dml.dir/serverTest.cpp.o
dml: CMakeFiles/dml.dir/master_node.cpp.o
dml: CMakeFiles/dml.dir/worker_node.cpp.o
dml: CMakeFiles/dml.dir/build.make
dml: /usr/local/Cellar/open-mpi/4.1.0/lib/libmpi.dylib
dml: CMakeFiles/dml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable dml"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dml.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dml.dir/build: dml
.PHONY : CMakeFiles/dml.dir/build

CMakeFiles/dml.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dml.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dml.dir/clean

CMakeFiles/dml.dir/depend:
	cd "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi" "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi" "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build" "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build" "/Users/diegosanmiguel/Documents/Personal Work/afs/private/15418/distributed-ml-mpi/build/CMakeFiles/dml.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/dml.dir/depend


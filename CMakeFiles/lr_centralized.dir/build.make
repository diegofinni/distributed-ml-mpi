# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi

# Include any dependencies generated for this target.
include CMakeFiles/lr_centralized.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lr_centralized.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lr_centralized.dir/flags.make

CMakeFiles/lr_centralized.dir/centralized_main.cpp.o: CMakeFiles/lr_centralized.dir/flags.make
CMakeFiles/lr_centralized.dir/centralized_main.cpp.o: centralized_main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/lr_centralized.dir/centralized_main.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/lr_centralized.dir/centralized_main.cpp.o -c /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/centralized_main.cpp

CMakeFiles/lr_centralized.dir/centralized_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lr_centralized.dir/centralized_main.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/centralized_main.cpp > CMakeFiles/lr_centralized.dir/centralized_main.cpp.i

CMakeFiles/lr_centralized.dir/centralized_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lr_centralized.dir/centralized_main.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/centralized_main.cpp -o CMakeFiles/lr_centralized.dir/centralized_main.cpp.s

CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.requires:
.PHONY : CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.requires

CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.provides: CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.requires
	$(MAKE) -f CMakeFiles/lr_centralized.dir/build.make CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.provides.build
.PHONY : CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.provides

CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.provides.build: CMakeFiles/lr_centralized.dir/centralized_main.cpp.o

CMakeFiles/lr_centralized.dir/lr.cpp.o: CMakeFiles/lr_centralized.dir/flags.make
CMakeFiles/lr_centralized.dir/lr.cpp.o: lr.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/lr_centralized.dir/lr.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/lr_centralized.dir/lr.cpp.o -c /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/lr.cpp

CMakeFiles/lr_centralized.dir/lr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lr_centralized.dir/lr.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/lr.cpp > CMakeFiles/lr_centralized.dir/lr.cpp.i

CMakeFiles/lr_centralized.dir/lr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lr_centralized.dir/lr.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/lr.cpp -o CMakeFiles/lr_centralized.dir/lr.cpp.s

CMakeFiles/lr_centralized.dir/lr.cpp.o.requires:
.PHONY : CMakeFiles/lr_centralized.dir/lr.cpp.o.requires

CMakeFiles/lr_centralized.dir/lr.cpp.o.provides: CMakeFiles/lr_centralized.dir/lr.cpp.o.requires
	$(MAKE) -f CMakeFiles/lr_centralized.dir/build.make CMakeFiles/lr_centralized.dir/lr.cpp.o.provides.build
.PHONY : CMakeFiles/lr_centralized.dir/lr.cpp.o.provides

CMakeFiles/lr_centralized.dir/lr.cpp.o.provides.build: CMakeFiles/lr_centralized.dir/lr.cpp.o

CMakeFiles/lr_centralized.dir/master_node.cpp.o: CMakeFiles/lr_centralized.dir/flags.make
CMakeFiles/lr_centralized.dir/master_node.cpp.o: master_node.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/lr_centralized.dir/master_node.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/lr_centralized.dir/master_node.cpp.o -c /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/master_node.cpp

CMakeFiles/lr_centralized.dir/master_node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lr_centralized.dir/master_node.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/master_node.cpp > CMakeFiles/lr_centralized.dir/master_node.cpp.i

CMakeFiles/lr_centralized.dir/master_node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lr_centralized.dir/master_node.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/master_node.cpp -o CMakeFiles/lr_centralized.dir/master_node.cpp.s

CMakeFiles/lr_centralized.dir/master_node.cpp.o.requires:
.PHONY : CMakeFiles/lr_centralized.dir/master_node.cpp.o.requires

CMakeFiles/lr_centralized.dir/master_node.cpp.o.provides: CMakeFiles/lr_centralized.dir/master_node.cpp.o.requires
	$(MAKE) -f CMakeFiles/lr_centralized.dir/build.make CMakeFiles/lr_centralized.dir/master_node.cpp.o.provides.build
.PHONY : CMakeFiles/lr_centralized.dir/master_node.cpp.o.provides

CMakeFiles/lr_centralized.dir/master_node.cpp.o.provides.build: CMakeFiles/lr_centralized.dir/master_node.cpp.o

CMakeFiles/lr_centralized.dir/worker_node.cpp.o: CMakeFiles/lr_centralized.dir/flags.make
CMakeFiles/lr_centralized.dir/worker_node.cpp.o: worker_node.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/lr_centralized.dir/worker_node.cpp.o"
	/usr/lib64/ccache/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/lr_centralized.dir/worker_node.cpp.o -c /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/worker_node.cpp

CMakeFiles/lr_centralized.dir/worker_node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lr_centralized.dir/worker_node.cpp.i"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/worker_node.cpp > CMakeFiles/lr_centralized.dir/worker_node.cpp.i

CMakeFiles/lr_centralized.dir/worker_node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lr_centralized.dir/worker_node.cpp.s"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/worker_node.cpp -o CMakeFiles/lr_centralized.dir/worker_node.cpp.s

CMakeFiles/lr_centralized.dir/worker_node.cpp.o.requires:
.PHONY : CMakeFiles/lr_centralized.dir/worker_node.cpp.o.requires

CMakeFiles/lr_centralized.dir/worker_node.cpp.o.provides: CMakeFiles/lr_centralized.dir/worker_node.cpp.o.requires
	$(MAKE) -f CMakeFiles/lr_centralized.dir/build.make CMakeFiles/lr_centralized.dir/worker_node.cpp.o.provides.build
.PHONY : CMakeFiles/lr_centralized.dir/worker_node.cpp.o.provides

CMakeFiles/lr_centralized.dir/worker_node.cpp.o.provides.build: CMakeFiles/lr_centralized.dir/worker_node.cpp.o

# Object files for target lr_centralized
lr_centralized_OBJECTS = \
"CMakeFiles/lr_centralized.dir/centralized_main.cpp.o" \
"CMakeFiles/lr_centralized.dir/lr.cpp.o" \
"CMakeFiles/lr_centralized.dir/master_node.cpp.o" \
"CMakeFiles/lr_centralized.dir/worker_node.cpp.o"

# External object files for target lr_centralized
lr_centralized_EXTERNAL_OBJECTS =

lr_centralized: CMakeFiles/lr_centralized.dir/centralized_main.cpp.o
lr_centralized: CMakeFiles/lr_centralized.dir/lr.cpp.o
lr_centralized: CMakeFiles/lr_centralized.dir/master_node.cpp.o
lr_centralized: CMakeFiles/lr_centralized.dir/worker_node.cpp.o
lr_centralized: CMakeFiles/lr_centralized.dir/build.make
lr_centralized: /usr/local/depot/openmpi-4.0.2/lib/libmpi.so
lr_centralized: CMakeFiles/lr_centralized.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable lr_centralized"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lr_centralized.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lr_centralized.dir/build: lr_centralized
.PHONY : CMakeFiles/lr_centralized.dir/build

CMakeFiles/lr_centralized.dir/requires: CMakeFiles/lr_centralized.dir/centralized_main.cpp.o.requires
CMakeFiles/lr_centralized.dir/requires: CMakeFiles/lr_centralized.dir/lr.cpp.o.requires
CMakeFiles/lr_centralized.dir/requires: CMakeFiles/lr_centralized.dir/master_node.cpp.o.requires
CMakeFiles/lr_centralized.dir/requires: CMakeFiles/lr_centralized.dir/worker_node.cpp.o.requires
.PHONY : CMakeFiles/lr_centralized.dir/requires

CMakeFiles/lr_centralized.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lr_centralized.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lr_centralized.dir/clean

CMakeFiles/lr_centralized.dir/depend:
	cd /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi /afs/andrew.cmu.edu/usr4/nathanan/private/15418/finalProject/distributed-ml-mpi/CMakeFiles/lr_centralized.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lr_centralized.dir/depend


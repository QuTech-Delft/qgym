cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(QGym C CXX)

# If QGym was already included elsewhere in the project, don't include it
# again. There should be only one place for it and one version per project.
if(NOT TARGET qgym)

# Loads up the appropriate directories for installing stuff.
include(GNUInstallDirs)


#=============================================================================#
# Configuration options                                                       #
#=============================================================================#

# Library type option. Default is a shared object, because for CMake it doesn't
# matter, but outside of CMake dependency information is lost for static
# libraries. That requires the user to link all of qgym's direct and transitive
# dependencies as well, which is terribly ugly. setup.py *has* to do this
# however, because "pip install ." builds this in a temporary directory, so the
# shared objects that get built and installed and are then depended on by the
# Python lib get deleted by pip after the install.
option(
    BUILD_SHARED_LIBS
    "Whether libraries should be built as a shared object or as a static library"
    ON
)

# With what optimization level the library is to be built.
if(NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE Release CACHE STRING
        "Type of build (Debug, Release)" FORCE
    )
    set_property(
        CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release"
    )
endif()

# Whether the Python module should be built. This should only be enabled for
# setup.py's builds.
option(
    QGYM_BUILD_PYTHON
    "Whether the Python module should be built"
    OFF
)
mark_as_advanced(QGYM_BUILD_PYTHON)

# Where the Python module should be built.
set(
    QGYM_PYTHON_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/python/qgym"
    CACHE STRING "Where to install the Python library"
)
mark_as_advanced(QGYM_PYTHON_DIR)

# Used to override the (base)name of the Python extension.
set(
    QGYM_PYTHON_EXT ""
    CACHE STRING "Basename for the Python extension, or \"\" to let CMake's SWIG implementation handle it"
)
mark_as_advanced(QGYM_PYTHON_EXT)


#=============================================================================#
# CMake weirdness and compatibility                                           #
#=============================================================================#

# On Windows builds, CMake complains that the CMake internal variable
# "CMAKE_MODULE_LINKER_FLAGS_MAINTAINER" is not defined *the first time you
# configure*. Weirdly, all is good with the world if you then just try again.
# It seems to have to do with the "maintainer" build type in MSVC, but there
# is no documentation whatsoever. In any case, this just mimics what CMake
# does automatically the second time you invoke it, so it works around the
# issue.
if(NOT DEFINED CMAKE_MODULE_LINKER_FLAGS_MAINTAINER)
    set(
        CMAKE_MODULE_LINKER_FLAGS_MAINTAINER ""
        CACHE STRING "Flags used by the linker during the creation of modules during MAINTAINER builds."
    )
endif()


#=============================================================================#
# Global build configuration                                                  #
#=============================================================================#

# Since we have multiple libraries to link together, we unfortunately have to
# worry about RPATH handling on Linux and OSX. See
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Everything needs C++11 support.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows weirdness: need a .lib file to link against a DLL at compile-time
# (I think), but only the .dll is generated when there are no exported symbols.
# This sort of fixes that (https://stackoverflow.com/questions/1941443)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


#=============================================================================#
# Compile code generators                                                     #
#=============================================================================#

# Simple program that generates resource include files, to include the contents
# of a file as a constant inside the QGym library.
# add_executable(resource src/resource/main.cpp)
function(create_resource fname)
    set(infile "${CMAKE_CURRENT_SOURCE_DIR}/${fname}")
    get_filename_component(outdir "${CMAKE_CURRENT_BINARY_DIR}/${fname}" DIRECTORY)
    get_filename_component(outname "${CMAKE_CURRENT_BINARY_DIR}/${fname}" NAME_WE)
    set(outfile "${outdir}/${outname}.inc")
    add_custom_command(
        OUTPUT "${outfile}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}"
        COMMAND resource "${infile}" "${outdir}" "${outname}"
        DEPENDS "${infile}" resource
    )
    string(REGEX REPLACE "[\\./\\]" "_" target_name "${fname}")
    add_custom_target(
        ${target_name}
        DEPENDS "${outfile}"
    )
    add_dependencies(qgym ${target_name})
endfunction()


# #=============================================================================#
# # QGym library target                                                       #
# #=============================================================================#

# Create the QGym library. This will be built either as a shared object/DLL
# or as a static library based on BUILD_SHARED_LIBS; add_library switches
# automatically.
add_library(qgym
    "${CMAKE_CURRENT_SOURCE_DIR}/src/qgym/mapping.cc"
)

# There is no distinction between public and private header files right now,
# and they're all in the source directory. Note the / at the end of the path;
# this is necessary for the header files to be installed in the right location.
target_include_directories(qgym
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/src/"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include/"
)

# Configure compilation.
set_property(TARGET qgym PROPERTY POSITION_INDEPENDENT_CODE ON)
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(qgym PRIVATE -Wall -Wfatal-errors -ggdb)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    target_compile_options(qgym PRIVATE -Wall -Wfatal-errors -ggdb -Wno-unused-local-typedef)
elseif(MSVC)
    target_compile_options(qgym PRIVATE /MP /D_USE_MATH_DEFINES /EHsc /bigobj)
else()
    message(SEND_ERROR "Unknown compiler!")
endif()

# Enable optimizations only for release builds.
if(NOT MSVC AND "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    target_compile_options(qgym PRIVATE -O3)
endif()

#=============================================================================#
# Configure, build, and link dependencies                                     #
#=============================================================================#

# pthreads --------------------------------------------------------------------

# Look for thread support library (pthreads)
find_package(Threads REQUIRED)
target_link_libraries(qgym PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# openql -----------
add_subdirectory(deps/openql)
target_link_libraries(qgym PRIVATE ql)

# backward-cpp ----------------------------------------------------------------

# Stack trace helper library, nothing functional here.
# We don't need to add again since openql already added this.
add_backward(qgym)

# add_backward doesn't set INTERFACE_LINK_LIBRARIES, only LINK_LIBRARIES. That
# goes wrong when we're compiling statically, because said libraries are shared
# and need to be included in the final link.
set_property(TARGET qgym APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${BACKWARD_LIBRARIES})

#=============================================================================#
# Python module                                                               #
#=============================================================================#

# Include the python directory
if(QGYM_BUILD_PYTHON)
    add_subdirectory(python)
endif()


#=============================================================================#
# Installation                                                                #
#=============================================================================#

# Install targets for the QGym library.
install(
    TARGETS qgym
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
    DIRECTORY "$<TARGET_PROPERTY:qgym,INTERFACE_INCLUDE_DIRECTORIES>"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)


endif() # NOT TARGET qgym

cmake_minimum_required(VERSION 3.15...3.27)
project(Rex VERSION 1.1.0 LANGUAGES CXX)

# Building the Python bindings is strictly opt-in: everything that touches
# Python or pybind11 lives behind this flag, so a bare `cmake -S Rex -B build`
# never needs a Python interpreter, dev headers, or network access to fetch
# pybind11. C++-only consumers of Rex/teaRex (including the plain `makefile`
# build, left untouched alongside this file) are unaffected either way.
option(REX_BUILD_PYTHON "Build the pybind11 Python bindings for Rex/teaRex" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

add_library(rex SHARED src/Rex.cc)
target_include_directories(
    rex PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include>
)
target_compile_features(rex PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
add_library(tearex SHARED src/teaRex.cc)
target_include_directories(
    tearex PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                  $<INSTALL_INTERFACE:include>
)
target_compile_features(tearex PUBLIC cxx_std_17)
target_link_libraries(tearex PUBLIC rex Threads::Threads)
# librex/libtearex always land in the same directory as each other (both
# plain `cmake --install` and the wheel layout put them side by side), so
# tearex only needs to look next to itself to find rex at runtime. Without
# this, install(TARGETS ...) strips the build-tree RPATH and libtearex.so
# can't locate librex.so post-install.
# ($ORIGIN is the ELF spelling; macOS's dynamic loader wants @loader_path.)
if(APPLE)
    set(REX_RPATH_ORIGIN "@loader_path")
else()
    set(REX_RPATH_ORIGIN "$ORIGIN")
endif()
set_target_properties(tearex PROPERTIES INSTALL_RPATH "${REX_RPATH_ORIGIN}")

install(TARGETS rex tearex)
install(FILES include/Rex.h include/teaRex.h DESTINATION include)

########################################################################################
# Python bindings (optional)                                                          #
########################################################################################

if(REX_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    include(FetchContent)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v3.0.1
    )
    FetchContent_MakeAvailable(pybind11)

    python_add_library(
        _rex_py
        MODULE
        src/python/rex.cpp
        src/python/tearex.cpp
        src/python/tearex.hpp
        WITH_SOABI
    )
    target_link_libraries(_rex_py PRIVATE pybind11::headers)
    target_link_libraries(_rex_py PUBLIC rex tearex)
    # _rex_py is installed one level below librex/libtearex (pyrex/_rex_py.so
    # vs lib/librex.so) in both the wheel layout and a plain `cmake
    # --install`, so it needs to look up one directory then into lib/.
    set_target_properties(_rex_py PROPERTIES INSTALL_RPATH "${REX_RPATH_ORIGIN}/../lib")
    install(TARGETS _rex_py DESTINATION pyrex)
endif()

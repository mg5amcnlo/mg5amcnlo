cmake_minimum_required (VERSION 3.15...3.27)
project (
    madtrex
    VERSION 0.1
    LANGUAGES CXX
)

# MadtRex is layered strictly on top of madspace: it builds against an
# already-installed madspace (headers + libmadspace) and the bundled Rex fork
# in Rex/, and madspace knows nothing about it.

set(CMAKE_POSITION_INDEPENDENT_CODE 1)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "Enabling LTO")
    endif()
endif()

add_compile_options(-fno-omit-frame-pointer)

########################################################################################
# Load dependencies                                                                    #
########################################################################################

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
include(FetchContent)

# Must stay identical to madspace/CMakeLists.txt (and Rex/CMakeLists.txt):
# _madtrex_py takes and returns types registered by _madspace_py and pyrex,
# and pybind11 only shares registered types between modules built against the
# same internals version.
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

find_package(Threads REQUIRED)

# The installed madspace package (the directory holding its __init__.py, with
# include/ and lib/ inside it). Found by importing madspace with the build's
# Python unless given explicitly with -DMADSPACE_DIR=...
if(NOT MADSPACE_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import madspace, os; print(os.path.dirname(os.path.abspath(madspace.__file__)))"
        OUTPUT_VARIABLE MADSPACE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE madspace_import_result
        ERROR_VARIABLE madspace_import_error
    )
    if(NOT madspace_import_result EQUAL 0)
        message(FATAL_ERROR
            "Could not import madspace with ${Python_EXECUTABLE}:\n"
            "${madspace_import_error}\n"
            "Install madspace first (python madspace/install.py) and make it "
            "importable (e.g. PYTHONPATH=madspace/install), or pass "
            "-DMADSPACE_DIR=/path/to/site-packages/madspace."
        )
    endif()
endif()
set(MADSPACE_DIR "${MADSPACE_DIR}" CACHE PATH "Installed madspace package directory")
message(STATUS "Using madspace from ${MADSPACE_DIR}")

find_library(
    MADSPACE_LIBRARY
    NAMES madspace
    PATHS "${MADSPACE_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
add_library(madspace::madspace SHARED IMPORTED)
set_target_properties(
    madspace::madspace
    PROPERTIES
        IMPORTED_LOCATION "${MADSPACE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MADSPACE_DIR}/include;${MADSPACE_DIR}/include/madspace/compgraphs"
        INTERFACE_COMPILE_FEATURES cxx_std_20
)

########################################################################################
# Python bindings                                                                      #
########################################################################################

python_add_library(
    _madtrex_py
    MODULE
    src/python/madtrex.cpp
    src/tupperware.cpp
    src/madtrex.cpp
    src/binary_io.cpp
    src/param_handler.cpp
    src/pdfware.cpp
    Rex/src/Rex.cc
    Rex/src/teaRex.cc
    WITH_SOABI
)
target_include_directories(_madtrex_py PRIVATE include Rex/include)
target_compile_features(_madtrex_py PRIVATE cxx_std_20)
target_link_libraries(_madtrex_py PRIVATE pybind11::headers)
target_link_libraries(_madtrex_py PRIVATE madspace::madspace)
target_link_libraries(_madtrex_py PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(_madtrex_py PRIVATE Threads::Threads)
# madtrex/__init__.py imports madspace (which preloads libmadspace) before
# _madtrex_py, but point the RPATH at libmadspace as well, so the module also
# loads on its own: first relative to the module, for the side-by-side
# madtrex/install/madtrex <-> madspace/install/madspace layout that MadGraph
# and its gridpacks use (and which survives moving the whole tree), then at
# the very libmadspace this was built against.
if(APPLE)
    set(madtrex_origin "@loader_path")
else()
    set(madtrex_origin "$ORIGIN")
endif()
set_target_properties(
    _madtrex_py
    PROPERTIES
        INSTALL_RPATH "${madtrex_origin}/../../../madspace/install/madspace/lib;${MADSPACE_DIR}/lib"
        BUILD_RPATH "${MADSPACE_DIR}/lib"
)
install(TARGETS _madtrex_py DESTINATION madtrex)

# The madspace headers _madtrex_py is compiled against (its classes and their
# inline methods are baked in): record a digest of them, which
# madtrex/_compat.py compares with the madspace found at import time, so that
# a madspace reinstalled since is refused with a clear message instead of
# failing as a binary mismatch. Same digest as _compat.headers_digest(): one
# "<relative path> <sha256>" line per file, in sorted path order.
file(GLOB_RECURSE madspace_headers LIST_DIRECTORIES false
     RELATIVE "${MADSPACE_DIR}/include" "${MADSPACE_DIR}/include/*")
list(SORT madspace_headers)
set(madspace_headers_lines "")
foreach(header IN LISTS madspace_headers)
    file(SHA256 "${MADSPACE_DIR}/include/${header}" header_sha)
    string(APPEND madspace_headers_lines "${header} ${header_sha}\n")
endforeach()
string(SHA256 MADSPACE_HEADERS_DIGEST "${madspace_headers_lines}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json"
     "{\"madspace_headers\": \"${MADSPACE_HEADERS_DIGEST}\", \"madspace_dir\": \"${MADSPACE_DIR}\"}\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" DESTINATION madtrex)

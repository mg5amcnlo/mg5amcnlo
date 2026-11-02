cmake_minimum_required (VERSION 3.15...3.27)
project (
    madspace
    VERSION 0.1
    LANGUAGES CXX
)

option(ENABLE_CUDA "Enable CUDA GPU backend" OFF)
option(ENABLE_HIP "Enable HIP/ROCm GPU backend" OFF)
option(ENABLE_SIMD "Enable SIMD CPU acceleration" OFF)
# Extract API docstrings from the Doxygen comments (needs doxygen) and generate
# the .pyi type stubs. Off for internal MadGraph use; on for the wheels
# published to PyPI.
option(ENABLE_DOCS "Generate API docstrings and .pyi type stubs" OFF)
if(APPLE)
    option(ENABLE_OPENBLAS "Build OpenBLAS from source (default: OFF on Apple, ON elsewhere)" OFF)
else()
    option(ENABLE_OPENBLAS "Build OpenBLAS from source (default: OFF on Apple, ON elsewhere)" ON)
endif()

if(ENABLE_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;120")
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(COMPILE_CUDA 1)
    else()
        message(FATAL_ERROR "ENABLE_CUDA is ON but no CUDA compiler was found")
    endif()
endif()

if(ENABLE_HIP)
    if (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1150;gfx1151")
    endif()
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        set(COMPILE_HIP 1)
    else()
        message(FATAL_ERROR "ENABLE_HIP is ON but no HIP compiler was found")
    endif()
endif()

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
include(ExternalProject)

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

if(COMPILE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
if(COMPILE_HIP)
    find_package(rocblas REQUIRED)
    find_package(rocthrust REQUIRED)
    find_package(hipcub REQUIRED)
endif()

if(ENABLE_SIMD)
    set(
        sleef_libname
        ${CMAKE_STATIC_LIBRARY_PREFIX}sleef${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    ExternalProject_Add(libsleef
        GIT_REPOSITORY https://github.com/shibatch/sleef
        GIT_TAG 3.9.0
        UPDATE_DISCONNECTED true
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DSLEEF_BUILD_TESTS=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=1
            -DSLEEF_ENABLE_SSL=OFF
        INSTALL_BYPRODUCTS <INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}/${sleef_libname}
    )
    ExternalProject_Get_Property(libsleef install_dir)
    add_library(sleef::sleef INTERFACE IMPORTED GLOBAL)
    add_dependencies(sleef::sleef libsleef)
    file(MAKE_DIRECTORY ${install_dir}/include)
    target_include_directories(sleef::sleef INTERFACE ${install_dir}/include)
    target_link_libraries(
        sleef::sleef
        INTERFACE
        ${install_dir}/${CMAKE_INSTALL_LIBDIR}/${sleef_libname}
    )
    #find_package(sleef CONFIG REQUIRED)
endif()

if(ENABLE_OPENBLAS)
    set(
        openblas_libname
        ${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    ExternalProject_Add(libopenblas
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS
        GIT_TAG v0.3.33
        UPDATE_DISCONNECTED true
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_WITHOUT_LAPACK=ON
            -DBUILD_SINGLE=ON
            -DBUILD_DOUBLE=ON
            -DBUILD_COMPLEX=OFF
            -DBUILD_COMPLEX16=OFF
            -DBUILD_TESTING=OFF
            -DBUILD_WITHOUT_CBLAS=ON
            -DUSE_THREAD=OFF
            -DUSE_LOCKING=ON
            -DDYNAMIC_ARCH=ON
            -DCMAKE_POSITION_INDEPENDENT_CODE=1
        INSTALL_BYPRODUCTS <INSTALL_DIR>/lib/${openblas_libname}
    )
    ExternalProject_Get_Property(libopenblas install_dir)
    add_library(madspace::blas INTERFACE IMPORTED GLOBAL)
    add_dependencies(madspace::blas libopenblas)
    file(MAKE_DIRECTORY ${install_dir}/include)
    target_include_directories(madspace::blas INTERFACE ${install_dir}/include)
    target_link_libraries(
        madspace::blas
        INTERFACE
        ${install_dir}/lib/${openblas_libname}
    )
else()
    set(BLA_STATIC ON)
    set(BLA_SIZEOF_INTEGER 4)
    find_package(BLAS REQUIRED)
    add_library(madspace::blas INTERFACE IMPORTED GLOBAL)
    target_link_libraries(madspace::blas INTERFACE BLAS::BLAS)
endif()

########################################################################################
# Code generation                                                                      #
########################################################################################

# Headers are generated into the build tree (not the source tree), each kind
# in its own subdirectory added to the relevant targets' include path below.
set(MADSPACE_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(
    MADSPACE_GENERATED_HEADERS
    ${MADSPACE_GENERATED_DIR}/compgraphs/function_builder_mixin.inc
    ${MADSPACE_GENERATED_DIR}/compgraphs/opcode_mixin.inc
    ${MADSPACE_GENERATED_DIR}/compgraphs/instruction_set_mixin.inc
    ${MADSPACE_GENERATED_DIR}/python/instruction_set.hpp
    ${MADSPACE_GENERATED_DIR}/cpu/runtime_mixin.inc
    ${MADSPACE_GENERATED_DIR}/cpu/runtime_backward_mixin.inc
    ${MADSPACE_GENERATED_DIR}/cpu/kernel_checks.inc
    ${MADSPACE_GENERATED_DIR}/gpu/runtime_mixin.inc
    ${MADSPACE_GENERATED_DIR}/gpu/runtime_backward_mixin.inc
)

# Every target below lists one of these files as a source, so CMake reruns
# this command (and thus regenerates all of them together) before building
# anything that needs them, but only when instruction_set.yaml or the
# generator itself changes.
add_custom_command(
    OUTPUT ${MADSPACE_GENERATED_HEADERS}
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_headers.py"
        --out-dir "${MADSPACE_GENERATED_DIR}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/instruction_set.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/generate_headers.py"
    COMMENT "Generating madspace instruction-set headers from instruction_set.yaml"
    VERBATIM
)

########################################################################################
# Main madspace module                                                                 #
########################################################################################

add_library(
    madspace
    SHARED
    src/driver/adam_optimizer.cpp
    src/driver/format.cpp
    src/driver/tensor.cpp
    src/driver/context.cpp
    src/driver/event_generator.cpp
    src/driver/channel_generator.cpp
    src/driver/generator_data.cpp
    src/driver/status_file.cpp
    src/driver/thread_pool.cpp
    src/driver/io.cpp
    src/driver/tar.cpp
    src/driver/vegas_optimizer.cpp
    src/driver/discrete_optimizer.cpp
    src/driver/backend.cpp
    src/driver/lhe_output.cpp
    src/driver/systematics.cpp
    src/driver/event_histograms.cpp
    src/driver/madnis_training.cpp
    src/driver/random.cpp
    src/compgraphs/type.cpp
    src/compgraphs/function.cpp
    src/compgraphs/instruction.cpp
    src/compgraphs/optimizer.cpp
    ${MADSPACE_GENERATED_DIR}/compgraphs/instruction_set_mixin.inc
    src/phasespace/base.cpp
    src/phasespace/batch_sampler.cpp
    src/phasespace/invariants.cpp
    src/phasespace/two_particle.cpp
    src/phasespace/three_particle.cpp
    src/phasespace/luminosity.cpp
    src/phasespace/topology.cpp
    src/phasespace/t_propagator_mapping.cpp
    src/phasespace/color_ordered_mapping.cpp
    src/phasespace/phasespace.cpp
    src/phasespace/multichannel.cpp
    src/phasespace/madnis.cpp
    src/phasespace/rambo.cpp
    src/phasespace/cuts.cpp
    src/phasespace/custom_function.cpp
    src/phasespace/observable.cpp
    src/phasespace/histograms.cpp
    src/phasespace/chili.cpp
    src/phasespace/integrand.cpp
    src/phasespace/unweighter.cpp
    src/phasespace/vegas.cpp
    src/phasespace/mlp.cpp
    src/phasespace/flow.cpp
    src/phasespace/discrete_flow.cpp
    src/phasespace/discrete_sampler.cpp
    src/phasespace/channel_weights.cpp
    src/phasespace/channel_weight_network.cpp
    src/phasespace/pdf.cpp
    src/phasespace/matrix_element.cpp
    src/phasespace/mlm_clustering.cpp
    src/phasespace/cross_section.cpp
    src/phasespace/scale.cpp
    include/madspace/util.hpp
    include/madspace/driver.hpp
    include/madspace/driver/adam_optimizer.hpp
    include/madspace/driver/format.hpp
    include/madspace/driver/tensor.hpp
    include/madspace/driver/context.hpp
    include/madspace/driver/thread_pool.hpp
    include/madspace/driver/event_generator.hpp
    include/madspace/driver/channel_generator.hpp
    include/madspace/driver/generator_data.hpp
    include/madspace/driver/io.hpp
    include/madspace/driver/tar.hpp
    include/madspace/driver/vegas_optimizer.hpp
    include/madspace/driver/discrete_optimizer.hpp
    include/madspace/driver/backend.hpp
    include/madspace/driver/lhe_output.hpp
    include/madspace/driver/logger.hpp
    include/madspace/driver/madnis_training.hpp
    include/madspace/driver/random.hpp
    include/madspace/compgraphs.hpp
    include/madspace/compgraphs/type.hpp
    include/madspace/compgraphs/function.hpp
    ${MADSPACE_GENERATED_DIR}/compgraphs/function_builder_mixin.inc
    ${MADSPACE_GENERATED_DIR}/compgraphs/opcode_mixin.inc
    include/madspace/compgraphs/instruction.hpp
    include/madspace/compgraphs/optimizer.hpp
    include/madspace/phasespace.hpp
    include/madspace/phasespace/base.hpp
    include/madspace/phasespace/batch_sampler.hpp
    include/madspace/phasespace/invariants.hpp
    include/madspace/phasespace/two_particle.hpp
    include/madspace/phasespace/three_particle.hpp
    include/madspace/phasespace/luminosity.hpp
    include/madspace/phasespace/topology.hpp
    include/madspace/phasespace/t_propagator_mapping.hpp
    include/madspace/phasespace/color_ordered_mapping.hpp
    include/madspace/phasespace/phasespace.hpp
    include/madspace/phasespace/multichannel.hpp
    include/madspace/phasespace/madnis.hpp
    include/madspace/phasespace/rambo.hpp
    include/madspace/phasespace/cuts.hpp
    include/madspace/phasespace/observable.hpp
    include/madspace/phasespace/histograms.hpp
    include/madspace/phasespace/chili.hpp
    include/madspace/phasespace/integrand.hpp
    include/madspace/phasespace/unweighter.hpp
    include/madspace/phasespace/vegas.hpp
    include/madspace/phasespace/mlp.hpp
    include/madspace/phasespace/flow.hpp
    include/madspace/phasespace/discrete_flow.hpp
    include/madspace/phasespace/discrete_sampler.hpp
    include/madspace/phasespace/channel_weights.hpp
    include/madspace/phasespace/channel_weight_network.hpp
    include/madspace/phasespace/pdf.hpp
    include/madspace/phasespace/matrix_element.hpp
    include/madspace/phasespace/mlm_clustering.hpp
    include/madspace/phasespace/cross_section.hpp
    include/madspace/phasespace/scale.hpp
    include/madspace/mixmax/mixmax.hpp
    include/madspace/mixmax/mixmax_skip_N17.c
)

#target_compile_options(madspace PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(madspace PUBLIC include)
# PUBLIC: function.hpp/instruction.hpp #include these by quoted name, and
# that lookup also has to succeed for anything linking against madspace.
target_include_directories(madspace PUBLIC ${MADSPACE_GENERATED_DIR}/compgraphs)
target_compile_features(madspace PUBLIC cxx_std_20)
target_link_libraries(madspace PUBLIC nlohmann_json::nlohmann_json)
if (ENABLE_SIMD)
    target_compile_definitions(madspace PRIVATE SIMD_AVAILABLE=1)
endif()

install(TARGETS madspace DESTINATION madspace/lib)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "madspace/include"
    # *.h (not just *.hpp*) so umami.h -- the public UMAMI ABI header needed by
    # anyone implementing or consuming a MatrixElementApi-compatible shared
    # library -- is installed too, and *.c for mixmax/mixmax_skip_N17.c, which
    # mixmax.hpp #includes.
    FILES_MATCHING PATTERN "*.hpp*" PATTERN "*.h" PATTERN "*.c"
)
# Generated public headers: installed alongside function.hpp/instruction.hpp
# so a downstream #include "function_builder_mixin.inc" resolves the same
# way it does in this build (see MADSPACE_GENERATED_DIR above).
install(
    FILES
        ${MADSPACE_GENERATED_DIR}/compgraphs/function_builder_mixin.inc
        ${MADSPACE_GENERATED_DIR}/compgraphs/opcode_mixin.inc
    DESTINATION "madspace/include/madspace/compgraphs"
)


########################################################################################
# Compute kernel source files                                                          #
########################################################################################

set(
    KERNEL_SRC
    src/kernels/invariants.hpp
    src/kernels/kernels.hpp
    src/kernels/kinematics.hpp
    src/kernels/multichannel.hpp
    src/kernels/math.hpp
    src/kernels/rambo.hpp
    src/kernels/cuts.hpp
    src/kernels/chili.hpp
    src/kernels/mlm.hpp
    src/kernels/nn.hpp
    src/kernels/observables.hpp
    src/kernels/discrete.hpp
    src/kernels/vegas.hpp
    src/kernels/util.hpp
    src/kernels/definitions.hpp
    src/kernels/twoparticle.hpp
    src/kernels/threeparticle.hpp
    src/kernels/multichannel.hpp
    src/kernels/operations.hpp
)


########################################################################################
# GPU backends                                                                         #
########################################################################################

if(COMPILE_CUDA)
    add_library(
        madspace_cuda
        SHARED
        src/gpu/runtime.cu
        src/gpu/device.cu
        src/gpu/kernel_definitions.cuh
        src/gpu/runtime.hpp
        src/gpu/tensor.cuh
        src/gpu/device.hpp
        src/gpu/gpu_abstraction.cuh
        src/gpu/random.cuh
        ${MADSPACE_GENERATED_DIR}/gpu/runtime_mixin.inc
        ${MADSPACE_GENERATED_DIR}/gpu/runtime_backward_mixin.inc
        ${KERNEL_SRC}
    )
    target_include_directories(madspace_cuda PUBLIC include)
    target_include_directories(madspace_cuda PRIVATE ${MADSPACE_GENERATED_DIR}/gpu)
    set_property(TARGET madspace_cuda PROPERTY CUDA_STANDARD 20)
    target_link_libraries(madspace_cuda PRIVATE madspace)
    target_link_libraries(madspace_cuda PRIVATE CUDA::cublas)
    install(TARGETS madspace_cuda DESTINATION madspace/lib)
endif()

if(COMPILE_HIP)
    add_library(
        madspace_hip
        SHARED
        src/gpu/runtime.hip
        src/gpu/device.hip
        src/gpu/kernel_definitions.cuh
        src/gpu/runtime.hpp
        src/gpu/tensor.cuh
        src/gpu/device.hpp
        src/gpu/gpu_abstraction.cuh
        src/gpu/random.cuh
        ${MADSPACE_GENERATED_DIR}/gpu/runtime_mixin.inc
        ${MADSPACE_GENERATED_DIR}/gpu/runtime_backward_mixin.inc
        ${KERNEL_SRC}
    )
    target_include_directories(madspace_hip PUBLIC include)
    target_include_directories(madspace_hip PRIVATE ${MADSPACE_GENERATED_DIR}/gpu)
    set_property(TARGET madspace_hip PROPERTY HIP_STANDARD 20)
    set_property(TARGET madspace_hip PROPERTY HIP_EXTENSIONS OFF)
    target_link_libraries(madspace_hip PRIVATE madspace)
    target_link_libraries(madspace_hip PRIVATE hip::device)
    target_link_libraries(madspace_hip PRIVATE roc::rocblas)
    target_link_libraries(madspace_hip PRIVATE roc::rocthrust)
    target_link_libraries(madspace_hip PRIVATE hip::hipcub)
    install(TARGETS madspace_hip DESTINATION madspace/lib)
endif()


########################################################################################
# CPU backend                                                                          #
########################################################################################

if(NOT ENABLE_SIMD)
    set(CPU_TARGETS madspace_cpu)
    set(SIMD_FLAGS "")
    set(SIMD_DEFINES "")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    set(CPU_TARGETS madspace_cpu madspace_cpu_simd_128)
    set(SIMD_FLAGS "" "")
    set(SIMD_DEFINES "" "USE_SIMD_NEON=1")
else()
    set(
        CPU_TARGETS
        madspace_cpu madspace_cpu_simd_256 madspace_cpu_avx512y madspace_cpu_simd_512
    )
    set(
        SIMD_FLAGS
        ""
        "-mavx2 -mfma"
        "-mavx2 -mfma -mavx512f -mavx512vl -mavx512dq -mprefer-vector-width=256"
        "-mavx512f"
    )
    set(
        SIMD_DEFINES
        "" "USE_SIMD_AVX2=1" "USE_SIMD_AVX512Y=1" "USE_SIMD_AVX512=1"
    )
endif()

foreach(
    cpu_target simd_flag simd_define
    IN ZIP_LISTS CPU_TARGETS SIMD_FLAGS SIMD_DEFINES
)
    add_library(
        ${cpu_target}
        SHARED
        src/cpu/runtime.hpp
        src/cpu/device.hpp
        src/cpu/runtime.cpp
        src/cpu/device.cpp
        src/cpu/kernel_definitions.hpp
        ${MADSPACE_GENERATED_DIR}/cpu/runtime_mixin.inc
        ${MADSPACE_GENERATED_DIR}/cpu/runtime_backward_mixin.inc
        ${MADSPACE_GENERATED_DIR}/cpu/kernel_checks.inc
        src/cpu/tensor.hpp
        src/cpu/simd.hpp
        src/cpu/simd_arm.hpp
        src/cpu/simd_x86_256.hpp
        src/cpu/simd_x86_256y.hpp
        src/cpu/simd_x86_512.hpp
        ${KERNEL_SRC}
    )
    target_include_directories(${cpu_target} PUBLIC include)
    target_include_directories(${cpu_target} PRIVATE ${MADSPACE_GENERATED_DIR}/cpu)
    target_compile_features(${cpu_target} PUBLIC cxx_std_20)
    target_link_libraries(${cpu_target} PRIVATE madspace::blas)
    target_link_libraries(${cpu_target} PRIVATE madspace)
    if(NOT cpu_target STREQUAL "madspace_cpu")
        target_link_libraries(${cpu_target} PRIVATE sleef::sleef)
        target_compile_definitions(${cpu_target} PRIVATE ${simd_define})
        separate_arguments(simd_flag UNIX_COMMAND ${simd_flag})
        target_compile_options(${cpu_target} PRIVATE ${simd_flag})
        target_compile_definitions(${cpu_target} PRIVATE USE_SIMD=1)
    endif()
    install(TARGETS ${cpu_target} DESTINATION madspace/lib)
endforeach()


########################################################################################
# Python bindings                                                                      #
########################################################################################

# docstrings.hpp gives madspace.cpp the pydoc::doc() lookup. With ENABLE_DOCS
# it is filled from the Doxygen comments in the public headers (needs doxygen);
# otherwise a stub is written whose doc() returns "" and no doxygen is needed.
if(ENABLE_DOCS)
    file(GLOB_RECURSE MADSPACE_PUBLIC_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/madspace/*.hpp"
    )
    set(MADSPACE_DOCSTRINGS_ARGS
        --include-dir "${CMAKE_CURRENT_SOURCE_DIR}/include/madspace"
        --out "${MADSPACE_GENERATED_DIR}/python/docstrings.hpp"
        --require-doxygen
    )
    set(MADSPACE_DOCSTRINGS_DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/generate_docstrings.py"
        ${MADSPACE_PUBLIC_HEADERS}
    )
    set(MADSPACE_DOCSTRINGS_COMMENT
        "Generating madspace binding docstrings from Doxygen comments"
    )
else()
    set(MADSPACE_DOCSTRINGS_ARGS
        --stub --out "${MADSPACE_GENERATED_DIR}/python/docstrings.hpp"
    )
    set(MADSPACE_DOCSTRINGS_DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/generate_docstrings.py"
    )
    set(MADSPACE_DOCSTRINGS_COMMENT
        "Writing stub madspace binding docstrings (ENABLE_DOCS=OFF)"
    )
endif()
add_custom_command(
    OUTPUT ${MADSPACE_GENERATED_DIR}/python/docstrings.hpp
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_docstrings.py"
        ${MADSPACE_DOCSTRINGS_ARGS}
    DEPENDS ${MADSPACE_DOCSTRINGS_DEPS}
    COMMENT "${MADSPACE_DOCSTRINGS_COMMENT}"
    VERBATIM
)

python_add_library(
    _madspace_py
    MODULE
    src/python/madspace.cpp
    ${MADSPACE_GENERATED_DIR}/python/instruction_set.hpp
    ${MADSPACE_GENERATED_DIR}/python/docstrings.hpp
    src/python/function_runtime.cpp
    src/python/function_runtime.hpp
    src/python/python_function.cpp
    src/python/python_function.hpp
    WITH_SOABI
)
target_include_directories(_madspace_py PRIVATE ${MADSPACE_GENERATED_DIR}/python)
target_link_libraries(_madspace_py PRIVATE pybind11::headers)
target_link_libraries(_madspace_py PUBLIC madspace)
install(TARGETS _madspace_py DESTINATION madspace)

########################################################################################
# Python type stubs                                                                    #
########################################################################################

# Only built (and installed) with ENABLE_DOCS: pybind11-stubgen has to import
# the freshly built extension, and the stubs carry the docstrings extracted
# above, so they are only meaningful in a documentation build.
if(ENABLE_DOCS)
    set(MADSPACE_PYI_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_pyi)

    # Runs whenever _madspace_py is relinked, i.e. whenever madspace.cpp or,
    # transitively (via instruction_set.hpp), instruction_set.yaml change. Imports
    # the freshly built extension from an isolated staging copy of the madspace
    # package (see generate_pyi.py) so pybind11-stubgen can introspect it.
    add_custom_command(
        TARGET _madspace_py POST_BUILD
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_pyi.py"
            --package-dir "${CMAKE_CURRENT_SOURCE_DIR}/madspace"
            --module "$<TARGET_FILE:_madspace_py>"
            --lib "$<TARGET_FILE:madspace>"
            --stage-dir "${CMAKE_CURRENT_BINARY_DIR}/pyi_stage"
            --output-dir "${MADSPACE_PYI_DIR}"
        COMMENT "Generating madspace/_madspace_py.pyi"
        VERBATIM
    )
    install(FILES "${MADSPACE_PYI_DIR}/madspace/_madspace_py.pyi" DESTINATION madspace)
endif()

########################################################################################
# Source hash                                                                          #
########################################################################################

# Runs at install time (after everything has been compiled), so the embedded
# hash always reflects exactly what was just built rather than what was on
# disk when CMake was configured.
install(CODE "
    execute_process(
        COMMAND \"${Python_EXECUTABLE}\" \"${CMAKE_CURRENT_SOURCE_DIR}/source_hash.py\"
        OUTPUT_VARIABLE MADSPACE_SOURCE_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MADSPACE_SOURCE_HASH_RESULT
    )
    if(NOT \${MADSPACE_SOURCE_HASH_RESULT} EQUAL 0)
        message(FATAL_ERROR \"Failed to compute madspace source hash\")
    endif()
    file(WRITE \"\${CMAKE_INSTALL_PREFIX}/madspace/_source_hash.py\"
        \"SOURCE_HASH = \\\"\${MADSPACE_SOURCE_HASH}\\\"\\n\"
    )
")

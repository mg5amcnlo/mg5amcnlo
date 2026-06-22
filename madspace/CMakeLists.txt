cmake_minimum_required (VERSION 3.15...3.27)
project (
    madspace
    VERSION 0.1
    LANGUAGES CXX
)

option(ENABLE_CUDA "Enable CUDA GPU backend" OFF)
option(ENABLE_HIP "Enable HIP/ROCm GPU backend" OFF)
option(ENABLE_SIMD "Enable SIMD CPU acceleration" OFF)
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
    find_package(rocrand REQUIRED)
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
            -DBUILD_SINGLE=OFF
            -DBUILD_DOUBLE=ON
            -DBUILD_COMPLEX=OFF
            -DBUILD_COMPLEX16=OFF
            -DBUILD_TESTING=OFF
            -DBUILD_WITHOUT_CBLAS=ON
            -DUSE_THREAD=OFF
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
    src/driver/thread_pool.cpp
    src/driver/io.cpp
    src/driver/vegas_optimizer.cpp
    src/driver/discrete_optimizer.cpp
    src/driver/backend.cpp
    src/driver/lhe_output.cpp
    src/driver/madnis_training.cpp
    src/compgraphs/type.cpp
    src/compgraphs/function.cpp
    src/compgraphs/instruction.cpp
    src/compgraphs/optimizer.cpp
    src/compgraphs/instruction_set_mixin.inc
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
    include/madspace/driver/vegas_optimizer.hpp
    include/madspace/driver/discrete_optimizer.hpp
    include/madspace/driver/backend.hpp
    include/madspace/driver/lhe_output.hpp
    include/madspace/driver/logger.hpp
    include/madspace/driver/madnis_training.hpp
    include/madspace/compgraphs.hpp
    include/madspace/compgraphs/type.hpp
    include/madspace/compgraphs/function.hpp
    include/madspace/compgraphs/function_builder_mixin.inc
    include/madspace/compgraphs/opcode_mixin.inc
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
    include/madspace/phasespace/cross_section.hpp
    include/madspace/phasespace/scale.hpp
)

#target_compile_options(madspace PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(madspace PUBLIC include)
target_compile_features(madspace PUBLIC cxx_std_20)
target_link_libraries(madspace PUBLIC nlohmann_json::nlohmann_json)
if (ENABLE_SIMD)
    target_compile_definitions(madspace PRIVATE SIMD_AVAILABLE=1)
endif()

install(TARGETS madspace DESTINATION madspace/lib)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "madspace/include"
    FILES_MATCHING PATTERN "*.hpp*"
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
        ${KERNEL_SRC}
    )
    target_include_directories(madspace_cuda PUBLIC include)
    set_property(TARGET madspace_cuda PROPERTY CUDA_STANDARD 20)
    target_link_libraries(madspace_cuda PRIVATE madspace)
    target_link_libraries(madspace_cuda PRIVATE CUDA::cublas)
    target_link_libraries(madspace_cuda PRIVATE CUDA::curand)
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
        ${KERNEL_SRC}
    )
    target_include_directories(madspace_hip PUBLIC include)
    set_property(TARGET madspace_hip PROPERTY HIP_STANDARD 20)
    set_property(TARGET madspace_hip PROPERTY HIP_EXTENSIONS OFF)
    target_link_libraries(madspace_hip PRIVATE madspace)
    target_link_libraries(madspace_hip PRIVATE hip::device)
    target_link_libraries(madspace_hip PRIVATE roc::rocblas)
    target_link_libraries(madspace_hip PRIVATE roc::rocrand)
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
elseif(APPLE)
    set(CPU_TARGETS madspace_cpu madspace_cpu_neon)
    set(SIMD_FLAGS "" "")
    set(SIMD_DEFINES "" "USE_SIMD_NEON=1")
else()
    set(CPU_TARGETS madspace_cpu madspace_cpu_avx2 madspace_cpu_avx512)
    set(SIMD_FLAGS "" "-mavx2 -mfma" "-mavx512f")
    set(SIMD_DEFINES "" "USE_SIMD_AVX2=1" "USE_SIMD_AVX512=1")
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
        src/cpu/runtime_mixin.inc
        src/cpu/tensor.hpp
        src/cpu/simd.hpp
        src/cpu/simd_arm.hpp
        src/cpu/simd_x86_256.hpp
        ${KERNEL_SRC}
    )
    target_include_directories(${cpu_target} PUBLIC include)
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

python_add_library(
    _madspace_py
    MODULE
    src/python/madspace.cpp
    src/python/instruction_set.hpp
    src/python/function_runtime.cpp
    src/python/function_runtime.hpp
    WITH_SOABI
)
target_link_libraries(_madspace_py PRIVATE pybind11::headers)
target_link_libraries(_madspace_py PUBLIC madspace)
install(TARGETS _madspace_py DESTINATION madspace)

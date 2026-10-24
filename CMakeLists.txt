cmake_minimum_required(VERSION 3.20)

project(PhotonZip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)

option(PHOTONZIP_BUILD_PYTHON "Build the PhotonZip Python extension module" ON)
option(PHOTONZIP_BUILD_HDF5_FILTERS "Build the PhotonZip HDF5 filter plugins" ON)
set(PHOTONZIP_MANS_TARGET_PLATFORM "" CACHE STRING
    "TARGET_PLATFORM for the vendored MANS build (cpu or cpu_nv); empty: cpu_nv when a CUDA compiler is found, else cpu")

# =========================================================
# Vendored MANS (CPU, and NVIDIA GPUs through CUDA)
# =========================================================
if(NOT TARGET mans_common)
  set(_photonzip_mans_platform "${PHOTONZIP_MANS_TARGET_PLATFORM}")
  if(NOT _photonzip_mans_platform)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      set(_photonzip_mans_platform "cpu_nv")
    else()
      set(_photonzip_mans_platform "cpu")
    endif()
  endif()
  if(NOT _photonzip_mans_platform MATCHES "^cpu(_nv)?$")
    message(FATAL_ERROR "PHOTONZIP_MANS_TARGET_PLATFORM must be cpu or cpu_nv "
                        "(the DCU version of PhotonZip is on the dcu branch).")
  endif()
  message(STATUS "PhotonZip: MANS TARGET_PLATFORM=${_photonzip_mans_platform}")

  set(BUILD_HDF5_PLUGIN ${PHOTONZIP_BUILD_HDF5_FILTERS} CACHE BOOL "Build the vendored H5Z-MANS plugin" FORCE)
  set(TARGET_PLATFORM "${_photonzip_mans_platform}" CACHE STRING "Target platform of the vendored MANS build" FORCE)
  add_subdirectory(3rdparty/lossless/MANS)
endif()

# =========================================================
# Core library
# =========================================================
add_library(photonzip_core STATIC
  src/photonzip/core/codec_registry.cpp
  src/photonzip/codecs/mans/mans_codec.cpp
  src/photonzip/codecs/lc/lc_codec.cpp
  src/photonzip/codecs/pfpl/pfpl_codec.cpp
  src/photonzip/preprocess/delta_cpu.cpp
)

target_include_directories(photonzip_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lossless/MANS
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
)

target_link_libraries(photonzip_core PUBLIC mans_common)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(photonzip_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(TARGET mans_nv_core)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(photonzip_core PUBLIC MANS_ENABLE_NV)
  target_include_directories(photonzip_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(photonzip_core PUBLIC CUDA::cudart)
endif()

if(PHOTONZIP_BUILD_HDF5_FILTERS)
  add_subdirectory(filters)
endif()

if(PHOTONZIP_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

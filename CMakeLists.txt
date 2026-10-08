cmake_minimum_required(VERSION 3.20)

project(PhotonZip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)

option(PHOTONZIP_BUILD_PYTHON "Build the PhotonZip Python extension module" ON)
option(PHOTONZIP_BUILD_HDF5_FILTERS "Build the PhotonZip HDF5 filter plugins" ON)
option(PHOTONZIP_ENABLE_DCU "Build the DCU (Hygon DTK / HIP) backend: LC on DCU and MANS" OFF)
set(PHOTONZIP_DCU_HIP_COMPILER "" CACHE FILEPATH "DTK hipcc (or its dcc driver); found on PATH if empty")
set(PHOTONZIP_DCU_ARCH "" CACHE STRING "DCU offload architecture, e.g. gfx906 (Z100) or gfx928/gfx936 (K100)")
set(PHOTONZIP_DCU_HIP_FLAGS "" CACHE STRING "Extra flags for the DTK HIP compiler")

# =========================================================
# DCU backend
# =========================================================
# The HIP language has to be enabled before any target is created; CMake 3.22 otherwise
# leaves HIP_ARCHITECTURES empty on targets that already exist.
if(PHOTONZIP_ENABLE_DCU)
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(FATAL_ERROR "PHOTONZIP_ENABLE_DCU needs CMake >= 3.21 for the HIP language.")
  endif()
  if(PHOTONZIP_DCU_HIP_COMPILER)
    set(_photonzip_hipcc "${PHOTONZIP_DCU_HIP_COMPILER}")
  else()
    find_program(_photonzip_hipcc NAMES hipcc REQUIRED)
  endif()
  # DTK installs hipcc as a symlink to its dcc driver; CMake wants the real Clang driver.
  if(NOT CMAKE_HIP_COMPILER)
    get_filename_component(_photonzip_hip_real "${_photonzip_hipcc}" REALPATH)
    set(CMAKE_HIP_COMPILER "${_photonzip_hip_real}" CACHE FILEPATH "HIP compiler")
  endif()
  if(PHOTONZIP_DCU_ARCH)
    set(CMAKE_HIP_ARCHITECTURES "${PHOTONZIP_DCU_ARCH}" CACHE STRING "HIP offload architectures" FORCE)
  endif()
  if(PHOTONZIP_DCU_HIP_FLAGS)
    set(CMAKE_HIP_FLAGS "${PHOTONZIP_DCU_HIP_FLAGS} ${CMAKE_HIP_FLAGS}")
  endif()
  enable_language(HIP)
  find_package(hip CONFIG QUIET)

  set(PHOTONZIP_MANS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lossless/MANS)

  # HIP sources of both codecs. Each translation unit carries its own device code, so a
  # static library is enough (no relocatable device code needed).
  add_library(photonzip_dcu STATIC
    src/photonzip/core/dcu_runtime.hip
    src/photonzip/codecs/lc/lc_dcu.hip
    ${PHOTONZIP_MANS_DIR}/dcu/mans_dcu.cpp
    ${PHOTONZIP_MANS_DIR}/dcu/adm/mapping_uint16.hip
    ${PHOTONZIP_MANS_DIR}/dcu/ans/dcu_ans.hip
  )
  set_source_files_properties(${PHOTONZIP_MANS_DIR}/dcu/mans_dcu.cpp PROPERTIES LANGUAGE HIP)
  target_include_directories(photonzip_dcu
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${PHOTONZIP_MANS_DIR}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
  )
  target_compile_options(photonzip_dcu PRIVATE $<$<CONFIG:Release>:-O3>)
  target_compile_definitions(photonzip_dcu PUBLIC PHOTONZIP_ENABLE_DCU)
  if(PHOTONZIP_DCU_ARCH)
    set_property(TARGET photonzip_dcu PROPERTY HIP_ARCHITECTURES "${PHOTONZIP_DCU_ARCH}")
  endif()
  # Consumers are linked by the C++ driver, so name the HIP runtime explicitly when the
  # DTK ships its CMake package.
  if(TARGET hip::host)
    target_link_libraries(photonzip_dcu PUBLIC hip::host)
  endif()
endif()

# =========================================================
# Core library
# =========================================================
add_library(photonzip_core STATIC
  src/photonzip/core/codec_registry.cpp
  src/photonzip/codecs/lc/lc_codec.cpp
)

target_include_directories(photonzip_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(photonzip_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(PHOTONZIP_ENABLE_DCU)
  target_sources(photonzip_core PRIVATE src/photonzip/codecs/mans/mans_codec.cpp)
  target_link_libraries(photonzip_core PUBLIC photonzip_dcu)
endif()

if(PHOTONZIP_BUILD_HDF5_FILTERS)
  add_subdirectory(filters)
endif()

if(PHOTONZIP_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

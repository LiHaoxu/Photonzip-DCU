cmake_minimum_required(VERSION 3.20)

project(PhotonZip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)

option(PHOTONZIP_BUILD_PYTHON "Build the PhotonZip Python extension module" ON)
option(PHOTONZIP_BUILD_HDF5_FILTERS "Build the PhotonZip HDF5 filter plugins" ON)

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

if(PHOTONZIP_BUILD_HDF5_FILTERS)
  add_subdirectory(filters)
endif()

if(PHOTONZIP_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

cmake_minimum_required(VERSION 3.20)

# HDF5 filter plugins for the PhotonZip codecs. Each plugin is a standalone shared library
# that HDF5 dlopens from HDF5_PLUGIN_PATH and does not link photonzip_core. The LC CPU
# pipeline is header-only; with PHOTONZIP_ENABLE_DCU the plugins also link the HIP kernels
# in photonzip_dcu (H5Z-LC gains a DCU path, H5Z-MANS is DCU-only).

if(DEFINED HDF5_PREFER_PARALLEL)
  set(HDF5_PREFER_PARALLEL ${HDF5_PREFER_PARALLEL})
endif()
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP QUIET)

set(PHOTONZIP_FILTER_PLUGIN_DIR ${CMAKE_BINARY_DIR}/bin/plugins)

set(PHOTONZIP_FILTER_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/3rdparty
  ${HDF5_INCLUDE_DIRS}
)

function(photonzip_add_filter target source)
  add_library(${target} SHARED ${source})
  target_include_directories(${target} PRIVATE ${PHOTONZIP_FILTER_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${HDF5_LIBRARIES})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE)
    # -Bsymbolic keeps the plugin's own symbols from being interposed by the host process,
    # which would otherwise be able to bind the LC pipeline to another loaded copy.
    #
    # -z nodelete pins the plugin in memory. HDF5 dlcloses its plugins when the library shuts
    # down, which would drop the last reference to libgomp and unmap it while its worker
    # threads are still parked inside it, so the process would segfault on the way out.
    target_link_options(${target} PRIVATE -Wl,-Bsymbolic -Wl,-z,nodelete)
  endif()
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE CXX  # link with the C++ driver even when HIP objects are linked in
    LIBRARY_OUTPUT_DIRECTORY ${PHOTONZIP_FILTER_PLUGIN_DIR}
    OUTPUT_NAME "${target}"
    PREFIX "lib"
  )
endfunction()

photonzip_add_filter(H5Z-LC src/H5Z-LC.cpp)

if(PHOTONZIP_ENABLE_DCU)
  target_link_libraries(H5Z-LC PRIVATE photonzip_dcu)

  photonzip_add_filter(H5Z-MANS src/H5Z-MANS.cpp)
  target_link_libraries(H5Z-MANS PRIVATE photonzip_dcu)
endif()

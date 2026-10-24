cmake_minimum_required(VERSION 3.20)

# HDF5 filter plugins for the PhotonZip codecs. Each plugin is a standalone shared library
# that HDF5 dlopens from HDF5_PLUGIN_PATH and does not link photonzip_core. The LC and PFPL
# CPU pipelines are header-only; with PHOTONZIP_ENABLE_DCU the plugins also link the HIP
# kernels in photonzip_dcu (H5Z-LC gains a DCU path, H5Z-MANS is the DCU MANS plugin). The
# CPU/CUDA H5Z-MANS of the other builds comes from the vendored MANS (tools/H5Z-MANS).

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
    # which would otherwise be able to bind the LC and PFPL pipelines to each other's copies.
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
photonzip_add_filter(H5Z-PFPL src/H5Z-PFPL.cpp)

if(PHOTONZIP_ENABLE_DCU)
  target_link_libraries(H5Z-LC PRIVATE photonzip_dcu)

  photonzip_add_filter(H5Z-MANS src/H5Z-MANS.cpp)
  target_link_libraries(H5Z-MANS PRIVATE photonzip_dcu)
endif()

# =========================================================
# Round-trip tests
# =========================================================
# The tests point HDF5 at the plugin directory themselves, so they can be run straight from
# the build tree without exporting HDF5_PLUGIN_PATH.

function(photonzip_add_filter_test target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE ${PHOTONZIP_FILTER_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${HDF5_LIBRARIES})
  target_compile_definitions(${target} PRIVATE
    PHOTONZIP_H5Z_PLUGIN_DIR="${PHOTONZIP_FILTER_PLUGIN_DIR}")
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/filters)
endfunction()

photonzip_add_filter_test(h5z_lc_roundtrip test/h5z_lc_roundtrip.cpp)
photonzip_add_filter_test(h5z_pfpl_roundtrip test/h5z_pfpl_roundtrip.cpp)

add_dependencies(h5z_lc_roundtrip H5Z-LC)
add_dependencies(h5z_pfpl_roundtrip H5Z-PFPL)

if(BUILD_TESTING)
  add_test(NAME h5z_lc_roundtrip COMMAND h5z_lc_roundtrip)
  add_test(NAME h5z_pfpl_roundtrip COMMAND h5z_pfpl_roundtrip)
endif()

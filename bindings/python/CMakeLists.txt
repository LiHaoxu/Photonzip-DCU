find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native module.cpp)

target_link_libraries(_native PRIVATE photonzip_core)
set_target_properties(_native PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  # In DCU builds the static libraries carry HIP objects, which would make CMake link with
  # the DTK clang driver; that driver cannot read the GCC LTO objects pybind11 produces and
  # silently drops them. The HIP objects link fine with the C++ driver.
  LINKER_LANGUAGE CXX
)

install(TARGETS _native DESTINATION photonzip)

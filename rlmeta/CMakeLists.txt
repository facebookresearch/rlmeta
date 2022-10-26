cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(rlmeta)

set(CMAKE_CXX_STANDARD 17)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-comment -fPIC \
  -march=native -Wfatal-errors -fvisibility=hidden"
)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# PyTorch dependency
execute_process(
  COMMAND
  python -c
  "import os; import torch; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE
  TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
find_package(Torch REQUIRED)
if (MSVC)
  set(
    TORCH_PYTHON_LIBRARIES
    "${TORCH_PATH}/lib/libtorch.dll"
    "${TORCH_PATH}/lib/libtorch_python.dll"
  )
elseif(APPLE)
  set(
    TORCH_PYTHON_LIBRARIES
    "${TORCH_PATH}/lib/libtorch.dylib"
    "${TORCH_PATH}/lib/libtorch_python.dylib"
  )
else()
  set(
    TORCH_PYTHON_LIBRARIES
    "${TORCH_PATH}/lib/libtorch.so"
    "${TORCH_PATH}/lib/libtorch_python.so"
  )
endif()

# pybind11
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11
  ${CMAKE_CURRENT_BINARY_DIR}/third_party/pybind11
)

pybind11_add_module(
  _rlmeta_extension
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/nested_utils/nested_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/groupby.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/ops.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/pybind.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/samplers/prioritized_sampler.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/samplers/sampler.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/samplers/uniform_sampler.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/storage/circular_buffer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/storage/schema.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cc/storage/tensor_circular_buffer.cc
)
target_include_directories(
  _rlmeta_extension PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(_rlmeta_extension PUBLIC torch ${TORCH_PYTHON_LIBRARIES})

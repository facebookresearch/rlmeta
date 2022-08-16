cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(rlmeta)

set(CMAKE_CXX_STANDARD 17)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-comment -fPIC \
  -march=native -Wfatal-errors -fvisibility=hidden"
)

include(FetchContent)

# gRPC dependency
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.47.0
)

set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(grpc)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# proto files
get_filename_component(rpc_proto "${CMAKE_CURRENT_SOURCE_DIR}/protos/rpc.proto" ABSOLUTE)
get_filename_component(rpc_proto_path "${rpc_proto}" PATH)

# generated files
set(rpc_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/rpc.pb.cc")
set(rpc_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/rpc.pb.h")
set(rpc_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/rpc.grpc.pb.cc")
set(rpc_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/rpc.grpc.pb.h")
add_custom_command(
  OUTPUT "${rpc_proto_srcs}" "${rpc_proto_hdrs}" "${rpc_grpc_srcs}" "${rpc_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
       --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
       -I "${rpc_proto_path}"
       --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
       "${rpc_proto}"
  DEPENDS "${rpc_proto}"
)

# rpc_grpc_proto
add_library(
  rpc_grpc_proto
  ${rpc_grpc_srcs}
  ${rpc_grpc_hdrs}
  ${rpc_proto_srcs}
  ${rpc_proto_hdrs}
)
target_include_directories(
  rpc_grpc_proto
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(
  rpc_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)


# pybind11_add_module(
#   _rlmeta_rpc
#   ${CMAKE_CURRENT_SOURCE_DIR}/pybind.cc
#   ${CMAKE_CURRENT_SOURCE_DIR}/server.cc
# )
# target_include_directories(
#   _rlmeta_rpc
#   PUBLIC
#   ${CMAKE_CURRENT_BINARY_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/../..
# )
# target_link_libraries(
#   _rlmeta_rpc
#   PUBLIC
#   grpc++
#   rpc_grpc_proto
# )

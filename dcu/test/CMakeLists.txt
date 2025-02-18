function(add_alphasparse_example TEST_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  # Target compile options
#   target_compile_options(${TEST_TARGET} PRIVATE -Wno-unused-command-line-argument)

  # Include common client headers
  target_include_directories(${TEST_TARGET} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
  # Linker dependencies
  target_link_libraries(${TEST_TARGET} PUBLIC
      # CUDA::cudart
      # CUDA::cudart_static
      # CUDA::cusparse
      # CUDA::cusparse_static
      roc::rocsparse
      hip::host
      hip::device
      roc::rocprim
      roc::rccl
      # hip::amdhip64
      alphasparse
    )
  # Target properties
#   set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endfunction()

add_alphasparse_example(level2/mv_csr_s_dcu_test.cpp)
add_alphasparse_example(level2/mv_csr_d_dcu_test.cpp)
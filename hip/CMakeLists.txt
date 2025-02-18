add_subdirectory(kernel)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_test
    test/args.hip test/check.hip test/io.hip test/test_common.h
    test/check_r.hip
)

set(alphasparse_util
    util/auxiliary.hip util/handle.hip util/malloc.hip util/simplelist.hip util/thread.hip
)

add_library(alphasparse ${alphasparse_headers_public} ${alphasparse_test} ${alphasparse_util} ${alphasparse_source})
target_compile_definitions(alphasparse PUBLIC __HIP_PLATFORM_HCC__)
target_compile_definitions(alphasparse PUBLIC CUDA_ARCH=${CUDA_ARCH})
set_property(TARGET alphasparse PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(alphasparse PUBLIC
    roc::hipsparse
    hip::host
    hip::device
    roc::rocprim
    roc::rccl
    # hip::amdhip64
)
target_include_directories(alphasparse PRIVATE "/opt/rocm-4.5.2/hip/include/hip/" ${HIP_INCLUDE_DIRS} $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

add_subdirectory(test)

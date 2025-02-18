add_subdirectory(kernel)
# add_subdirectory(format)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_test
    test/test_common.cpp test/args.cpp test/check.cpp 
    test/io.cpp test/norm.cpp
)

set(alphasparse_cpu_format
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_create_coo.cpp 
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_coo.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_csr.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_csc.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_dia.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_bsr.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_sky.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_destroy.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_transpose.cpp
    ${PROJECT_SOURCE_DIR}/hygon/util/thread.cpp 
    ${PROJECT_SOURCE_DIR}/hygon/util/partition.cpp 
    ${PROJECT_SOURCE_DIR}/hygon/util/prefix_sum.cpp 
    ${PROJECT_SOURCE_DIR}/hygon/util/bisearch.cpp
    ${PROJECT_SOURCE_DIR}/hygon/util/malloc.cpp
    ${PROJECT_SOURCE_DIR}/hygon/util/timer.cpp
)

set(alphasparse_util
    util/malloc.cpp util/thread.cpp util/timer.cpp util/partition.cpp util/prefix_sum.cpp util/bisearch.cpp
)

add_library(alphasparse ${alphasparse_headers_public} ${alphasparse_test} ${alphasparse_util} ${alphasparse_cpu_format} ${ASM_SOURCES} ${alphasparse_source} )

# target_link_libraries(alphasparse PUBLIC
#     mkl
# )
target_include_directories(alphasparse PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

add_subdirectory(test)

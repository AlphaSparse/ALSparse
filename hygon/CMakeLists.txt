add_subdirectory(kernel)
add_subdirectory(format)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_test
    test/test_common.cpp test/args.cpp test/check.cpp 
    test/io.cpp test/norm.cpp
)

set(alphasparse_util
    util/malloc.cpp util/thread.cpp util/timer.cpp util/partition.cpp util/prefix_sum.cpp util/bisearch.cpp util/pack.cpp util/bitmap.cpp
)

add_library(alphasparse ${alphasparse_headers_public} ${alphasparse_test} ${alphasparse_util} ${alphasparse_format} ${ASM_SOURCES} ${alphasparse_source} )

# target_link_libraries(alphasparse PUBLIC
#     mkl
# )
target_include_directories(alphasparse PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

add_subdirectory(test)

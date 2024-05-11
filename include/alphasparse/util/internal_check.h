#pragma once

/**
 * @brief header for parameter check utils
 */

#include <stdlib.h>
#include "../spdef.h"
#include "../types.h"
#include "../spmat.h"
#include <stdbool.h>

#define check_null_return(pointer, error) \
    if (((void *)pointer) == NULL)        \
    {                                     \
        return error;                     \
    }

#define check_return(expr, error) \
    if (expr)                     \
    {                             \
        return error;             \
    }

// [min max)
#define check_between_return(num, min, max, error) \
    if ((num) >= (max) || (num) < (min))           \
    {                                              \
        return error;                              \
    }

#define check_error_return(func)                  \
    {                                             \
        alphasparseStatus_t _status = func;       \
        if (_status != ALPHA_SPARSE_STATUS_SUCCESS) \
            return _status;                       \
    }

bool check_equal_row_col(const alphasparse_matrix_t A);
bool check_equal_colA_rowB(const alphasparse_matrix_t A, const alphasparse_matrix_t B, const alphasparseOperation_t transA);

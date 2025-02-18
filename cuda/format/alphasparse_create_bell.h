#pragma once

#include "alphasparse.h"

template<typename T>
alphasparseStatus_t
alphasparseCreateBlockedEll(alphasparseSpMatDescr_t* spMatDescr,
                            int64_t rows,
                            int64_t cols,
                            int64_t ellBlockSize,
                            int64_t ellCols,
                            int* ellColInd,
                            T* ellValue,
                            alphasparseIndexType_t ellIdxType,
                            alphasparseIndexBase_t idxBase,
                            alphasparseDataType valueType)
{
  *spMatDescr = nullptr;
  // Allocate
  try {
    *spMatDescr = new _alphasparseSpMatDescr;

    (*spMatDescr)->init = true;

    (*spMatDescr)->rows = rows;
    (*spMatDescr)->cols = cols;

    (*spMatDescr)->ell_cols = ellCols;
    (*spMatDescr)->block_dim = ellBlockSize;

    (*spMatDescr)->col_data = ellColInd;
    (*spMatDescr)->val_data = ellValue;

    (*spMatDescr)->const_col_data = ellColInd;
    (*spMatDescr)->const_val_data = ellValue;

    (*spMatDescr)->row_type = ellIdxType;
    (*spMatDescr)->col_type = ellIdxType;
    (*spMatDescr)->data_type = valueType;

    (*spMatDescr)->idx_base = idxBase;
    (*spMatDescr)->block_dir = ALPHASPARSE_DIRECTION_ROW;
    (*spMatDescr)->format = ALPHA_SPARSE_FORMAT_BLOCKED_ELL;
    (*spMatDescr)->descr = new _alphasparse_mat_descr;
    (*spMatDescr)->descr->base = idxBase;
    // (*spMatDescr)->info = new

    // Initialize descriptor

    (*spMatDescr)->batch_count = 1;
    (*spMatDescr)->batch_stride = 0;
    (*spMatDescr)->offsets_batch_stride = 0;
    (*spMatDescr)->columns_values_batch_stride = 0;
  } catch (...) {
    printf("error!!!!!\n");
    return ALPHA_SPARSE_STATUS_NOT_INITIALIZED;
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

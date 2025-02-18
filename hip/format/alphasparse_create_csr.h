#pragma once

#include "alphasparse.h"


template<typename T>
alphasparseStatus_t
alphasparseCreateCsr(alphasparseSpMatDescr_t* spMatDescr,
                       const int rows,
                       const int cols,
                       const int nnz,
                       int* csrRowOffsets,
                       int* col_data,
                       T* val_data,
                       const alphasparseIndexType_t csrRowOffsetsType,
                       const alphasparseIndexType_t csrColIndType,
                       const alphasparseIndexBase_t idxBase,
                       const alphasparseDataType valueType)
{
    *spMatDescr = nullptr;
    // Allocate
    try
    {
        *spMatDescr = new _alphasparseSpMatDescr;

        (*spMatDescr)->init = true;

        (*spMatDescr)->rows = rows;
        (*spMatDescr)->cols = cols;
        (*spMatDescr)->nnz  = nnz;

        (*spMatDescr)->row_data = csrRowOffsets;
        (*spMatDescr)->col_data = col_data;
        (*spMatDescr)->val_data = val_data;

        (*spMatDescr)->const_row_data = csrRowOffsets;
        (*spMatDescr)->const_col_data = col_data;
        (*spMatDescr)->const_val_data = val_data;

        (*spMatDescr)->row_type  = csrRowOffsetsType;
        (*spMatDescr)->col_type  = csrColIndType;
        (*spMatDescr)->data_type = valueType;

        (*spMatDescr)->idx_base = idxBase;
        (*spMatDescr)->format   = ALPHA_SPARSE_FORMAT_CSR;
        (*spMatDescr)->descr   = new _alphasparse_mat_descr;
        (*spMatDescr)->descr->base = idxBase;
        // (*spMatDescr)->info = new 
        
        // Initialize descriptor

        (*spMatDescr)->batch_count                 = 1;
        (*spMatDescr)->batch_stride                = 0;
        (*spMatDescr)->offsets_batch_stride        = 0;
        (*spMatDescr)->columns_values_batch_stride = 0;
    }
    catch(...)
    {
        printf("error!!!!!\n");
        return ALPHA_SPARSE_STATUS_NOT_INITIALIZED;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

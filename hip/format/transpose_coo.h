#pragma once

#include "alphasparse.h"
#include <memory.h>
#include <stdlib.h>
#include <vector>
#include "alphasparse_create_coo.h"
#include "./coo_order.h"

template<typename T, typename U>//T int, U datatype
alphasparseStatus_t transpose_coo(alphasparseSpMatDescr_t &A) {
    // 计算矩阵的转置
    U* Avalues = (U*)malloc(sizeof(U) * A->nnz);
    T* Arowidx = (T*)malloc(sizeof(T) * A->nnz); 
    T* Acolidx = (T*)malloc(sizeof(T) * A->nnz); 

    hipMemcpy(Avalues, A->val_data, A->nnz * sizeof(U), hipMemcpyDeviceToHost);
    hipMemcpy(Arowidx, A->col_data, A->nnz * sizeof(T), hipMemcpyDeviceToHost);
    hipMemcpy(Acolidx, A->row_data, A->nnz * sizeof(T), hipMemcpyDeviceToHost);
    coo_order<T, U>(A->nnz, Arowidx, Acolidx, Avalues);

    U* dvalues;
    T* drowidx;
    T* dcolidx;

    hipMalloc((void**)&dvalues, A->nnz * sizeof(U));
    hipMalloc((void**)&drowidx, A->nnz * sizeof(T));
    hipMalloc((void**)&dcolidx, A->nnz * sizeof(T));
    hipMemcpy(dvalues, Avalues, A->nnz * sizeof(U), hipMemcpyHostToDevice);
    hipMemcpy(drowidx, Arowidx, A->nnz * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(dcolidx, Acolidx, A->nnz * sizeof(T), hipMemcpyHostToDevice);

    alphasparseCreateCoo(&A,
                      A->cols,
                      A->rows,
                      A->nnz,
                      drowidx,
                      dcolidx,
                      dvalues,
                      A->row_type,
                      A->idx_base,
                      A->data_type);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
inline T conj_val(T val)
{
    return val;
}

template<>
inline hipFloatComplex conj_val(hipFloatComplex val)
{
    return hipConjf(val);
}

template<>
inline hipDoubleComplex conj_val(hipDoubleComplex val)
{
    return hipConj(val);
}


template<typename T, typename U>//T int, U datatype
alphasparseStatus_t transpose_conj_coo(alphasparseSpMatDescr_t &A) {
    // 计算矩阵的转置
    U* Avalues = (U*)malloc(sizeof(U) * A->nnz);
    T* Arowidx = (T*)malloc(sizeof(T) * A->nnz); 
    T* Acolidx = (T*)malloc(sizeof(T) * A->nnz); 

    hipMemcpy(Avalues, A->val_data, A->nnz * sizeof(U), hipMemcpyDeviceToHost);
    hipMemcpy(Arowidx, A->col_data, A->nnz * sizeof(T), hipMemcpyDeviceToHost);
    hipMemcpy(Acolidx, A->row_data, A->nnz * sizeof(T), hipMemcpyDeviceToHost);

    coo_order<T, U>(A->nnz, Arowidx, Acolidx, Avalues);

    for(int i = 0; i < A->nnz; i ++)
        Avalues[i] = conj_val<U>(Avalues[i]);

    U* dvalues;
    T* drowidx;
    T* dcolidx;

    hipMalloc((void**)&dvalues, A->nnz * sizeof(U));
    hipMalloc((void**)&drowidx, A->nnz * sizeof(T));
    hipMalloc((void**)&dcolidx, A->nnz * sizeof(T));
    hipMemcpy(dvalues, Avalues, A->nnz * sizeof(U), hipMemcpyHostToDevice);
    hipMemcpy(drowidx, Arowidx, A->nnz * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(dcolidx, Acolidx, A->nnz * sizeof(T), hipMemcpyHostToDevice);

    alphasparseCreateCoo(&A,
                      A->cols,
                      A->rows,
                      A->nnz,
                      drowidx,
                      dcolidx,
                      dvalues,
                      A->row_type,
                      A->idx_base,
                      A->data_type);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
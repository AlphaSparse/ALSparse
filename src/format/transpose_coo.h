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

    cudaMemcpy(Avalues, A->val_data, A->nnz * sizeof(U), cudaMemcpyDeviceToHost);
    cudaMemcpy(Arowidx, A->col_data, A->nnz * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(Acolidx, A->row_data, A->nnz * sizeof(T), cudaMemcpyDeviceToHost);
    coo_order<T, U>(A->nnz, Arowidx, Acolidx, Avalues);

    U* dvalues;
    T* drowidx;
    T* dcolidx;

    cudaMalloc((void**)&dvalues, A->nnz * sizeof(U));
    cudaMalloc((void**)&drowidx, A->nnz * sizeof(T));
    cudaMalloc((void**)&dcolidx, A->nnz * sizeof(T));
    cudaMemcpy(dvalues, Avalues, A->nnz * sizeof(U), cudaMemcpyHostToDevice);
    cudaMemcpy(drowidx, Arowidx, A->nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dcolidx, Acolidx, A->nnz * sizeof(T), cudaMemcpyHostToDevice);

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
inline cuFloatComplex conj_val(cuFloatComplex val)
{
    return cuConjf(val);
}

template<>
inline cuDoubleComplex conj_val(cuDoubleComplex val)
{
    return cuConj(val);
}


template<typename T, typename U>//T int, U datatype
alphasparseStatus_t transpose_conj_coo(alphasparseSpMatDescr_t &A) {
    // 计算矩阵的转置
    U* Avalues = (U*)malloc(sizeof(U) * A->nnz);
    T* Arowidx = (T*)malloc(sizeof(T) * A->nnz); 
    T* Acolidx = (T*)malloc(sizeof(T) * A->nnz); 

    cudaMemcpy(Avalues, A->val_data, A->nnz * sizeof(U), cudaMemcpyDeviceToHost);
    cudaMemcpy(Arowidx, A->col_data, A->nnz * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(Acolidx, A->row_data, A->nnz * sizeof(T), cudaMemcpyDeviceToHost);

    coo_order<T, U>(A->nnz, Arowidx, Acolidx, Avalues);

    for(int i = 0; i < A->nnz; i ++)
        Avalues[i] = conj_val<U>(Avalues[i]);

    U* dvalues;
    T* drowidx;
    T* dcolidx;

    cudaMalloc((void**)&dvalues, A->nnz * sizeof(U));
    cudaMalloc((void**)&drowidx, A->nnz * sizeof(T));
    cudaMalloc((void**)&dcolidx, A->nnz * sizeof(T));
    cudaMemcpy(dvalues, Avalues, A->nnz * sizeof(U), cudaMemcpyHostToDevice);
    cudaMemcpy(drowidx, Arowidx, A->nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dcolidx, Acolidx, A->nnz * sizeof(T), cudaMemcpyHostToDevice);

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
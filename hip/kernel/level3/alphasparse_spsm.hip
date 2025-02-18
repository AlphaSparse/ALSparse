#include "../../format/transpose_csr.h"
#include "../../format/coo2csr.h"
#include "alphasparse.h"
#include "spsm_csr_n_lo_roc.h"

#include <iostream>

template<typename T, typename U, typename O>
alphasparseStatus_t
spsm_template(
    alphasparseHandle_t handle,
    alphasparseOperation_t opA,
    alphasparseOperation_t opB,
    const void *alpha,
    alphasparseSpMatDescr_t matA,
    alphasparseDnMatDescr_t matB,
    alphasparseDnMatDescr_t matC,
    alphasparseSpSMAlg_t alg
) {
    // if (matA->format == ALPHA_SPARSE_FORMAT_COO) {
    //     T m = matA->rows;
    //     T* dCsrRowPtr = NULL;
    //     cudaMalloc((void**)&dCsrRowPtr, sizeof(T) * (m + 1));
    //     alphasparseXcoo2csr(matA->row_data, matA->nnz, m, dCsrRowPtr);
    //     matA->row_data = dCsrRowPtr;
    //     matA->const_row_data = dCsrRowPtr;
    //     matA->format = ALPHA_SPARSE_FORMAT_CSR;
    // }
    if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        transpose_csr<T, U>(matA);
    }
    // if (matB->order == ALPHASPARSE_ORDER_ROW) {
    //     U* dmatB_value = NULL;
    //     cudaMalloc((void**)&dmatB_value, sizeof(U) * matB->rows * matB->ld);
    //     transpose_dn_mat<T, U>(handle, matB->rows, matB->ld, (U*)matB->values, dmatB_value);
    //     cudaMemcpy(matB->values, dmatB_value, matB->rows * matB->ld * sizeof(U), cudaMemcpyDeviceToDevice);
    //     matB->order = ALPHASPARSE_ORDER_COL;
    //     int64_t temp = matB->cols;
    //     matB->cols = matB->rows;
    //     matB->rows = temp;
    // }
    if (matA->descr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT &&
        matA->descr->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER) {
        switch (alg) {
            case ALPHASPARSE_SPSM_CSR_ALG_ROC: {
                spsm_csr_n_lo_roc_solve<T, U>(
                    handle,
                    (T)matA->rows,
                    (T)matC->cols,
                    (T)matA->nnz,
                    *((U*)alpha),
                    (U*)matA->val_data,
                    (T*)matA->row_data,
                    (T*)matA->col_data,
                    (U*)matB->values,
                    (T)matB->ld,
                    (U*)matC->values
                );
                break;
            }
            case ALPHASPARSE_SPSM_CSR_ALG_CW: {
                // spsm_csr_n_lo_cw<T, U>(
                //     handle,
                //     (T)matA->rows,
                //     (T)matC->cols,
                //     (T)matA->nnz,
                //     *((U*)alpha),
                //     (U*)matA->val_data,
                //     (T*)matA->row_data,
                //     (T*)matA->col_data,
                //     (U*)matB->values,
                //     (O)matB->order,
                //     (T)matB->ld,
                //     (U*)matC->values,
                //     (O)matC->order,
                //     (T)matC->ld
                // );
                break;
            }
            default: {
                break;
            }
        }
    } else if (matA->descr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT &&
        matA->descr->fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER) {
        switch (alg) {
            case ALPHASPARSE_SPSM_ALG_DEFAULT: {
                // spsm_csr_n_hi<T, U>(
                //     handle,
                //     (T)matA->rows,
                //     (T)matC->cols,
                //     (T)matA->nnz,
                //     *((U*)alpha),
                //     (U*)matA->val_data,
                //     (T*)matA->row_data,
                //     (T*)matA->col_data,
                //     (U*)matB->values,
                //     (T)matB->ld,
                //     (U*)matC->values
                // );
                break;
            }
            case ALPHASPARSE_SPSM_CSR_ALG_CW: {
                // spsm_csr_n_up_cw<T, U>(
                //     handle,
                //     (T)matA->rows,
                //     (T)matC->cols,
                //     (T)matA->nnz,
                //     *((U*)alpha),
                //     (U*)matA->val_data,
                //     (T*)matA->row_data,
                //     (T*)matA->col_data,
                //     (U*)matB->values,
                //     (O)matB->order,
                //     (T)matB->ld,
                //     (U*)matC->values,
                //     (O)matC->order,
                //     (T)matC->ld
                // );
                break;
            }
            default: {
                break;
            }
        }
    } else if (matA->descr->diag_type == ALPHA_SPARSE_DIAG_UNIT &&
        matA->descr->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER) {
        // spsm_csr_u_lo_roc<T, U>(
        //     handle,
        //     (T)matA->rows,
        //     (T)matC->cols,
        //     (T)matA->nnz,
        //     *((U*)alpha),
        //     (U*)matA->val_data,
        //     (T*)matA->row_data,
        //     (T*)matA->col_data,
        //     (U*)matB->values,
        //     (T)matB->ld,
        //     (U*)matC->values
        // );
    } else if (matA->descr->diag_type == ALPHA_SPARSE_DIAG_UNIT &&
        matA->descr->fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER) {
        // spsm_csr_u_hi<T, U>(
        //     handle,
        //     (T)matA->rows,
        //     (T)matC->cols,
        //     (T)matA->nnz,
        //     *((U*)alpha),
        //     (U*)matA->val_data,
        //     (T*)matA->row_data,
        //     (T*)matA->col_data,
        //     (U*)matB->values,
        //     (T)matB->ld,
        //     (U*)matC->values
        // );
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpSM_solve(
    alphasparseHandle_t handle,
    alphasparseOperation_t opA,
    alphasparseOperation_t opB,
    const void* alpha,
    alphasparseSpMatDescr_t matA,
    alphasparseDnMatDescr_t matB,
    alphasparseDnMatDescr_t matC,
    alphasparseDataType computeType,
    alphasparseSpSMAlg_t alg,
    alphasparseSpSMDescr_t spsmDescr
) {
    // single real ; i32
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
        matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F) {
        return spsm_template<int32_t, float, alphasparseOrder_t>(handle, opA, opB, alpha, matA, matB, matC, alg);
    }
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
        matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F) {
        return spsm_template<int32_t, double, alphasparseOrder_t>(handle, opA, opB, alpha, matA, matB, matC, alg);
    }
    // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
    //     matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
    //     return spsm_template<int32_t, hipFloatComplex, alphasparseOrder_t>(handle, opA, opB, alpha, matA, matB, matC, alg);
    // }
    // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
    //     matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
    //     return spsm_template<int32_t, hipDoubleComplex, alphasparseOrder_t>(handle, opA, opB, alpha, matA, matB, matC, alg);
    // }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpSM_bufferSize(
    alphasparseHandle_t handle,
    alphasparseOperation_t opA,
    alphasparseOperation_t opB,
    const void* alpha,
    alphasparseSpMatDescr_t matA,
    alphasparseDnMatDescr_t matB,
    alphasparseDnMatDescr_t matC,
    alphasparseDataType computeType,
    alphasparseSpSMAlg_t alg,
    alphasparseSpSMDescr_t spsmDescr,
    size_t* bufferSize
) {
    *bufferSize = 4;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


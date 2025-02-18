#pragma once

#include <map>
#include <string>
#include "alphasparse.h"

std::map<cusparseSpSVAlg_t, std::string> cuda_spsv_alg_map {
    {CUSPARSE_SPSV_ALG_DEFAULT, "CUSPARSE_SPSV_ALG_DEFAULT"},
};

std::map<alphasparseSpSVAlg_t, std::string> alpha_spsv_alg_map {
    {ALPHA_SPARSE_SPSV_ALG_DEFAULT, "ALPHA_SPARSE_SPSV_ALG_DEFAULT"},
    {ALPHA_SPARSE_SPSV_ALG_CSR_CW, "ALPHA_SPARSE_SPSV_ALG_CSR_CW"},
};

std::map<cusparseOperation_t, std::string> cuda_op_map {
    {CUSPARSE_OPERATION_NON_TRANSPOSE, "CUSPARSE_OPERATION_NON_TRANSPOSE"}, 
    {CUSPARSE_OPERATION_TRANSPOSE, "CUSPARSE_OPERATION_TRANSPOSE"},
    {CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE"}
};
std::map<alphasparseOperation_t, std::string> alpha_op_map {
    {ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, "ALPHA_SPARSE_OPERATION_NON_TRANSPOSE"},
    {ALPHA_SPARSE_OPERATION_TRANSPOSE, "ALPHA_SPARSE_OPERATION_TRANSPOSE"},
    {ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE, "ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE"}
};
std::map<alphasparseOperation_t, cusparseOperation_t> alpha2cuda_op_map {
    {ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE}, 
    {ALPHA_SPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE},
    {ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
};
std::map<cusparseFillMode_t, std::string> cuda_fill_map {
    {CUSPARSE_FILL_MODE_UPPER, "CUSPARSE_FILL_MODE_UPPER"},
    {CUSPARSE_FILL_MODE_LOWER, "CUSPARSE_FILL_MODE_LOWER"}
};
std::map<alphasparse_fill_mode_t, std::string> alpha_fill_map {
    {ALPHA_SPARSE_FILL_MODE_UPPER, "ALPHA_SPARSE_FILL_MODE_UPPER"},
    {ALPHA_SPARSE_FILL_MODE_LOWER, "ALPHA_SPARSE_FILL_MODE_LOWER"}
};
std::map<alphasparse_fill_mode_t, cusparseFillMode_t> alpha2cuda_fill_map {
    {ALPHA_SPARSE_FILL_MODE_UPPER, CUSPARSE_FILL_MODE_UPPER},
    {ALPHA_SPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_LOWER}
};
std::map<cusparseDiagType_t, std::string> cuda_diag_map {
    {CUSPARSE_DIAG_TYPE_NON_UNIT, "CUSPARSE_DIAG_TYPE_NON_UNIT"},
    {CUSPARSE_DIAG_TYPE_UNIT, "CUSPARSE_DIAG_TYPE_UNIT"}
};
std::map<alphasparse_diag_type_t, std::string> alpha_diag_map {
    {ALPHA_SPARSE_DIAG_NON_UNIT, "ALPHA_SPARSE_DIAG_NON_UNIT"},
    {ALPHA_SPARSE_DIAG_UNIT, "ALPHA_SPARSE_DIAG_UNIT"}
};
std::map<alphasparse_diag_type_t, cusparseDiagType_t> alpha2cuda_diag_map {
    {ALPHA_SPARSE_DIAG_NON_UNIT, CUSPARSE_DIAG_TYPE_NON_UNIT},
    {ALPHA_SPARSE_DIAG_UNIT, CUSPARSE_DIAG_TYPE_UNIT}
};
std::map<cusparseOrder_t, std::string> cuda_order_map {
    {CUSPARSE_ORDER_ROW, "CUSPARSE_ORDER_ROW"},
    {CUSPARSE_ORDER_COL, "CUSPARSE_ORDER_COL"},
};
std::map<alphasparseOrder_t, std::string> alpha_order_map {
    {ALPHASPARSE_ORDER_ROW, "ALPHASPARSE_ORDER_ROW"},
    {ALPHASPARSE_ORDER_COL, "ALPHASPARSE_ORDER_COL"},
};
std::map<alphasparseOrder_t, cusparseOrder_t> alpha2cuda_order_map {
    {ALPHASPARSE_ORDER_ROW, CUSPARSE_ORDER_ROW},
    {ALPHASPARSE_ORDER_COL, CUSPARSE_ORDER_COL},
};
std::map<cudaDataType, std::string> cuda_datatype_map {
    {CUDA_R_32F, "CUDA_R_32F"},
    {CUDA_R_64F, "CUDA_R_64F"},
    {CUDA_C_32F, "CUDA_C_32F"},
    {CUDA_C_64F, "CUDA_C_64F"}
};
std::map<alphasparseDataType, std::string> alpha_datatype_map {
    {ALPHA_R_32F, "ALPHA_R_32F"},
    {ALPHA_R_64F, "ALPHA_R_64F"},
    {ALPHA_C_32F, "ALPHA_C_32F"},
    {ALPHA_C_64F, "ALPHA_C_64F"}
};
std::map<alphasparseDataType, cudaDataType> alpha2cuda_datatype_map {
    {ALPHA_R_32F, CUDA_R_32F},
    {ALPHA_R_64F, CUDA_R_64F},
    {ALPHA_C_32F, CUDA_C_32F},
    {ALPHA_C_64F, CUDA_C_64F}
};


template<typename U>
alphasparseDataType get_alpha_datatype() {
  return 0;
}

template<>
alphasparseDataType get_alpha_datatype<double>() {
  return ALPHA_R_64F;
}

template<>
alphasparseDataType get_alpha_datatype<float>() {
  return ALPHA_R_32F;
}

template<>
alphasparseDataType get_alpha_datatype<cuDoubleComplex>() {
  return ALPHA_C_64F;
}

template<>
alphasparseDataType get_alpha_datatype<cuFloatComplex>() {
  return ALPHA_C_32F;
}

alphasparseSpSVAlg_t get_alpha_spsv_alg(int alg_num) {
  alphasparseSpSVAlg_t alpha_alg = ALPHA_SPARSE_SPSV_ALG_DEFAULT;
  switch (alg_num) {
    case 0: {
      alpha_alg = ALPHA_SPARSE_SPSV_ALG_DEFAULT;
      break;
    }
    case 2: {
      alpha_alg = ALPHA_SPARSE_SPSV_ALG_CSR_CW;
      break;
    }
    default: {
      break;
    }
  }
  return alpha_alg;
}

const char* get_filename(const char *file) {
  const char* lastSlash = strrchr(file, '/');
  if (lastSlash != NULL) {
    return lastSlash + 1;
  } else {
    return file;
  }
  return NULL;
}

/*
    对角线有元素的方阵的左下角每行非零元个数统计
*/
template<typename T>
void get_coo_row_nnz_cnt(
    const T *coo_row_idx, 
    const T *coo_col_idx, 
    const T m,
    const T nnz,
    T *row_nnz_cnt
) {
    // printf("m: %d\n", m);
    T ptr = 0;
    for (int row = 0; row < m; row++) {
        T cnt = 0;
        while (ptr < nnz && row == coo_row_idx[ptr] && coo_row_idx[ptr] >= coo_col_idx[ptr]) {
            printf("%d,%d\n", row, coo_col_idx[ptr]);
            cnt++;
            ptr++;
        }
        // printf("%d,%d\n", row, cnt);
        row_nnz_cnt[row] = cnt;
        while (ptr < nnz && row == coo_row_idx[ptr]) {
            ptr++;
        }
    }
    return;
}

/*
    对角线有元素的方阵的左下角每行非零元个数统计
*/
template<typename T>
void get_csr_row_nnz_cnt(
    const T *csr_row_ptr, 
    const T *csr_col_idx, 
    const T m,
    const T nnz,
    T *row_nnz_cnt
) {
    for (int row = 0; row < m; row++) {
      int cnt = 0;
      for (int ptr = csr_row_ptr[row]; ptr < csr_row_ptr[row + 1] && csr_col_idx[ptr] <= row; ptr++) {
        cnt++;
      }
    //   printf("row: %d, cnt: %d\n", row, cnt);
      row_nnz_cnt[row] = cnt;
    }
    return;
}

// [row_start, row_end)
template<typename T>
T 
get_elem_cnt_all(
    const T row_start,
    const T row_end,
    const T *row_nnz_cnt
) {
    T cnt = 0;
    for (T row = row_start; row < row_end; row++) {
        cnt += row_nnz_cnt[row];
    }
    return cnt;
}


template<typename T>
void warp_divide(
    const T *row_nnz_cnt, 
    const T m, 
    const T border,
    T *len, 
    T *warp_num
) {
    const T WARP_SIZE = 32;
    warp_num[0] = 0;
    T row_end;
    T elem_cnt_all = 0;
    T k = 1;
    double elem_cnt_avg = 0;
    for (T row_start = 0; row_start < m; row_start += WARP_SIZE) {
        row_end = row_start + WARP_SIZE;          // [row_start, row_end)
        row_end = (row_end > m) ? m : row_end;
        elem_cnt_all = get_elem_cnt_all(row_start, row_end, row_nnz_cnt);
        elem_cnt_avg = (double)elem_cnt_all / (row_end - row_start);
        if (elem_cnt_avg >= border) {  // warp-level
        for (T row_cur = row_start + 1; row_cur <= row_end; row_cur++) {
            warp_num[k] = row_cur;
            k++;
        }
        } else {
            warp_num[k] = row_end;
            k++;
        }
    }
    *len = k;
    return;
}




// csr format
template<typename U>
void print_cusparse_matA(
    cusparseSpMatDescr_t matA
) {
    int64_t m, n;
    int64_t nnz;
    int64_t *csrRowOffsets;
    int64_t *csrColInd;
    U *csrValues;
    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueType;
    cusparseCsrGet(matA,
                &m,
                &n,
                &nnz,
                (void **)&csrRowOffsets,
                (void **)&csrColInd,
                (void **)&csrValues,
                &csrRowOffsetsType,
                &csrColIndType,
                &idxBase,
                &valueType);
    int32_t *csr_row_ptr = (int32_t *)malloc(sizeof(int32_t) * (m + 1));
    int32_t *csr_col_idx = (int32_t *)malloc(sizeof(int32_t) * nnz);
    U *csr_val = (U *)malloc(sizeof(U) * nnz);
    cudaMemcpy(csr_row_ptr, csrRowOffsets, sizeof(int32_t) * (m + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(csr_col_idx, csrColInd, sizeof(int32_t) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(csr_val, csrValues, sizeof(U) * nnz, cudaMemcpyDeviceToHost);
    std::cout << "m: " << m << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "csr_row_ptr" << std::endl;
    for (int i = 0; i < m + 1; i++) {
        std::cout << csr_row_ptr[i] << " ";
    }
    std::cout << "\ncsr_col_idx\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_col_idx[i] << " ";
    }
    std::cout << "\ncsr_val\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_val[i] << "; ";
    }
    std::cout << std::endl;
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
}

// csr format
template<typename U>
void print_alphasparse_matA(
    alphasparseSpMatDescr_t matA
) {
    int m = matA->rows;
    int nnz = matA->nnz;
    int *csr_row_ptr = (int *)malloc(sizeof(int) * (m + 1));
    int *csr_col_idx = (int *)malloc(sizeof(int) * nnz);
    U *csr_val = (U *)malloc(sizeof(U) * nnz);
    cudaMemcpy(csr_row_ptr, matA->row_data, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(csr_col_idx, matA->col_data, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(csr_val, matA->val_data, sizeof(U) * nnz, cudaMemcpyDeviceToHost);
    std::cout << "m: " << m << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "csr_row_ptr" << std::endl;
    for (int i = 0; i < m + 1; i++) {
        std::cout << csr_row_ptr[i] << " ";
    }
    std::cout << "\ncsr_col_idx\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_col_idx[i] << " ";
    }
    std::cout << "\ncsr_val\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_val[i] << "; ";
    }
    std::cout << std::endl;
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
}



// CPU端 单线程计算COO SPSV
template <typename T, typename U>
void 
spsv_coo_cpu(
    const T* coo_row_idx,
    const T* coo_col_idx,
    const U* coo_val,
    const T m,
    const T n,
    const T nnz,
    const U alpha,
    const U* x,
    U* y,
    alphasparse_fill_mode_t fillmode,
    alphasparse_diag_type_t diagtype
) {
    T row = 0;
    T col = 0;
    U val = U{};
    U tmp_sum = U{};
    if (fillmode == ALPHA_SPARSE_FILL_MODE_LOWER) {
      for (T i = 0; i < nnz; ) {
          row = coo_row_idx[i];
          tmp_sum = U{};
          T j = i;
          while (coo_row_idx[j] == row) {
              col = coo_col_idx[j];
              val = coo_val[j];
              if (col < row) {
                  tmp_sum += val * y[col];
              } else if (col == row) {
                  y[row] = (alpha * x[row] - tmp_sum) / val;
              }
              j++;
          }
          i = j;
      }
    } else if (fillmode == ALPHA_SPARSE_FILL_MODE_LOWER) {
      for (T i = nnz - 1; i >= 0; ) {
          row = coo_row_idx[i];
          tmp_sum = U{};
          T j = i;
          while (coo_row_idx[j] == row) {
              col = coo_col_idx[j];
              val = coo_val[j];
              if (col > row) {
                  tmp_sum += val * y[col];
              } else if (col == row) {
                  y[row] = (alpha * x[row] - tmp_sum) / val;
              }
              j--;
          }
          i = j;
      }
    }
    
    return;
}

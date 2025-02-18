#pragma once


/*
    description:
        coo格式的矩阵A作为输入，在A的基础上进行修改，将A裁剪成正方形，若对角线无元素则在该对角线位置补1
*/
template<typename T>
void mat_patch_trim_s(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    float **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    float *tmp_val = (float *)malloc(sizeof(float) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.f;
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.f;
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    float *new_val = (float *)malloc(sizeof(float) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(float) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}


template<typename T>
void mat_patch_trim_d(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    double **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    double *tmp_val = (double *)malloc(sizeof(double) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.0;
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.0;
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    double *new_val = (double *)malloc(sizeof(double) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(double) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}

template<typename T>
void mat_patch_trim_c(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    cuFloatComplex **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    cuFloatComplex *tmp_val = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_cuFloatComplex(1.f, 1.f);
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_cuFloatComplex(1.f, 1.f);
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    cuFloatComplex *new_val = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(cuFloatComplex) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}

template<typename T>
void mat_patch_trim_z(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    cuDoubleComplex **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    cuDoubleComplex *tmp_val = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_cuDoubleComplex(1.f, 1.f);
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_cuDoubleComplex(1.f, 1.f);
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    cuDoubleComplex *new_val = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(cuDoubleComplex) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}

/*
    description:
        coo格式的矩阵A每行元素求和归一化
        a_{ii} = \sum_{j = 0}^{n - 1} {a_{ij}} + 1, for all i in [0, m)
        a_{ij} /= a_{ii}, for all {i, j} in [{0, 0}, {m, n})
        complex类型的数据后续再考虑
*/
template<typename T, typename U>
void mat_adjust_nnz(
    const int *coo_row_index,
    const int *coo_col_index,
    U *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    //using namespace std;
    // if (std::is_same<U, cuFloatComplex>::value || std::is_same<U, cuDoubleComplex>::value) {
    //     return;
    // }
    // if (fillmode == CUSPARSE_FILL_MODE_LOWER) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            U partial_sum = {};
            while(row_id == coo_row_index[j] && j < nnz) {
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            coo_values[j - 1] = partial_sum + 1;
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
            while (row_id == coo_row_index[j] && j < nnz) {
                j++;
            }
        }
    // } else if (fillmode == CUSPARSE_FILL_MODE_UPPER) {

    // }
}



void mat_adjust_nnz_s(
    const int *coo_row_index,
    const int *coo_col_index,
    float *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum = 0.f;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                    diag_ptr = j;
                }
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            coo_values[diag_ptr] = partial_sum + 1;
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum = 0.f;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    }
}


void mat_adjust_nnz_d(
    const int *coo_row_index,
    const int *coo_col_index,
    double *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum = 0.0;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                  diag_ptr = j;
                }
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            coo_values[diag_ptr] = partial_sum + 1;
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum = 0.f;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    }
}


void mat_adjust_nnz_c(
    const int *coo_row_index,
    const int *coo_col_index,
    cuFloatComplex *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum_real = 0.f;
            float partial_sum_imag = 0.f;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                  diag_ptr = j;
                }
                partial_sum_real += fabs(cuCrealf(coo_values[j]));
                partial_sum_imag += fabs(cuCimagf(coo_values[j]));
                j++;
            }
            coo_values[diag_ptr] = make_cuFloatComplex(partial_sum_real + 1, partial_sum_imag + 1);
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_cuFloatComplex(cuCrealf(coo_values[k])/partial_sum_real, cuCimagf(coo_values[k])/partial_sum_imag);
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum_real = 0.f;
            float partial_sum_imag = 0.f;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum_real += fabs(cuCrealf(coo_values[j]));
                partial_sum_imag += fabs(cuCimagf(coo_values[j]));
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_cuFloatComplex(cuCrealf(coo_values[k])/partial_sum_real, cuCimagf(coo_values[k])/partial_sum_imag);
            }
        }
    }
}

void mat_adjust_nnz_z(
    const int *coo_row_index,
    const int *coo_col_index,
    cuDoubleComplex *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum_real = 0.0;
            double partial_sum_imag = 0.0;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                  diag_ptr = j;
                }
                partial_sum_real += fabs(cuCreal(coo_values[j]));
                partial_sum_imag += fabs(cuCimag(coo_values[j]));
                j++;
            }
            coo_values[diag_ptr] = make_cuDoubleComplex(partial_sum_real + 1, partial_sum_imag + 1);
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_cuDoubleComplex(cuCreal(coo_values[k])/partial_sum_real, cuCimag(coo_values[k])/partial_sum_imag);
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum_real = 0.0;
            double partial_sum_imag = 0.0;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum_real += fabs(cuCreal(coo_values[j]));
                partial_sum_imag += fabs(cuCimag(coo_values[j]));
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_cuDoubleComplex(cuCreal(coo_values[k])/partial_sum_real, cuCimag(coo_values[k])/partial_sum_imag);
            }
        }
    }
}


/*
    description:
        检查coo格式的矩阵A是否存在没有非零元的行
*/
template <typename T>
bool has_coo_zero_row(
        const T nnz,
        const T *coo_row_idx
) {
  for (T i = 1; i < nnz; i++) {
    if (coo_row_idx[i] - coo_row_idx[i - 1] > 1) {
      return true;
    }
  }
  return false;
}


/*
    description:
        检查coo格式的矩阵A是否存在有对角线元素没有非零元的情况
*/
template <typename T>
bool has_coo_zero_diag(
        const T m,
        const T nnz,
        const T *coo_row_idx,
        const T *coo_col_idx
) {
  int i = 0;
  int row;
  for (row = 0; row < m; row++) {
    while (i < nnz && coo_row_idx[i] == row && coo_row_idx[i] > coo_col_idx[i]) {
      i++;
    }
    if (i >= nnz || coo_row_idx[i] != row || coo_row_idx[i] != coo_col_idx[i]) {
      printf("i: %d\n", i);
      if (i < nnz) {
        printf("row: %d, col: %d\n", coo_row_idx[i], coo_col_idx[i]);
      }
      return true;
    }
    while (i < nnz && coo_row_idx[i] == row) {
      i++;
    }
  }
  if (!(row == m)) {
    printf("row: %d, m: %d\n", row, m);
  }
  return !(row == m);
}

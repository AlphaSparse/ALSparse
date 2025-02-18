// #include "dcu_gemv_scalar.h"
#include "dcu_gemv_vector.h"
// #include "dcu_gemv_row_partition.h"
// #include "dcu_gemv_adaptive.h"
// #include "dcu_gemv_merge.h"
// #include "dcu_gemv_xxx.h"

// #include "alphasparse/tuning_dcu/csr_tuning_dcu.h"

//min_nnz_row max_nnz_row rows cols nnz var_nnz_row avr_nnz_row sparsity
static void get_feature(alphasparseHandle_t handle,
                        const ALPHA_INT *csr_row_ptr,
                        ALPHA_INT m,
                        ALPHA_INT n,
                        ALPHA_INT nnz_num,
                        double *feature)
{
    // Temporary arrays to hold device data
    std::vector<ALPHA_INT> hptr(m + 1);
    hipMemcpyAsync(hptr.data(), csr_row_ptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyDeviceToHost, handle->stream);
    hipStreamSynchronize(handle->stream);

    ALPHA_INT min_nnz_row = INT32_MAX;
    ALPHA_INT max_nnz_row = 0;
    ALPHA_INT rows        = m;
    ALPHA_INT cols        = n;
    ALPHA_INT nnz         = nnz_num;
    double var_nnz_row    = 0.;
    double avr_nnz_row    = (double)nnz / rows;
    double sparsity       = (double)nnz / (rows * cols);

    for (ALPHA_INT i = 0; i < m; i++) {
        ALPHA_INT row_nnz = hptr[i + 1] - hptr[i];
        min_nnz_row       = row_nnz < min_nnz_row ? row_nnz : min_nnz_row;
        max_nnz_row       = row_nnz > max_nnz_row ? row_nnz : max_nnz_row;
        var_nnz_row += 1.0f * (row_nnz - avr_nnz_row) * (row_nnz - avr_nnz_row);
    }

    var_nnz_row = var_nnz_row / rows;

    feature[0] = min_nnz_row;
    feature[1] = max_nnz_row;
    feature[2] = rows;
    feature[3] = cols;
    feature[4] = nnz;
    feature[5] = var_nnz_row; //TODO: The calculation result is inconsistent with excel
    feature[6] = avr_nnz_row;
    feature[7] = sparsity;
}

template <typename TYPE>
alphasparseStatus_t
dcu_gemv_csr(alphasparseHandle_t handle,
      ALPHA_INT m,
      ALPHA_INT n,
      ALPHA_INT nnz,
      const TYPE alpha,
      const TYPE *csr_val,
      const ALPHA_INT *csr_row_ptr,
      const ALPHA_INT *csr_col_ind,
      alphasparse_mat_info_t info,
      const TYPE *x,
      const TYPE beta,
      TYPE *y)
{
    alphasparseStatus_t st;
    u_int32_t flag = 0;

    csrgemv_algo algo = ALPHA_CSRMV_VECTOR;

    const ALPHA_INT nnz_per_row = nnz / m;

    // if (info) {
    //     if (info->csrmv_info->algo == ALPHA_CSRMV_AUTO) { //TODO: move to csrmv anasisy interface
    //         const ALPHA_INT FEATURE_NUM = 8;
    //         double feature[8];
    //         double outputs[2];
    //         double time;

    //         // time = get_time_us();
    //         get_feature(handle, csr_row_ptr, m, n, nnz, feature);
    //         // time = (get_time_us() - time) / (1e3);
    //         // std::cout << "get features time: " << time << std::endl;
    //         // for(int i = 0; i < FEATURE_NUM; i++) {
    //         //     printf("%f ", feature[i]);
    //         // }
    //         // printf("\n");

    //         // time = get_time_us();
    //         dcu_csrgemv_algo_dt_score(feature, outputs);
    //         algo                   = outputs[0] > outputs[1] ? ALPHA_CSRMV_VECTOR : ALPHA_CSRMV_ADAPTIVE;
    //         info->csrmv_info->algo = algo;
    //         // time = (get_time_us() - time) / (1e3);
    //         // std::cout << outputs[0] << " " << output[1] << std::endl;
    //         // if (algo == ADAPTIVE)
    //         //     std::cout << time << "ADAPTIVE" << std::endl;
    //         // else
    //         //     std::cout << time << "VECTOR" << std::endl;
    //         // return ALPHA_SPARSE_STATUS_SUCCESS;
    //     } else {
    //         algo = info->csrmv_info->algo;
    //     }
    // }

    if (handle->check_flag) {
        return csr_gemv_vector_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    }

    if (algo == ALPHA_CSRMV_VECTOR) {
        st = csr_gemv_vector_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    // } else if (algo == ALPHA_CSRMV_SCALAR) {
    //     st = csr_gemv_scalar_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag);
    // } else if (algo == ALPHA_CSRMV_ROW_PARTITION) {
    //     st = csr_gemv_row_partition_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag, info);
    // } else if (algo == ALPHA_CSRMV_ADAPTIVE) {
    //     st = csr_gemv_adaptive_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag, info);
    // } else if (algo == ALPHA_CSRMV_MERGE) {
    //     st = csr_gemv_merge_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag, info);
    // } else if (algo == ALPHA_CSRMV_XXX) {
    //     st = csr_gemv_xxx_dispatch(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y, flag, info);
    } else {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    return st;
}

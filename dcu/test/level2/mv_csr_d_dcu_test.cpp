/**
 * @brief ict dcu mv csr test
 * @author HPCRC, ICT
 */

#include "../include/test_common.h"

#include <hip/hip_runtime_api.h>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "alphasparse.h"

const char *file;
bool check;

alphasparseOperation_t transA;
alphasparse_layout_t layout;
struct alpha_matrix_descr descr;

// csr format
ALPHA_INT rm, rk, rnnz;
ALPHA_INT *csr_row_ptr, *csr_row_ptr_end, *csr_col_index;
double *csr_values;

// parms for kernel
double *x;
double *icty;
double *rocy;
ALPHA_INT sizex, sizey;
const double alpha = 2.f;
const double beta  = 3.f;

const ALPHA_INT warm_up = 5;
ALPHA_INT trials        = 10;

static void roc_mv_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);

    rocsparse_int m   = rm;
    rocsparse_int n   = rk;
    rocsparse_int nnz = rnnz;

    // Generate problem
    std::vector<rocsparse_int> hAptr(m + 1);
    std::vector<rocsparse_int> hAcol(nnz);
    std::vector<double> hAval(nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    rocsparse_int *dAptr = NULL;
    rocsparse_int *dAcol = NULL;
    double *dAval        = NULL;
    double *dx           = NULL;
    double *dy           = NULL;

    hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void **)&dAval, sizeof(double) * nnz);
    hipMalloc((void **)&dx, sizeof(double) * sizex);
    hipMalloc((void **)&dy, sizeof(double) * sizey);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx, x, sizeof(double) * sizex, hipMemcpyHostToDevice);
    hipMemcpy(dy, rocy, sizeof(double) * sizey, hipMemcpyHostToDevice);

    double halpha = alpha;
    double hbeta  = beta;

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrmv
        roc_call_exit(
            rocsparse_dcsrmv(handle, rocsparse_operation_none, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, nullptr, dx, &hbeta, dy),
            "rocsparse_scsrmv");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        // Call rocsparse csrmv
        roc_call_exit(rocsparse_dcsrmv(handle, rocsparse_operation_none, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, nullptr, dx, &hbeta, dy),
                      "rocsparse_scsrmv");

        // Device synchronization
        hipDeviceSynchronize();
    }

    time             = (get_time_us() - time) / (trials * 1e3);
    double bandwidth = static_cast<double>(sizeof(double) * (2 * m + nnz) + sizeof(rocsparse_int) * (m + 1 + nnz)) / time / 1e6;
    double gflops    = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout << time;
    // std::cout << " " << bandwidth;
    // std::cout << " " << gflops;
    std::cout << std::endl;

    hipMemcpy(rocy, dy, sizeof(double) * sizey, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);
}

static void alpha_mv_dcu()
{
    // rocSPARSE handle
    alphasparseHandle_t handle;
    init_handle(&handle);
    alphasparse_get_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);

    // Generate problem
    ALPHA_INT m   = rm;
    ALPHA_INT n   = rk;
    ALPHA_INT nnz = rnnz;

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
    ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz);
    double *hAval    = (double *)alpha_malloc(sizeof(double) * nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    ALPHA_INT *dAptr = NULL;
    ALPHA_INT *dAcol = NULL;
    double *dAval    = NULL;
    double *dx       = NULL;
    double *dy       = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(double) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dx, sizeof(double) * sizex));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dy, sizeof(double) * sizey));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dx, x, sizeof(double) * sizex, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dy, icty, sizeof(double) * sizey, hipMemcpyHostToDevice));

    double halpha = alpha;
    double hbeta  = beta;

    // Matrix descriptor
    alpha_matrix_descr_t descrA;
    alphasparse_create_mat_descr(&descrA);

    alphasparse_mat_info_t info          = (alphasparse_mat_info_t)alpha_malloc(sizeof(struct _alphasparse_mat_info));
    alphasparse_csrmv_info_t csrmv_info  = (alphasparse_csrmv_info_t)alpha_malloc(sizeof(struct _alphasparse_csrmv_info));
    info->csrmv_info                         = csrmv_info;
    info->csrmv_info->csr_adaptive_has_tuned = false;

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu csrmv
        alpha_call_exit(
            alphasparse_d_csrmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, info, dx, &hbeta, dy),
            "alphasparse_dcu_d_csrmv");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        // Call alphasparse_dcu csrmv
        alpha_call_exit(
            alphasparse_d_csrmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, info, dx, &hbeta, dy),
            "alphasparse_dcu_d_csrmv");
        // Device synchronization
        hipDeviceSynchronize();
    }

    time             = (get_time_us() - time) / (trials * 1e3);
    double bandwidth = static_cast<double>(sizeof(double)) * (2 * m + nnz) + sizeof(ALPHA_INT) * (m + 1 + nnz) / time / 1e6;
    double gflops    = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout << time;
    // std::cout << " " << bandwidth;
    // std::cout << " " << gflops;
    std::cout << ",";

    hipMemcpy(icty, dy, sizeof(double) * sizey, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    alphasparse_destroy_mat_descr(descrA);
    alphasparse_destory_handle(handle);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    file   = args_get_data_file(argc, argv);
    check  = args_get_if_check(argc, argv);
    transA = alpha_args_get_transA(argc, argv);
    descr  = alpha_args_get_matrix_descrA(argc, argv);

    alphasparseIndexBase_t csr_index;

    alphasparse_matrix_t coo, csr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    double *coo_values;
    // read coo
    alpha_read_coo_d(file, &rm, &rk, &rnnz, &coo_row_index, &coo_col_index, &coo_values);

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, rm, rk, rnnz, coo_row_index, coo_col_index, coo_values),
        "alphasparse_d_create_coo");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(
        alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
        "alphasparse_convert_csr");
    // 获取csr格式里的数据
    alpha_call_exit(
        alphasparse_d_export_csr(csr, &csr_index, &rm, &rk, &csr_row_ptr, &csr_row_ptr_end, &csr_col_index, &csr_values),
        "alphasparse_d_export_csr");

    sizex = rk, sizey = rm;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        sizex = rm;
        sizey = rk;
    }
    // init x y
    x    = (double *)alpha_malloc(sizex * sizeof(double));
    icty = (double *)alpha_malloc(sizey * sizeof(double));
    rocy = (double *)alpha_malloc(sizey * sizeof(double));

    alpha_fill_random_d(x, 0, sizex);
    alpha_fill_random_d(icty, 1, sizey);
    alpha_fill_random_d(rocy, 1, sizey);

    alpha_mv_dcu();

    if (check) {
        roc_mv_dcu();

        check_d((double *)rocy, sizey, (double *)icty, sizey);
    }
    printf("\n");

    alpha_free(x);
    alpha_free(icty);
    alpha_free(rocy);
    // alpha_free(csr_row_ptr);
    // alpha_free(csr_row_ptr_end);
    // alpha_free(csr_col_index);
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);
    return 0;
}
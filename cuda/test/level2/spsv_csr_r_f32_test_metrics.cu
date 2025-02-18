
#include "../test_common.h"

/**
 * @brief ict dcu mv hyb test
 * @author HPCRC, ICT
 */

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../format/alphasparse_create_csr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"
#include "../include/spsv/mat_adjust.h"
#include "../include/common.h"
#include "alphasparse.h"
#include <iostream>
#include <chrono>

typedef float DATA_TYPE;

const char* file;
const char* metrics_file;
int thread_num;
bool check_flag;
bool metrics_flag;
int iter = 1;
int warmup = 1;

const char* filename;

float cuda_time_bufferSize;
float cuda_time_analysis;
float cuda_time_solve_avg;
double cuda_time_total;
// double cuda_bandwidth;
// double cuda_gflops;

float alpha_time_bufferSize;
float alpha_time_analysis;
float alpha_time_solve_avg;
double alpha_time_total;
// double alpha_bandwidth;
// double alpha_gflops;

double speedup;

int m, n, nnz;
int *coo_row_index, *coo_col_index;
DATA_TYPE* coo_values;

// coo format
DATA_TYPE* x_val;
DATA_TYPE* ict_y;
DATA_TYPE* cuda_y;

float error;

// parms for kernel
const DATA_TYPE alpha = 2.f;

cudaDataType cuda_datatype;
alphasparseDataType alpha_datatype;

cusparseOperation_t cuda_op;
alphasparseOperation_t alpha_op;

cusparseFillMode_t cuda_fillmode;
alphasparse_fill_mode_t alpha_fillmode;

cusparseDiagType_t cuda_diagtype;
alphasparse_diag_type_t alpha_diagtype;

cusparseSpSVAlg_t cuda_alg;
alphasparseSpSVAlg_t alpha_alg;

cudaEvent_t event_start, event_stop;

static void
cuda_spsv() {
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Offload data to device
    DATA_TYPE* dX = NULL;
    DATA_TYPE* dY = NULL;
    int* dCsrRowPtr = NULL;
    int* dArow = NULL;
    int* dAcol = NULL;
    DATA_TYPE* dAval = NULL;

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(DATA_TYPE) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));
    CHECK_CUDA(cudaMemcpy(dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dAval, coo_values, nnz * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);
    cusparseDnVecDescr_t vecX, vecY;
    cusparseSpMatDescr_t matA;
    CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(DATA_TYPE)));
    CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(DATA_TYPE)));
    CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, cuda_y, m * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, dX, cuda_datatype));
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, dY, cuda_datatype));
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
                                    m,
                                    n,
                                    nnz,
                                    dCsrRowPtr,
                                    dAcol,
                                    dAval,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    cuda_datatype));
    cusparseSpSVDescr_t spsvDescr;
    cusparseSpSV_createDescr(&spsvDescr);
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &cuda_fillmode, sizeof(cuda_fillmode)))
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &cuda_diagtype, sizeof(cuda_diagtype)))
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    GPU_TIMER_START(cuda_time_bufferSize, event_start, event_stop);
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle,
                                            cuda_op,
                                            &alpha,
                                            matA,
                                            vecX,
                                            vecY,
                                            cuda_datatype,
                                            cuda_alg,
                                            spsvDescr,
                                            &bufferSize))
    GPU_TIMER_END(cuda_time_bufferSize, event_start, event_stop);

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    GPU_TIMER_START(cuda_time_analysis, event_start, event_stop);
    CHECK_CUSPARSE(cusparseSpSV_analysis(handle,
                                        cuda_op,
                                        &alpha,
                                        matA,
                                        vecX,
                                        vecY,
                                        cuda_datatype,
                                        cuda_alg,
                                        spsvDescr,
                                        dBuffer))
    GPU_TIMER_END(cuda_time_analysis, event_start, event_stop);

    std::vector<double> times_solve;
    for (int i = 0; i < iter + warmup; i++) {
        float time_solve;
        GPU_TIMER_START(time_solve, event_start, event_stop);
        CHECK_CUSPARSE(cusparseSpSV_solve(handle,
                                        cuda_op,
                                        &alpha,
                                        matA,
                                        vecX,
                                        vecY,
                                        cuda_datatype,
                                        cuda_alg,
                                        spsvDescr))
        cudaDeviceSynchronize();
        GPU_TIMER_END(time_solve, event_start, event_stop);
        if (i >= warmup) {
            times_solve.push_back(time_solve);
        }
    }
    cuda_time_solve_avg = get_avg_time_2(times_solve);
    // cuda_bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + 
    //                  sizeof(int) * (m + 1 + nnz) / cuda_time_solve_avg / 1e6;
    // cuda_gflops = static_cast<double>(2 * nnz) / cuda_time_solve_avg / 1e6;

    CHECK_CUDA(cudaMemcpy(cuda_y, dY, sizeof(DATA_TYPE) * m, cudaMemcpyDeviceToHost));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE(cusparseDestroy(handle))

    CHECK_CUDA(cudaFree(dArow));
    CHECK_CUDA(cudaFree(dAcol));
    CHECK_CUDA(cudaFree(dAval));
    CHECK_CUDA(cudaFree(dCsrRowPtr));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dBuffer));
}

static void
alpha_spsv() {
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    // Offload data to device
    DATA_TYPE* dX = NULL;
    DATA_TYPE* dY = NULL;
    int* dCsrRowPtr = NULL;
    int* dArow = NULL;
    int* dAcol = NULL;
    DATA_TYPE* dAval = NULL;

    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(DATA_TYPE) * nnz));
    PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1)));

    CHECK_CUDA(cudaMemcpy(dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dAval, coo_values, nnz * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, m, dCsrRowPtr);

    alphasparseDnVecDescr_t vecX, vecY;
    alphasparseSpMatDescr_t matA;
    CHECK_CUDA(cudaMalloc((void**)&dX, n * sizeof(DATA_TYPE)));
    CHECK_CUDA(cudaMalloc((void**)&dY, m * sizeof(DATA_TYPE)));
    CHECK_CUDA(cudaMemcpy(dX, x_val, n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, ict_y, m * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    alphasparseCreateDnVec(&vecX, n, (void*)dX, alpha_datatype);
    alphasparseCreateDnVec(&vecY, m, (void*)dY, alpha_datatype);
    alphasparseCreateCsr(&matA,
                        m,
                        n,
                        nnz,
                        dCsrRowPtr,
                        dAcol,
                        dAval,
                        ALPHA_SPARSE_INDEXTYPE_I32,
                        ALPHA_SPARSE_INDEXTYPE_I32,
                        ALPHA_SPARSE_INDEX_BASE_ZERO,
                        alpha_datatype);
    alphasparseSpSVDescr_t spsvDescr;
    alphasparseSpSV_createDescr(&spsvDescr);
    alphasparseSpMatSetAttribute(matA, ALPHASPARSE_SPMAT_FILL_MODE, &alpha_fillmode, sizeof(alpha_fillmode));
    alphasparseSpMatSetAttribute(matA, ALPHASPARSE_SPMAT_DIAG_TYPE, &alpha_diagtype, sizeof(alpha_diagtype));
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    GPU_TIMER_START(alpha_time_bufferSize, event_start, event_stop);
    alphasparseSpSV_bufferSize(handle,
                                alpha_op,
                                &alpha,
                                matA,
                                vecX,
                                vecY,
                                alpha_datatype,
                                alpha_alg,
                                spsvDescr,
                                &bufferSize);
    GPU_TIMER_END(alpha_time_bufferSize, event_start, event_stop);

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    GPU_TIMER_START(alpha_time_analysis, event_start, event_stop);
    alphasparseSpSV_analysis(handle,
                            alpha_op,
                            &alpha,
                            matA,
                            vecX,
                            vecY,
                            alpha_datatype,
                            alpha_alg,
                            spsvDescr,
                            dBuffer);
    GPU_TIMER_END(alpha_time_analysis, event_start, event_stop);

    std::vector<double> times_solve;
    for (int i = 0; i < iter + warmup; i++) {
        float time_solve;
        GPU_TIMER_START(time_solve, event_start, event_stop);
        alphasparseSpSV_solve(handle,
                            alpha_op,
                            &alpha,
                            matA,
                            vecX,
                            vecY,
                            alpha_datatype,
                            alpha_alg,
                            spsvDescr,
                            dBuffer);
        cudaDeviceSynchronize();
        GPU_TIMER_END(time_solve, event_start, event_stop);

        if (i >= warmup) {
            times_solve.push_back(time_solve);
        }
    }
    alpha_time_solve_avg = get_avg_time_2(times_solve);
    // alpha_bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + 
    //                   sizeof(int) * (m + 1 + nnz) / alpha_time_solve_avg / 1e6;
    // alpha_gflops = static_cast<double>(2 * nnz) / alpha_time_solve_avg / 1e6;

    CHECK_CUDA(cudaMemcpy(ict_y, dY, sizeof(DATA_TYPE) * m, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dArow));
    CHECK_CUDA(cudaFree(dAcol));
    CHECK_CUDA(cudaFree(dAval));
    CHECK_CUDA(cudaFree(dCsrRowPtr));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dBuffer));
}

int
main(int argc, const char* argv[]) {
    // args
    args_help(argc, argv);
    file = args_get_data_file(argc, argv);
    metrics_file = args_save_metrics_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    metrics_flag = args_get_if_calculate_metrics(argc, argv);
    iter = args_get_iter(argc, argv);
    warmup = args_get_warmup(argc, argv);

    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    alpha_matrix_descr mat_descrA = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_fill_mode_t fillA = mat_descrA.mode;
    alphasparse_diag_type_t diagA = mat_descrA.diag;

    int algo_num = args_get_alg_num(argc, argv);

    alpha_datatype = get_alpha_datatype<DATA_TYPE>();
    cuda_datatype = alpha2cuda_datatype_map[alpha_datatype];
    alpha_op = transA;
    cuda_op = alpha2cuda_op_map[alpha_op];
    alpha_fillmode = fillA;
    cuda_fillmode = alpha2cuda_fill_map[alpha_fillmode];
    alpha_diagtype = diagA;
    cuda_diagtype = alpha2cuda_diag_map[alpha_diagtype];
    alpha_alg = get_alpha_spsv_alg(algo_num);
    cuda_alg = CUSPARSE_SPSV_ALG_DEFAULT;

    filename = get_filename(file);

    // read coo
    alpha_read_coo<DATA_TYPE>(file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
    
    if (m != n || m == 0) {
        // printf("%s - the row number and the column number are NOT equal!\n", file);
        return 0;
    }
    // if (nnz < 1e+6) {
    //     return 0;
    // }

    coo_order<int32_t, DATA_TYPE>(nnz, coo_row_index, coo_col_index, coo_values);
    
    if (alpha_diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        // 补充对角线元素
        mat_patch_trim_s<int32_t>(&m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
    }
    // 矩阵行元素求和归一化
    mat_adjust_nnz_s(coo_row_index, coo_col_index, coo_values, m, n, nnz, alpha_fillmode, alpha_diagtype);

    // init x y
    x_val = (DATA_TYPE*)alpha_malloc(n * sizeof(DATA_TYPE));

    alpha_fill_random(x_val, 0, n);
    
    // int execute_type = 0;
    int execute_type = 1;
    if (execute_type == 0) {
        if (algo_num < 0) {
            // printf("\n%s,", filename);
            cuda_y = (DATA_TYPE*)alpha_malloc(m * sizeof(DATA_TYPE));
            alpha_fill_random(cuda_y, 1, m);
            cuda_spsv();
            cuda_time_total = cuda_time_bufferSize + cuda_time_analysis + cuda_time_solve_avg;
            if (metrics_flag) {
                std::ofstream outfile(metrics_file, std::ios::app);
                outfile << "CSR,"
                        << cuda_op_map[cuda_op] << ","
                        << cuda_fill_map[cuda_fillmode] << ","
                        << cuda_diag_map[cuda_diagtype] << ","
                        << cuda_spsv_alg_map[cuda_alg] << ","
                        << cuda_datatype_map[cuda_datatype] << ","
                        << filename << ","
                        << cuda_time_bufferSize << ","
                        << cuda_time_analysis << ","
                        << cuda_time_solve_avg << ","
                        << cuda_time_total << "\n";
                outfile.close();
            }
            printf("%.6lf,", cuda_time_total);
            free(cuda_y);
        } else {
            ict_y = (DATA_TYPE*)alpha_malloc(m * sizeof(DATA_TYPE));
            alpha_fill_random(ict_y, 1, m);
            alpha_spsv();
            alpha_time_total = alpha_time_bufferSize + alpha_time_analysis + alpha_time_solve_avg;
            if (metrics_flag) {
                std::ofstream outfile(metrics_file, std::ios::app);
                outfile << "CSR,"
                        << alpha_op_map[alpha_op] << ","
                        << alpha_fill_map[alpha_fillmode] << ","
                        << alpha_diag_map[alpha_diagtype] << ","
                        << alpha_spsv_alg_map[alpha_alg] << ","
                        << alpha_datatype_map[alpha_datatype] << ","
                        << filename << ","
                        << alpha_time_bufferSize << ","
                        << alpha_time_analysis << ","
                        << alpha_time_solve_avg << ","
                        << alpha_time_total << "\n";
                outfile.close();
            }
            printf("%.6lf,", alpha_time_total);
            free(ict_y);
        }
    } else if (execute_type == 1) {
        cuda_y = (DATA_TYPE*)alpha_malloc(m * sizeof(DATA_TYPE));
        alpha_fill_random(cuda_y, 1, m);
        cuda_spsv();
        cuda_time_total = cuda_time_bufferSize + cuda_time_analysis + cuda_time_solve_avg;
        ict_y = (DATA_TYPE*)alpha_malloc(m * sizeof(DATA_TYPE));
        alpha_fill_random(ict_y, 1, m);
        alpha_spsv();
        alpha_time_total = alpha_time_bufferSize + alpha_time_analysis + alpha_time_solve_avg;
        double speedup_analysis = cuda_time_analysis / alpha_time_analysis;
        double speedup_solve = cuda_time_solve_avg / alpha_time_solve_avg;
        speedup = cuda_time_total / alpha_time_total;
        if (check_flag) {
            check((DATA_TYPE*)cuda_y, m, (DATA_TYPE*)ict_y, m, &error);
        }
        if (metrics_flag) {
            std::ofstream outfile(metrics_file, std::ios::app);
            outfile << "CSR,"
                    << cuda_op_map[cuda_op] << ","
                    << cuda_fill_map[cuda_fillmode] << ","
                    << cuda_diag_map[cuda_diagtype] << ","
                    << cuda_spsv_alg_map[cuda_alg] << ","
                    << cuda_datatype_map[cuda_datatype] << ","
                    << filename << ","
                    << cuda_time_bufferSize << ","
                    << cuda_time_analysis << ","
                    << cuda_time_solve_avg << ","
                    << cuda_time_total << "\n";
            outfile << "CSR,"
                    << alpha_op_map[alpha_op] << ","
                    << alpha_fill_map[alpha_fillmode] << ","
                    << alpha_diag_map[alpha_diagtype] << ","
                    << alpha_spsv_alg_map[alpha_alg] << ","
                    << alpha_datatype_map[alpha_datatype] << ","
                    << filename << ","
                    << alpha_time_bufferSize << ","
                    << alpha_time_analysis << ","
                    << alpha_time_solve_avg << ","
                    << alpha_time_total << ","
                    << speedup_analysis << ","
                    << speedup_solve << ","
                    << speedup;
            if (check_flag) {
                outfile << "," << error;
            }
            outfile << "\n";
            outfile.close();
        }
        printf("%.6lf,", speedup_solve);
        if (check_flag) {
            std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
            std::cout.precision(12);
            std::cout << "\nError: " << error << std::endl;
            std::cout << "\t i\t\tcuda_y, \t\tict_y\n"; 

            for (int i = 0, cnt = 0; i < m && cnt < 20; i++) {
                if (fabs(cuda_y[i] - ict_y[i]) > 1e-6) {
                    std::cout << cnt << ", \t";
                    std::cout << i << ", \t";
                    std::cout << cuda_y[i] << ", \t";
                    std::cout << ict_y[i] << std::endl;
                    cnt++;
                }
            }
            std::cout << std::endl;
        }
        free(ict_y);
        free(cuda_y);
    }
    free(x_val);

    return 0;
}

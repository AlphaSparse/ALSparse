#include <assert.h>
#include <alphasparse.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>

#include "include/args.h"
#include "include/io.h"
#include "include/norm.h"

#define THRESH 50.0f
int check_s_l2(const float *answer_data, size_t answer_size, const float *result_data,
               size_t result_size, const float *x, const float *y, const float alpha,
               const float beta, int argc, const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_indx, *col_indx;
  float *val, normA;
  alpha_read_coo(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  ALPHA_INT size_x = k;
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    size_x = m;
  }
  float normD = InfiniteNorm_s(answer_size, answer_data, result_data);
  float normX = InfiniteNorm_s(size_x, x, NULL);
  if (fabs(alpha) > 1.0f) normX *= alpha;
  float normY = InfiniteNorm_s(answer_size, result_data, NULL);
  if (fabs(beta) > 1.0f) normY *= beta;
  float eps = CalEpisilon_s();

  if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
    normA = GeNorm1_s(k, nnz, col_indx, val);
  else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
    normA = DiNorm1_s(k, nnz, row_indx, col_indx, val, descr);
  else
    normA = TrSyNorm1_s(k, nnz, row_indx, col_indx, val, descr);

  alpha_free(row_indx);
  alpha_free(col_indx);
  alpha_free(val);

  float resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                         alpha_max(m, k));
  printf(
      "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, row "
      ": %d, col : %d\n",
      normD, normX, normY, normA, eps, nnz, m, k);
  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_d_l2(const double *answer_data, size_t answer_size, const double *result_data,
               size_t result_size, const double *x, const double *y, const double alpha,
               const double beta, int argc, const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_indx, *col_indx;
  double *val, normA;
  alpha_read_coo_d(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  ALPHA_INT size_x = k;
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    size_x = m;
  }
  double normD = InfiniteNorm_d(answer_size, answer_data, result_data);
  double normX = InfiniteNorm_d(size_x, x, NULL);
  if (fabs(alpha) > 1.0f) normX *= alpha;
  double normY = InfiniteNorm_d(answer_size, result_data, NULL);
  if (fabs(beta) > 1.0f) normY *= beta;
  double eps = CalEpisilon_d();

  if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
    normA = GeNorm1_d(k, nnz, col_indx, val);
  else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
    normA = DiNorm1_d(k, nnz, row_indx, col_indx, val, descr);
  else
    normA = TrSyNorm1_d(k, nnz, row_indx, col_indx, val, descr);

  alpha_free(row_indx);
  alpha_free(col_indx);
  alpha_free(val);

  double resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                          alpha_max(m, k));
  printf(
      "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, row "
      ": %d, col : %d\n",
      normD, normX, normY, normA, eps, nnz, m, k);
  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_c_l2(const ALPHA_Complex8 *answer_data, size_t answer_size, const ALPHA_Complex8 *result_data,
               size_t result_size, const ALPHA_Complex8 *x, const ALPHA_Complex8 *y,
               const ALPHA_Complex8 alpha, const ALPHA_Complex8 beta, int argc, const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_indx, *col_indx;
  ALPHA_Complex8 *val;
  float normA;
  alpha_read_coo_c(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  ALPHA_INT size_x = k;
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    size_x = m;
  }
  float normD = InfiniteNorm_c(answer_size, answer_data, result_data);
  float normX = InfiniteNorm_c(size_x, x, NULL);
  if (fabs(alpha.real) > 1.0f) normX *= alpha.real;
  float normY = InfiniteNorm_c(answer_size, result_data, NULL);
  if (fabs(beta.real) > 1.0f) normY *= beta.real;
  float eps = CalEpisilon_s();

  if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
    normA = GeNorm1_c(k, nnz, col_indx, val);
  else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
    normA = DiNorm1_c(k, nnz, row_indx, col_indx, val, descr);
  else
    normA = TrSyNorm1_c(k, nnz, row_indx, col_indx, val, descr);

  alpha_free(row_indx);
  alpha_free(col_indx);
  alpha_free(val);

  float resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                         alpha_max(m, k));
  printf(
      "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, row "
      ": %d, col : %d\n",
      normD, normX, normY, normA, eps, nnz, m, k);
  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_z_l2(const ALPHA_Complex16 *answer_data, size_t answer_size,
               const ALPHA_Complex16 *result_data, size_t result_size, const ALPHA_Complex16 *x,
               const ALPHA_Complex16 *y, const ALPHA_Complex16 alpha, const ALPHA_Complex16 beta,
               int argc, const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_indx, *col_indx;
  ALPHA_Complex16 *val;
  double normA;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  ALPHA_INT size_x = k;
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    size_x = m;
  }
  double normD = InfiniteNorm_z(answer_size, answer_data, result_data);
  double normX = InfiniteNorm_z(size_x, x, NULL);
  if (fabs(alpha.real) > 1.0f) normX *= alpha.real;
  double normY = InfiniteNorm_z(answer_size, result_data, NULL);
  if (fabs(beta.real) > 1.0f) normY *= beta.real;
  double eps = CalEpisilon_d();

  if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
    normA = GeNorm1_z(k, nnz, col_indx, val);
  else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
    normA = DiNorm1_z(k, nnz, row_indx, col_indx, val, descr);
  else
    normA = TrSyNorm1_z(k, nnz, row_indx, col_indx, val, descr);

  alpha_free(row_indx);
  alpha_free(col_indx);
  alpha_free(val);

  double resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                          alpha_max(m, k));
  printf(
      "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, row "
      ": %d, col : %d\n",
      normD, normX, normY, normA, eps, nnz, m, k);
  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_s_l3(const float *answer_data, const ALPHA_INT ldans, size_t answer_size,
               const float *result_data, const ALPHA_INT ldres, size_t result_size,
               const ALPHA_INT *res_col_indx, const float *x, const ALPHA_INT ldx, const float *y,
               const ALPHA_INT ldy, const float alpha, const float beta, int argc,
               const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = args_get_data_fileB(argc, argv);
  float resid;
  if (file[0] != '\0') {  // mv mm trsv and trsm cases
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    float *val, normA;
    alpha_read_coo(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
    ALPHA_INT rowsx = k, rowsy = m;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    ALPHA_INT columns = args_get_columns(argc, argv, k);

    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      rowsx = m;
      rowsy = k;
    }

    float normD = DesDiffNorm1_s(rowsy, columns, answer_data, ldans, result_data, ldres, layout);
    float normX = DesNorm1_s(rowsx, columns, x, ldx, layout);
    float normY = DesNorm1_s(rowsy, columns, result_data, ldy, layout);
    float eps = CalEpisilon_s();

    if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
      normA = GeNorm1_s(k, nnz, col_indx, val);
    else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
      normA = DiNorm1_s(k, nnz, row_indx, col_indx, val, descr);
    else
      normA = TrSyNorm1_s(k, nnz, row_indx, col_indx, val, descr);

    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(alpha_max(m, k), columns));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  } else  // spmm and spmmd cases
  {
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    float *val;
    float normA, normX, normD, normY;
    float eps = CalEpisilon_s();
    ALPHA_INT res_row, res_col;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    const char *fileA, *fileB;
    alphasparse_matrix_t coo;
    fileA = args_get_data_fileA(argc, argv);
    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      fileB = args_get_data_fileB(argc, argv);
    else
      fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo(fileA, &m, &k, &nnz, &row_indx, &col_indx, &val);
    res_row = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      res_row = k;
    }
    ALPHA_INT maxrc = alpha_max(m, k);
    // alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_s_create_coo");
    normA = GeNorm1_s(k, nnz, col_indx, val);
    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);
    alpha_read_coo(fileB, &m, &k, &nnz, &row_indx, &col_indx, &val);
    // alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_s_create_coo");
    normX = GeNorm1_s(k, nnz, col_indx, val);
    res_col = k;

    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    if (res_col_indx == NULL) {
      normD = DesDiffNorm1_s(res_row, res_col, answer_data, ldans, result_data, ldres, layout);
      normY = DesNorm1_s(res_row, res_col, result_data, ldy, layout);
    } else {  // for spmm case
      float *diff_vec = (float *)malloc(sizeof(float) * result_size);
      memset(diff_vec, '\0', sizeof(float) * result_size);
      for (size_t i = 0; i < result_size; i++) diff_vec[i] = answer_data[i] - result_data[i];
      printf("y rows %d cols %d\n", res_row, res_col);
      normD = GeNorm1_s(res_col, result_size, res_col_indx, diff_vec);
      normY = GeNorm1_s(res_col, result_size, res_col_indx, result_data);
      alpha_free(diff_vec);
    }

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(maxrc, k));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  }

  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_d_l3(const double *answer_data, const ALPHA_INT ldans, size_t answer_size,
               const double *result_data, const ALPHA_INT ldres, size_t result_size,
               const ALPHA_INT *res_col_indx, const double *x, const ALPHA_INT ldx, const double *y,
               const ALPHA_INT ldy, const double alpha, const double beta, int argc,
               const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = args_get_data_fileB(argc, argv);
  double resid;
  if (file[0] != '\0') {  // mv mm trsv and trsm cases
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    double *val, normA;
    alpha_read_coo_d(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
    ALPHA_INT rowsx = k, rowsy = m;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    ALPHA_INT columns = args_get_columns(argc, argv, k);

    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      rowsx = m;
      rowsy = k;
    }

    double normD = DesDiffNorm1_d(rowsy, columns, answer_data, ldans, result_data, ldres, layout);
    double normX = DesNorm1_d(rowsx, columns, x, ldx, layout);
    double normY = DesNorm1_d(rowsy, columns, result_data, ldy, layout);
    double eps = CalEpisilon_d();

    if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
      normA = GeNorm1_d(k, nnz, col_indx, val);
    else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
      normA = DiNorm1_d(k, nnz, row_indx, col_indx, val, descr);
    else
      normA = TrSyNorm1_d(k, nnz, row_indx, col_indx, val, descr);

    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(alpha_max(m, k), columns));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  } else  // spmm and spmmd cases
  {
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    double *val;
    double normA, normX, normD, normY;
    double eps = CalEpisilon_d();
    ALPHA_INT res_row, res_col;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    const char *fileA, *fileB;
    alphasparse_matrix_t coo;
    fileA = args_get_data_fileA(argc, argv);
    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      fileB = args_get_data_fileB(argc, argv);
    else
      fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileA, &m, &k, &nnz, &row_indx, &col_indx, &val);
    res_row = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      res_row = k;
    }
    ALPHA_INT maxrc = alpha_max(m, k);
    // alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_d_create_coo");
    normA = GeNorm1_d(k, nnz, col_indx, val);
    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);
    alpha_read_coo_d(fileB, &m, &k, &nnz, &row_indx, &col_indx, &val);
    // alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_d_create_coo");
    normX = GeNorm1_d(k, nnz, col_indx, val);
    res_col = k;

    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    if (res_col_indx == NULL) {
      normD = DesDiffNorm1_d(res_row, res_col, answer_data, ldans, result_data, ldres, layout);
      normY = DesNorm1_d(res_row, res_col, result_data, ldy, layout);
    } else {  // for spmm case
      double *diff_vec = (double *)malloc(sizeof(double) * result_size);
      memset(diff_vec, '\0', sizeof(double) * result_size);
      for (size_t i = 0; i < result_size; i++) diff_vec[i] = answer_data[i] - result_data[i];
      printf("y rows %d cols %d\n", res_row, res_col);
      normD = GeNorm1_d(res_col, result_size, res_col_indx, diff_vec);
      normY = GeNorm1_d(res_col, result_size, res_col_indx, result_data);
      alpha_free(diff_vec);
    }

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(maxrc, k));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  }

  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_c_l3(const ALPHA_Complex8 *answer_data, const ALPHA_INT ldans, size_t answer_size,
               const ALPHA_Complex8 *result_data, const ALPHA_INT ldres, size_t result_size,
               const ALPHA_INT *res_col_indx, const ALPHA_Complex8 *x, const ALPHA_INT ldx,
               const ALPHA_Complex8 *y, const ALPHA_INT ldy, const ALPHA_Complex8 alpha,
               const ALPHA_Complex8 beta, int argc, const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = args_get_data_fileB(argc, argv);
  float resid;
  if (file[0] != '\0') {  // mv mm trsv and trsm cases
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    ALPHA_Complex8 *val;
    float normA;
    alpha_read_coo_c(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
    ALPHA_INT rowsx = k, rowsy = m;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    ALPHA_INT columns = args_get_columns(argc, argv, k);

    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      rowsx = m;
      rowsy = k;
    }

    float normD = DesDiffNorm1_c(rowsy, columns, answer_data, ldans, result_data, ldres, layout);
    float normX = DesNorm1_c(rowsx, columns, x, ldx, layout);
    float normY = DesNorm1_c(rowsy, columns, result_data, ldy, layout);
    float eps = CalEpisilon_s();

    if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
      normA = GeNorm1_c(k, nnz, col_indx, val);
    else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
      normA = DiNorm1_c(k, nnz, row_indx, col_indx, val, descr);
    else
      normA = TrSyNorm1_c(k, nnz, row_indx, col_indx, val, descr);

    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(alpha_max(m, k), columns));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  } else  // spmm and spmmd cases
  {
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    ALPHA_Complex8 *val;
    float normA, normX, normD, normY;
    float eps = CalEpisilon_s();
    ALPHA_INT res_row, res_col;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    const char *fileA, *fileB;
    alphasparse_matrix_t coo;
    fileA = args_get_data_fileA(argc, argv);
    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      fileB = args_get_data_fileB(argc, argv);
    else
      fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo_c(fileA, &m, &k, &nnz, &row_indx, &col_indx, &val);
    res_row = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      res_row = k;
    }
    ALPHA_INT maxrc = alpha_max(m, k);
    // alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_d_create_coo");
    normA = GeNorm1_c(k, nnz, col_indx, val);
    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);
    alpha_read_coo_c(fileB, &m, &k, &nnz, &row_indx, &col_indx, &val);
    // alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_d_create_coo");
    normX = GeNorm1_c(k, nnz, col_indx, val);
    res_col = k;

    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    if (res_col_indx == NULL) {
      normD = DesDiffNorm1_c(res_row, res_col, answer_data, ldans, result_data, ldres, layout);
      normY = DesNorm1_c(res_row, res_col, result_data, ldy, layout);
    } else {  // for spmm case
      ALPHA_Complex8 *diff_vec = (ALPHA_Complex8 *)malloc(sizeof(ALPHA_Complex8) * result_size);
      memset(diff_vec, '\0', sizeof(ALPHA_Complex8) * result_size);
      for (size_t i = 0; i < result_size; i++) {
        diff_vec[i].real = answer_data[i].real - result_data[i].real;
        diff_vec[i].imag = answer_data[i].imag - result_data[i].imag;
      }
      printf("y rows %d cols %d\n", res_row, res_col);
      normD = GeNorm1_c(res_col, result_size, res_col_indx, diff_vec);
      normY = GeNorm1_c(res_col, result_size, res_col_indx, result_data);
      alpha_free(diff_vec);
    }

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(maxrc, k));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  }

  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_z_l3(const ALPHA_Complex16 *answer_data, const ALPHA_INT ldans, size_t answer_size,
               const ALPHA_Complex16 *result_data, const ALPHA_INT ldres, size_t result_size,
               const ALPHA_INT *res_col_indx, const ALPHA_Complex16 *x, const ALPHA_INT ldx,
               const ALPHA_Complex16 *y, const ALPHA_INT ldy, const ALPHA_Complex16 alpha,
               const ALPHA_Complex16 beta, int argc, const char *argv[]) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }

  const char *file = args_get_data_file(argc, argv);
  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = args_get_data_fileB(argc, argv);
  double resid;
  if (file[0] != '\0') {  // mv mm trsv and trsm cases
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    ALPHA_Complex16 *val;
    double normA;
    alpha_read_coo_z(file, &m, &k, &nnz, &row_indx, &col_indx, &val);
    ALPHA_INT rowsx = k, rowsy = m;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    ALPHA_INT columns = args_get_columns(argc, argv, k);

    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      rowsx = m;
      rowsy = k;
    }

    double normD = DesDiffNorm1_z(rowsy, columns, answer_data, ldans, result_data, ldres, layout);
    double normX = DesNorm1_z(rowsx, columns, x, ldx, layout);
    double normY = DesNorm1_z(rowsy, columns, result_data, ldy, layout);
    double eps = CalEpisilon_d();

    if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
      normA = GeNorm1_z(k, nnz, col_indx, val);
    else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
      normA = DiNorm1_z(k, nnz, row_indx, col_indx, val, descr);
    else
      normA = TrSyNorm1_z(k, nnz, row_indx, col_indx, val, descr);

    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(alpha_max(m, k), columns));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  } else  // spmm and spmmd cases
  {
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_indx, *col_indx;
    ALPHA_Complex16 *val;
    double normA, normX, normD, normY;
    double eps = CalEpisilon_d();
    ALPHA_INT res_row, res_col;
    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    const char *fileA, *fileB;
    alphasparse_matrix_t coo;
    fileA = args_get_data_fileA(argc, argv);
    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      fileB = args_get_data_fileB(argc, argv);
    else
      fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo_z(fileA, &m, &k, &nnz, &row_indx, &col_indx, &val);
    res_row = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
      res_row = k;
    }
    ALPHA_INT maxrc = alpha_max(m, k);
    // alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_d_create_coo");
    normA = GeNorm1_z(k, nnz, col_indx, val);
    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);
    alpha_read_coo_z(fileB, &m, &k, &nnz, &row_indx, &col_indx, &val);
    // alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_indx,
    //                                       col_indx, val),
    //               "alphasparse_d_create_coo");
    normX = GeNorm1_z(k, nnz, col_indx, val);
    res_col = k;

    alphasparse_destroy(coo);
    alpha_free(row_indx);
    alpha_free(col_indx);
    alpha_free(val);

    if (res_col_indx == NULL) {
      normD = DesDiffNorm1_z(res_row, res_col, answer_data, ldans, result_data, ldres, layout);
      normY = DesNorm1_z(res_row, res_col, result_data, ldy, layout);
    } else {  // for spmm case
      ALPHA_Complex16 *diff_vec = (ALPHA_Complex16 *)malloc(sizeof(ALPHA_Complex16) * result_size);
      memset(diff_vec, '\0', sizeof(ALPHA_Complex16) * result_size);
      for (size_t i = 0; i < result_size; i++) {
        diff_vec[i].real = answer_data[i].real - result_data[i].real;
        diff_vec[i].imag = answer_data[i].imag - result_data[i].imag;
      }
      printf("y rows %d cols %d\n", res_row, res_col);
      normD = GeNorm1_z(res_col, result_size, res_col_indx, diff_vec);
      normY = GeNorm1_z(res_col, result_size, res_col_indx, result_data);
      alpha_free(diff_vec);
    }

    resid = normD / (alpha_max(normA, 1.0f) * alpha_max(normX, 1.0f) * alpha_max(normY, 1.0f) * eps *
                     alpha_max(maxrc, k));
    printf(
        "normD : %.16f, normX : %.16f, normY : %.16f, normA : %.16f, episilon : %.16f, nnz : %d, "
        "row : %d, col : %d\n",
        normD, normX, normY, normA, eps, nnz, m, k);
  }

  if (resid > THRESH) {
    printf("error,%.16f\n", resid);
    return -1;
  } else {
    printf("correct,%.16f\n", resid);
    return 0;
  }
}

int check_s(const float *answer_data, size_t answer_size, const float *result_data,
            size_t result_size) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  float max_error = fabsf(answer_data[0] - result_data[0]);
  float max_result = fabsf(result_data[0]);
  for (size_t i = 1; i < size; i++) {
    float err = fabsf(answer_data[i] - result_data[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabsf(result_data[i]));
  }
  float relative_error = max_error / (max_result);
  if (relative_error > 2e-5) {
    printf("error,%.10f\n", relative_error);
    return -1;
  } else {
    printf("correct,%.10f\n", relative_error);
    return 0;
  }
}

int check_d(const double *answer_data, size_t answer_size, const double *result_data,
            size_t result_size) {
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n", answer_size, result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  double max_error = fabs(answer_data[0] - result_data[0]);
  double max_result = fabs(result_data[0]);
  for (size_t i = 1; i < size; i++) {
    double err = fabs(answer_data[i] - result_data[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabs(result_data[i]));
  }
  double relative_error = max_error / max_result;
  if (relative_error > 2e-12) {
    // printf("\nSignificant numeric error.\n");
    // printf("inf-norm = %.16lf\n\n", relative_error);
    printf("error,%.10f\n", relative_error);
    return -1;
  } else {
    // printf("\ncorrect!!!\n");
    // printf("inf-norm = %.16lf\n\n", relative_error);
    printf("correct,%.10f\n", relative_error);

    return 0;
  }
}

int check_c(const ALPHA_Complex8 *answer_data, size_t answer_size, const ALPHA_Complex8 *result_data,
            size_t result_size) {
  return check_s((float *)answer_data, answer_size * 2, (float *)result_data, result_size * 2);
}

int check_z(const ALPHA_Complex16 *answer_data, size_t answer_size, const ALPHA_Complex16 *result_data,
            size_t result_size) {
  return check_d((double *)answer_data, answer_size * 2, (double *)result_data, result_size * 2);
}

void check_ALPHA_INT_vec(ALPHA_INT *a, ALPHA_INT size_ans, ALPHA_INT *b, ALPHA_INT size_res) {
  if (size_ans != size_res) {
    printf("size not equal %d vs %d\n", size_ans, size_res);
    return;
  }
  for (ALPHA_INT i = 0; i < size_ans; i++) {
    if (a[i] != b[i]) {
      printf("diff in pos %d, %d vs %d\n", i, a[i], b[i]);
      return;
    }
  }
  printf("passed!\n");
}

// void check_matrix_equal(const alphasparse_matrix_t A, const alphasparse_matrix_t B) {
//   if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_s_t *)A->mat)->rows == ((spmat_csr_s_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_s_t *)A->mat)->rows == ((spmat_coo_s_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_s_t *)A->mat)->rows == ((spmat_csc_s_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_s_t *)A->mat)->rows == ((spmat_bsr_s_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_s_t *)A->mat)->rows == ((spmat_dia_s_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_s_t *)A->mat)->rows == ((spmat_sky_s_t *)A->mat)->cols;
//     else
//       return false;
//   } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_d_t *)A->mat)->rows == ((spmat_csr_d_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_d_t *)A->mat)->rows == ((spmat_coo_d_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_d_t *)A->mat)->rows == ((spmat_csc_d_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_d_t *)A->mat)->rows == ((spmat_bsr_d_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_d_t *)A->mat)->rows == ((spmat_dia_d_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_d_t *)A->mat)->rows == ((spmat_sky_d_t *)A->mat)->cols;
//     else
//       return false;
//   } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_c_t *)A->mat)->rows == ((spmat_csr_c_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_c_t *)A->mat)->rows == ((spmat_coo_c_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_c_t *)A->mat)->rows == ((spmat_csc_c_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_c_t *)A->mat)->rows == ((spmat_bsr_c_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_c_t *)A->mat)->rows == ((spmat_dia_c_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_c_t *)A->mat)->rows == ((spmat_sky_c_t *)A->mat)->cols;
//     else
//       return false;
//   } else {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_z_t *)A->mat)->rows == ((spmat_csr_z_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_z_t *)A->mat)->rows == ((spmat_coo_z_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_z_t *)A->mat)->rows == ((spmat_csc_z_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_z_t *)A->mat)->rows == ((spmat_bsr_z_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_z_t *)A->mat)->rows == ((spmat_dia_z_t *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_z_t *)A->mat)->rows == ((spmat_sky_z_t *)A->mat)->cols;
//     else
//       return false;
//   }
// }
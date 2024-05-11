#include <alphasparse.h>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <stdio.h>

#include "include/args.h"
#include "include/io.h"

int
check(const int32_t* answer_data,
      size_t answer_size,
      const int32_t* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  if (std::isinf(answer_data[0]) || std::isinf(result_data[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  int32_t max_error = abs(answer_data[0] - result_data[0]);
  int32_t max_result = abs(result_data[0]);
  long long sum_error = 0;
  long long sum_result = 0;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data[i]) || std::isinf(result_data[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    int32_t err = abs(answer_data[i] - result_data[i]);
    // if (err>1) v
    // printf("\n%f:%f\n", answer_data[i], result_data[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, abs(result_data[i]));
    sum_error += err;
    sum_result += answer_data[i];
  }
  float max_relative_error = (float)max_error / (max_result);
  double sum_relative_error = (double)sum_error / (sum_result);
  std::cout << std::scientific << "max relative error: " << max_relative_error
            << "\nsum relative error: " << sum_relative_error << std::endl;
  return 0;
}

int
check(const int8_t* answer_data,
      size_t answer_size,
      const int8_t* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  if (std::isinf(answer_data[0]) || std::isinf(result_data[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  int32_t max_error = abs(answer_data[0] - result_data[0]);
  int32_t max_result = abs(result_data[0]);
  long long sum_error = 0;
  long long sum_result = 0;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data[i]) || std::isinf(result_data[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    int32_t err = abs(answer_data[i] - result_data[i]);
    // if (err>1) v
    // printf("\n%f:%f\n", answer_data[i], result_data[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, abs(result_data[i]));
    sum_error += err;
    sum_result += answer_data[i];
  }
  float max_relative_error = (float)max_error / (max_result);
  double sum_relative_error = (double)sum_error / (sum_result);
  std::cout << std::scientific << "max relative error: " << max_relative_error
            << "\nsum relative error: " << sum_relative_error << std::endl;
  return 0;
}

int
check(const float* answer_data,
      size_t answer_size,
      const float* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  if (std::isinf(answer_data[0]) || std::isinf(result_data[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  float max_error = fabsf(answer_data[0] - result_data[0]);
  float max_result = fabsf(result_data[0]);
  double sum_error = 0.;
  double sum_result = 0.;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data[i]) || std::isinf(result_data[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    float err = fabsf(answer_data[i] - result_data[i]);
    // if (err>1)
    //   printf("\n========err:%d: %f:%f\n", i, answer_data[i], result_data[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabsf(result_data[i]));
    sum_error += err;
    sum_result += answer_data[i];
  }
  float max_relative_error = max_error / (max_result);
  double sum_relative_error = sum_error / (sum_result);
  if (max_relative_error > 1e-6) {
    std::cout << std::scientific
              << "\nerror: require 1e-6\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return -1;
  } else {
    std::cout << std::scientific
              << "\ncorrect: require 1e-6\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return 0;
  }
}

int
check(const double* answer_data,
      size_t answer_size,
      const double* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  if (std::isinf(answer_data[0]) || std::isinf(result_data[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  double max_error = fabs(answer_data[0] - result_data[0]);
  double max_result = fabs(result_data[0]);
  double sum_error = 0.;
  double sum_result = 0.;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data[i]) || std::isinf(result_data[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    double err = fabs(answer_data[i] - result_data[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabs(result_data[i]));
    sum_error += err;
    sum_result += answer_data[i];
  }
  float max_relative_error = max_error / (max_result);
  double sum_relative_error = sum_error / (sum_result);
  if (max_relative_error > 1e-12) {
    std::cout << std::scientific
              << "\nerror: require 1e-12\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return -1;
  } else {
    std::cout << std::scientific
              << "\ncorrect: require 1e-12\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return 0;
  }
}

int
check(const cuFloatComplex* answer_data,
      size_t answer_size,
      const cuFloatComplex* result_data,
      size_t result_size)
{
  float *answer_data_s, *result_data_s;
  answer_data_s =
    (float*)alpha_memalign(sizeof(float) * answer_size, DEFAULT_ALIGNMENT);
  result_data_s =
    (float*)alpha_memalign(sizeof(float) * result_size, DEFAULT_ALIGNMENT);
  for (int i = 0; i < answer_size; i++) {
    answer_data_s[i] = cuCabsf(answer_data[i]);
    result_data_s[i] = cuCabsf(result_data[i]);
    ;
  }
  return check(
    (float*)answer_data_s, answer_size, (float*)result_data_s, result_size);
}

int
check(const cuDoubleComplex* answer_data,
      size_t answer_size,
      const cuDoubleComplex* result_data,
      size_t result_size)
{
  double *answer_data_d, *result_data_d;
  answer_data_d =
    (double*)alpha_memalign(sizeof(double) * answer_size, DEFAULT_ALIGNMENT);
  result_data_d =
    (double*)alpha_memalign(sizeof(double) * result_size, DEFAULT_ALIGNMENT);
  for (int i = 0; i < answer_size; i++) {
    answer_data_d[i] = cuCabs(answer_data[i]);
    result_data_d[i] = cuCabs(result_data[i]);
    ;
  }
  return check(
    (double*)answer_data_d, answer_size, (double*)result_data_d, result_size);
}

int
check(const half* answer_data,
      size_t answer_size,
      const half* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  float *answer_data_s, *result_data_s;
  answer_data_s =
    (float*)alpha_memalign(sizeof(float) * answer_size, DEFAULT_ALIGNMENT);
  result_data_s =
    (float*)alpha_memalign(sizeof(float) * result_size, DEFAULT_ALIGNMENT);
  for (int i = 0; i < answer_size; i++) {
    answer_data_s[i] = __half2float(answer_data[i]);
    result_data_s[i] = __half2float(result_data[i]);
  }
  if (std::isinf(answer_data_s[0]) || std::isinf(result_data_s[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  float max_error = fabsf(answer_data_s[0] - result_data_s[0]);
  float max_result = fabsf(result_data_s[0]);
  double sum_error = 0.;
  double sum_result = 0.;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data_s[i]) || std::isinf(result_data_s[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    float err = fabsf(answer_data_s[i] - result_data_s[i]);
    // if (err>100)
    //   printf("\n========err: %f:%f\n", answer_data_s[i], result_data_s[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabsf(result_data_s[i]));
    sum_error += err;
    sum_result += answer_data_s[i];
  }
  float max_relative_error = max_error / (max_result);
  double sum_relative_error = sum_error / (sum_result);
  if (max_relative_error > 5e-3) {
    std::cout << std::scientific
              << "\nerror: require 5e-3\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return -1;
  } else {
    std::cout << std::scientific
              << "\ncorrect: require 5e-3\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return 0;
  }
}

int
check(const nv_bfloat16* answer_data,
      size_t answer_size,
      const nv_bfloat16* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  float *answer_data_s, *result_data_s;
  answer_data_s =
    (float*)alpha_memalign(sizeof(float) * answer_size, DEFAULT_ALIGNMENT);
  result_data_s =
    (float*)alpha_memalign(sizeof(float) * result_size, DEFAULT_ALIGNMENT);
  for (int i = 0; i < answer_size; i++) {
    answer_data_s[i] = __bfloat162float(answer_data[i]);
    result_data_s[i] = __bfloat162float(result_data[i]);
  }
  if (std::isinf(answer_data_s[0]) || std::isinf(result_data_s[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  float max_error = fabsf(answer_data_s[0] - result_data_s[0]);
  float max_result = fabsf(result_data_s[0]);
  double sum_error = 0.;
  double sum_result = 0.;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data_s[i]) || std::isinf(result_data_s[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    float err = fabsf(answer_data_s[i] - result_data_s[i]);
    // if (err>100)
    //   printf("\n========err: %f:%f\n", answer_data_s[i], result_data_s[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabsf(result_data_s[i]));
    sum_error += err;
    sum_result += answer_data_s[i];
  }
  float max_relative_error = max_error / (max_result);
  double sum_relative_error = sum_error / (sum_result);
  if (max_relative_error > 1e-2) {
    std::cout << std::scientific
              << "\nerror: require 1e-2\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return -1;
  } else {
    std::cout << std::scientific
              << "\ncorrect: require 1e-2\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return 0;
  }
}

int
check(const half2* answer_data,
      size_t answer_size,
      const half2* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  float *answer_data_s, *result_data_s;
  answer_data_s =
    (float*)alpha_memalign(sizeof(float) * answer_size, DEFAULT_ALIGNMENT);
  result_data_s =
    (float*)alpha_memalign(sizeof(float) * result_size, DEFAULT_ALIGNMENT);
  float ax, ay, rx, ry;
  for (int i = 0; i < answer_size; i++) {
    ax = __half2float(answer_data[i].x);
    ay = __half2float(answer_data[i].y);
    rx = __half2float(result_data[i].x);
    ry = __half2float(result_data[i].y);
    answer_data_s[i] = sqrt(ax * ax + ay * ay);
    result_data_s[i] = sqrt(rx * rx + ry * ry);
  }
  if (std::isinf(answer_data_s[0]) || std::isinf(result_data_s[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  float max_error = fabsf(answer_data_s[0] - result_data_s[0]);
  float max_result = fabsf(result_data_s[0]);
  double sum_error = 0.;
  double sum_result = 0.;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data_s[i]) || std::isinf(result_data_s[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    float err = fabsf(answer_data_s[i] - result_data_s[i]);
    // if (err>100)
    //   printf("\n========err: %f:%f\n", answer_data_s[i], result_data_s[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabsf(result_data_s[i]));
    sum_error += err;
    sum_result += answer_data_s[i];
  }
  float max_relative_error = max_error / (max_result);
  double sum_relative_error = sum_error / (sum_result);
  if (max_relative_error > 5e-3) {
    std::cout << std::scientific
              << "\nerror: require 5e-3\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return -1;
  } else {
    std::cout << std::scientific
              << "\ncorrect: require 5e-3\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return 0;
  }
}

int
check(const nv_bfloat162* answer_data,
      size_t answer_size,
      const nv_bfloat162* result_data,
      size_t result_size)
{
  if (answer_size != result_size) {
    printf("answer_size ans result_size is not equal (%ld %ld)\n",
           answer_size,
           result_size);
    return -1;
  }
  size_t size = answer_size;
  if (size <= 0) {
    printf("answer_size ans result_size is less than 0\n");
    return -1;
  }
  float *answer_data_s, *result_data_s;
  answer_data_s =
    (float*)alpha_memalign(sizeof(float) * answer_size, DEFAULT_ALIGNMENT);
  result_data_s =
    (float*)alpha_memalign(sizeof(float) * result_size, DEFAULT_ALIGNMENT);
  float ax, ay, rx, ry;
  for (int i = 0; i < answer_size; i++) {
    ax = __bfloat162float(answer_data[i].x);
    ay = __bfloat162float(answer_data[i].y);
    rx = __bfloat162float(result_data[i].x);
    ry = __bfloat162float(result_data[i].y);
    answer_data_s[i] = sqrt(ax * ax + ay * ay);
    result_data_s[i] = sqrt(rx * rx + ry * ry);
  }
  if (std::isinf(answer_data_s[0]) || std::isinf(result_data_s[0])) {
    std::cout << "\nValue overflow! answer_data[0] or result_data[0] is inf.\n";
    return 0;
  }
  float max_error = fabsf(answer_data_s[0] - result_data_s[0]);
  float max_result = fabsf(result_data_s[0]);
  double sum_error = 0.;
  double sum_result = 0.;
  for (size_t i = 1; i < size; i++) {
    if (std::isinf(answer_data_s[i]) || std::isinf(result_data_s[i])) {
      std::cout << "\nValue overflow! answer_data[" << i << "] or result_data["
                << i << "] is inf.\n";
      return 0;
    }
    float err = fabsf(answer_data_s[i] - result_data_s[i]);
    // if (err>100)
    //   printf("\n========err: %f:%f\n", answer_data_s[i], result_data_s[i]);
    max_error = alpha_max(max_error, err);
    max_result = alpha_max(max_result, fabsf(result_data_s[i]));
    sum_error += err;
    sum_result += answer_data_s[i];
  }
  float max_relative_error = max_error / (max_result);
  double sum_relative_error = sum_error / (sum_result);
  if (max_relative_error > 1e-2) {
    std::cout << std::scientific
              << "\nerror: require 1e-2\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return -1;
  } else {
    std::cout << std::scientific
              << "\ncorrect: require 1e-2\nmax relative error: "
              << max_relative_error
              << "\nsum relative error: " << sum_relative_error << std::endl;
    return 0;
  }
}

void
check_int_vec(int* a, int size_ans, int* b, int size_res)
{
  if (size_ans != size_res) {
    printf("size not equal %d vs %d\n", size_ans, size_res);
    return;
  }
  for (int i = 0; i < size_ans; i++) {
    if (a[i] != b[i]) {
      printf("diff in pos %d, %d vs %d\n", i, a[i], b[i]);
      return;
    }
  }
  printf("passed!\n");
}

// void check_matrix_equal(const alphasparse_matrix_t A, const
// alphasparse_matrix_t B) {
//   if (A->datatype == ALPHA_R_32F) {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_s_t *)A->mat)->rows == ((spmat_csr_s_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_s_t *)A->mat)->rows == ((spmat_coo_s_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_s_t *)A->mat)->rows == ((spmat_csc_s_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_s_t *)A->mat)->rows == ((spmat_bsr_s_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_s_t *)A->mat)->rows == ((spmat_dia_s_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_s_t *)A->mat)->rows == ((spmat_sky_s_t
//       *)A->mat)->cols;
//     else
//       return false;
//   } else if (A->datatype == ALPHA_R_64F) {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_d_t *)A->mat)->rows == ((spmat_csr_d_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_d_t *)A->mat)->rows == ((spmat_coo_d_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_d_t *)A->mat)->rows == ((spmat_csc_d_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_d_t *)A->mat)->rows == ((spmat_bsr_d_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_d_t *)A->mat)->rows == ((spmat_dia_d_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_d_t *)A->mat)->rows == ((spmat_sky_d_t
//       *)A->mat)->cols;
//     else
//       return false;
//   } else if (A->datatype == ALPHA_C_32F) {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_c_t *)A->mat)->rows == ((spmat_csr_c_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_c_t *)A->mat)->rows == ((spmat_coo_c_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_c_t *)A->mat)->rows == ((spmat_csc_c_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_c_t *)A->mat)->rows == ((spmat_bsr_c_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_c_t *)A->mat)->rows == ((spmat_dia_c_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_c_t *)A->mat)->rows == ((spmat_sky_c_t
//       *)A->mat)->cols;
//     else
//       return false;
//   } else {
//     if (A->format == ALPHA_SPARSE_FORMAT_CSR)
//       return ((spmat_csr_z_t *)A->mat)->rows == ((spmat_csr_z_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_COO)
//       return ((spmat_coo_z_t *)A->mat)->rows == ((spmat_coo_z_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
//       return ((spmat_csc_z_t *)A->mat)->rows == ((spmat_csc_z_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
//       return ((spmat_bsr_z_t *)A->mat)->rows == ((spmat_bsr_z_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
//       return ((spmat_dia_z_t *)A->mat)->rows == ((spmat_dia_z_t
//       *)A->mat)->cols;
//     else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
//       return ((spmat_sky_z_t *)A->mat)->rows == ((spmat_sky_z_t
//       *)A->mat)->cols;
//     else
//       return false;
//   }
// }
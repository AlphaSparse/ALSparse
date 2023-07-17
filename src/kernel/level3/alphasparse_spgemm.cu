#include "alphasparse_spgemm_csr.h"
#include "alphasparse_spgemm_nnz_csr.h"
#include "alphasparse_spgemm_copy_csr.h"
#include "alphasparse_spgemm_speck_csr.h"
#include <iostream>

static constexpr int spECK_STATIC_MEM_PER_BLOCK {49152};
// spECK_DYNAMIC_MEM_PER_BLOCK should be- 49152 for all devices before Volta and Turing (cc < 7.0)
// - 65536 for Turing devices (cc7.5)
// - 98304 for Volta devices (cc7.0)
// - 101376 for Ampere consumer devices (RTX 30xx) (cc8.6)
// - 166912 for Ampere professional devices (e.g. A100) (cc8.0)
static constexpr int spECK_DYNAMIC_MEM_PER_BLOCK{98304};

template<typename T, typename U>
alphasparseStatus_t
spgemm_template(alphasparseHandle_t handle,
                alphasparseOperation_t opA,
                alphasparseOperation_t opB,
                const void* alpha,
                alphasparseSpMatDescr_t matA,
                alphasparseSpMatDescr_t matB,
                const void* beta,
                alphasparseSpMatDescr_t matC,
                void * externalBuffer2)
{
  switch (matA->format) {
    case ALPHA_SPARSE_FORMAT_CSR: {
      T nnz_C;
      T* dCptr = NULL;
      T* dCcol = NULL;
      U* dCval = NULL;
      // double time1 = get_time_us();
      cudaMalloc((void**)&dCptr, sizeof(T) * (matA->rows + 1));
      // cudaMemset(dCptr, 0, sizeof(T) * (matA->rows + 1));
      spgemm_nnz_csr(handle,
                     (T)matA->rows,
                     (T)matB->cols,
                     (T)matA->cols,
                     (T)matA->nnz,
                     matA->row_data,
                     matA->col_data,
                     (T)matB->nnz,
                     matB->row_data,
                     matB->col_data,
                     (T)0,//(T)matC->nnz,
                     (T*)nullptr,//matC->row_data,
                     (T*)nullptr,//matC->col_data,
                     dCptr,
                     &nnz_C,
                     externalBuffer2);
      // double time2 = get_time_us();
      // printf("nnz compute time: %lf ms nnzC %d\n", (time2 - time1) / (1e3), nnz_C);
      // time1 = get_time_us();
      cudaMalloc((void**)&dCcol, sizeof(T) * nnz_C);
      cudaMalloc((void**)&dCval, sizeof(U) * nnz_C);
      // time2 = get_time_us();
      // printf("allocation time: %lf ms\n", (time2 - time1) / (1e3));
      // T* hCptr = (T*)malloc(sizeof(T) * (matA->rows + 1));
      // cudaMemcpy(hCptr, dCptr, sizeof(T) * (matA->rows + 1), cudaMemcpyDeviceToHost);
      // for(int i = 0; i < 50; i++) printf("hCptr %d = %d\n", i, hCptr[i]);
      bool mul = true;
      bool add = false;
      // time1 = get_time_us();
      spgemm_csr<T, U>(handle,
                       (T)matA->rows,
                       (T)matB->cols,
                       (T)matA->cols,
                       *((U*)alpha),
                       (T*)matA->row_data,
                       (T*)matA->col_data,
                       (U*)matA->val_data,
                       (T)matA->nnz,
                       (T*)matB->row_data,
                       (T*)matB->col_data,
                       (U*)matB->val_data,
                       *((U*)beta),
                      //  (T*)matC->row_data,
                      //  (T*)matC->col_data,
                      //  (U*)matC->val_data,
                       dCptr,//so call D row ptr
                       dCcol,//so call D col ind
                       dCval,//so call D val
                       dCptr,
                       dCcol,
                       dCval,
                       matA->idx_base,
                       matB->idx_base,
                       matC->idx_base,
                       matC->idx_base,
                       mul,
                       add,
                       externalBuffer2);

      matC->val_data = dCval;
      matC->row_data = dCptr;
      matC->col_data = dCcol;
      matC->nnz = nnz_C;
      // time2 = get_time_us();
      // printf("compute time: %lf ms\n", (time2 - time1) / (1e3));
      break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

template<typename T, typename U>
alphasparseStatus_t
spgemm_speck_template(alphasparseHandle_t handle,
                    alphasparseOperation_t opA,
                    alphasparseOperation_t opB,
                    const void* alpha,
                    alphasparseSpMatDescr_t matA,
                    alphasparseSpMatDescr_t matB,
                    const void* beta,
                    alphasparseSpMatDescr_t matC,
                    void * externalBuffer2)
{
  switch (matA->format) {
    case ALPHA_SPARSE_FORMAT_CSR: {      
      spgemm_csr_spECK<T, U, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(handle, opA, opB, *((U*)alpha), matA, matB, *((U*)beta), matC, externalBuffer2);
      break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

// template<typename T, typename U>
// alphasparseStatus_t
// spgemm_fast_template(alphasparseHandle_t handle,
//                     alphasparseOperation_t opA,
//                     alphasparseOperation_t opB,
//                     const void* alpha,
//                     alphasparseSpMatDescr_t matA,
//                     alphasparseSpMatDescr_t matB,
//                     const void* beta,
//                     alphasparseSpMatDescr_t matC,
//                     void * externalBuffer2)
// {
//   switch (matA->format) {
//     case ALPHA_SPARSE_FORMAT_CSR: {      
//       fast(handle, *((U*)alpha), matA, matB, matC);
//       break;
//     }
//     return ALPHA_SPARSE_STATUS_SUCCESS;
//   }
//   return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
// }

template<typename T, typename U>
alphasparseStatus_t
spgemm_copy_template(alphasparseHandle_t handle,
                    alphasparseOperation_t opA,
                    alphasparseOperation_t opB,
                    const void* alpha,
                    alphasparseSpMatDescr_t matA,
                    alphasparseSpMatDescr_t matB,
                    const void* beta,
                    alphasparseSpMatDescr_t matC)
{
  switch (matA->format) {
    case ALPHA_SPARSE_FORMAT_CSR: {      
      spgemm_copy_csr<T, U>(handle,
                       (T*)matC->row_data,
                       (T*)matC->col_data,
                       (U*)matC->val_data,
                       (T)matC->nnz,
                       *((U*)beta),
                       matC->idx_base);
      break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

alphasparseStatus_t
alphasparseSpGEMM_compute(alphasparseHandle_t handle,
                          alphasparseOperation_t opA,
                          alphasparseOperation_t opB,
                          const void* alpha,
                          alphasparseSpMatDescr_t matA,
                          alphasparseSpMatDescr_t matB,
                          const void* beta,
                          alphasparseSpMatDescr_t matC,
                          alphasparseDataType computeType,
                          alphasparseSpGEMMAlg_t alg,
                          alphasparseSpGEMMDescr_t spgemmDescr,
                          size_t* bufferSize2,
                          void* externalBuffer2)
{
  if (externalBuffer2 == NULL) {
    *bufferSize2 = 4;
    // size_t buffer_size = ((matA->nnz - 1) / 256 + 1) * 256;

    // // Group arrays
    // buffer_size += sizeof(int32_t) * 256 * CSRGEMM_MAXGROUPS;
    // buffer_size += sizeof(int32_t) * 256;
    // buffer_size += ((sizeof(int32_t) * matA->rows - 1) / 256 + 1) * 256;

    // // Permutation arrays
    // buffer_size += ((sizeof(int32_t) * matA->rows - 1) / 256 + 1) * 256;
    // buffer_size += ((sizeof(int32_t) * matA->rows - 1) / 256 + 1) * 256;
    // buffer_size += ((sizeof(int32_t) * matA->rows - 1) / 256 + 1) * 256;
    // *bufferSize2 = buffer_size;

    // size_t cubTempBytesScan = 0;
    // void *cubTmp = nullptr;
    // cub::DeviceScan::ExclusiveSum(cubTmp, cubTempBytesScan, (uint32_t *)matC->row_data, (uint32_t *)matC->row_data, matC->rows + 1);
    // size_t buffer_size = cubTempBytesScan;
    // size_t d_combined_pointers_size = sizeof(uint32_t) * (4 + 2 * matA->rows) + divup(cubTempBytesScan, sizeof(uint32_t)) * sizeof(uint32_t);
    // if (matA->nnz > 10000)
    //     d_combined_pointers_size += sizeof(uint32_t) * matA->rows;
    // buffer_size += d_combined_pointers_size;
    // const int kernelCountNumeric = 6;
    // const int kernelCountCounting = 6;
    // uint32_t maxRowLength = max(1, (uint32_t)matB->cols * 12 / 10);
    // const int staticSharedMemPerBlockCounting = 48, staticSharedMemPerBlockNumeric = 24;
    // const int warpsNumeric = 1024 / 32;
    // const int warpsCounting = 1024 / 32;
    // const int sharedBytesPerWarpNumeric = spECK_STATIC_MEM_PER_BLOCK / warpsNumeric - staticSharedMemPerBlockNumeric; 
    // const int entriesPerWarpNumeric = sharedBytesPerWarpNumeric / (sizeof(uint32_t) + sizeof(double));
    // const int sharedBytesPerWarpCounting = spECK_STATIC_MEM_PER_BLOCK / warpsNumeric - staticSharedMemPerBlockCounting;
    // const int entriesPerWarpCounting = sharedBytesPerWarpCounting / sizeof(uint32_t);
    // int maxNnzPerBlockNumeric = entriesPerWarpNumeric * warpsNumeric * 2 / 3;
    // int maxNnzPerBlockCounting = entriesPerWarpCounting * warpsCounting * 4 / 5;
    // uint32_t actualKernelCount = min(kernelCountCounting,
    //                                  uint32_t(
    //                                      std::log2(
    //                                          divup(
    //                                              int(maxRowLength),
    //                                              min(
    //                                                  maxNnzPerBlockCounting >> (kernelCountCounting - 1),
    //                                                  maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)))) +
    //                                      1));
    // size_t combinedBlockStartSize = sizeof(uint32_t) * (1 + kernelCountCounting + matA->rows * (1 + actualKernelCount));
    // buffer_size += combinedBlockStartSize;
    // buffer_size += sizeof(uint32_t) ;
    // *bufferSize2 = buffer_size;
    // printf("buffer_size %d\n",buffer_size);
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F) {
    return spgemm_speck_template<uint32_t, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F) {
    // return spgemm_template<int32_t, double>(
    return spgemm_speck_template<uint32_t, double>(
      handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
    return spgemm_template<int32_t, cuFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
    return spgemm_template<int32_t, cuDoubleComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
  //     computeType == ALPHA_R_16F) {
  //   return spgemm_template<int32_t, half>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_C_16F && matC->data_type == ALPHA_C_16F) {
  //   return spgemm_template<int32_t, half2>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  // }
#if (CUDA_ARCH >= 80)
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16BF && matC->data_type == ALPHA_R_16BF) {
    return spgemm_template<int32_t, nv_bfloat16>(
      handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16BF && matC->data_type == ALPHA_C_16BF) {
    return spgemm_template<int32_t, nv_bfloat162>(
      handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
#endif
  return ALPHA_SPARSE_STATUS_SUCCESS;
}


alphasparseStatus_t
alphasparseSpGEMM_copy(alphasparseHandle_t handle,
                      alphasparseOperation_t opA,
                      alphasparseOperation_t opB,
                      const void* alpha,
                      alphasparseSpMatDescr_t matA,
                      alphasparseSpMatDescr_t matB,
                      const void* beta,
                      alphasparseSpMatDescr_t matC,
                      alphasparseDataType computeType,
                      alphasparseSpGEMMAlg_t alg,
                      alphasparseSpGEMMDescr_t spgemmDescr)
{  
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F) {
    return spgemm_copy_template<int32_t, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F) {
    return spgemm_copy_template<int32_t, double>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
    return spgemm_copy_template<int32_t, cuFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
    return spgemm_copy_template<int32_t, cuDoubleComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
      computeType == ALPHA_R_16F) {
    return spgemm_copy_template<int32_t, half>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16F && matC->data_type == ALPHA_C_16F) {
    return spgemm_copy_template<int32_t, half2>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
#if (CUDA_ARCH >= 80)
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16BF && matC->data_type == ALPHA_R_16BF) {
    return spgemm_copy_template<int32_t, nv_bfloat16>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16BF && matC->data_type == ALPHA_C_16BF) {
    return spgemm_copy_template<int32_t, nv_bfloat162>(
      handle, opA, opB, alpha, matA, matB, beta, matC);
  }
#endif
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
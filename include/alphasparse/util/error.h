#pragma once

/**
 * @brief header for error detection utils
 */ 

#include "../types.h"

#define alpha_call_exit(func, message)\
  do{\
    alphasparseStatus_t status = func;\
    if (status != ALPHA_SPARSE_STATUS_SUCCESS)\
    {\
      printf("%s\n", message);\
      switch (status)\
      {\
      case ALPHA_SPARSE_STATUS_NOT_INITIALIZED:\
        printf("status : not initialized!!!\n");\
        break;\
      case ALPHA_SPARSE_STATUS_ALLOC_FAILED:\
        printf("status : alloc failed!!!\n");\
        break;\
      case ALPHA_SPARSE_STATUS_INVALID_VALUE:\
        printf("status : invalid value!!!\n");\
        break;\
      case ALPHA_SPARSE_STATUS_EXECUTION_FAILED:\
        printf("status : execution failed!!!\n");\
        break;\
      case ALPHA_SPARSE_STATUS_INTERNAL_ERROR:\
        printf("status : internal error!!!\n");\
        break;\
      case ALPHA_SPARSE_STATUS_NOT_SUPPORTED:\
        printf("status : not supported!!!\n");\
        break;\
      case ALPHA_SPARSE_STATUS_INVALID_POINTER:\
        printf("status : invalid pointer!!!\n");\
        break;\
      case ALPHA_SPARSE_STATUS_INVALID_HANDLE:\
        printf("status : invalid handle!!!\n");\
        break;\
      default:\
        printf("status : status invalid!!!\n");\
        break;\
      }\
      fflush(0);\
      exit(-1);\
    }\
  }while (0);


#define mkl_call_exit(func, message)\
  do{\
    int status = func;\
    if (status != 0)\
    {\
      printf("%s\n", message);\
      switch (status)\
      {\
      case 1:\
        printf("status : not initialized!!!\n");\
        break;\
      case 2:\
        printf("status : alloc failed!!!\n");\
        break;\
      case 3:\
        printf("status : invalid value!!!\n");\
        break;\
      case 4:\
        printf("status : execution failed!!!\n");\
        break;\
      case 5:\
        printf("status : internal error!!!\n");\
        break;\
      case 6:\
        printf("status : not supported!!!\n");\
        break;\
      default:\
        printf("status : status invalid!!!\n");\
        break;\
      }\
      fflush(0);\
      exit(-1);\
    }\
  }while (0);

#define roc_call_exit(func, message)\
  do{\
    int status = func;\
    if (status != 0)\
    {\
      printf("%s\n", message);\
      switch (status)\
      {\
      case 1:\
        printf("status : handle not initialized, invalid or null!!!\n");\
        break;\
      case 2:\
        printf("status : function is not implemented!!!\n");\
        break;\
      case 3:\
        printf("status : invalid pointer parameter!!!\n");\
        break;\
      case 4:\
        printf("status : invalid size parameter!!!\n");\
        break;\
      case 5:\
        printf("status : failed memory allocation, copy, dealloc!!!\n");\
        break;\
      case 6:\
        printf("status : other internal library failure!!!\n");\
        break;\
      case 7:\
        printf("status : invalid value parameter!!!\n");\
        break;\
      case 8:\
        printf("status : device arch is not supported!!!\n");\
        break;\
      case 9:\
        printf("status : encountered zero pivot!!!\n");\
        break;\
      default:\
        printf("status : status invalid!!!\n");\
        break;\
      }\
      fflush(0);\
      exit(-1);\
    }\
  }while (0);

#define THROW_IF_CUDA_ERROR(INPUT_STATUS_FOR_CHECK)\
  {\
      cudaError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;\
      if(TMP_STATUS_FOR_CHECK != cudaSuccess)\
      {\
          return get_alphasparse_status_for_cuda_status(TMP_STATUS_FOR_CHECK);\
      }\
  }

#define PRINT_IF_CUDA_ERROR(INPUT_STATUS_FOR_CHECK)\
    {\
        cudaError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;\
        if(TMP_STATUS_FOR_CHECK != cudaSuccess)\
        { \
            fprintf(stderr,\
                    "cuda error code: %d at %s:%d\n",\
                    TMP_STATUS_FOR_CHECK,\
                    __FILE__,\
                    __LINE__);\
        }\
    }

#define RETURN_IF_ALPHA_SPARSE_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        alphasparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != ALPHA_SPARSE_STATUS_SUCCESS)            \
        {                                                               \
            return TMP_STATUS_FOR_CHECK;                                \
        }                                                               \
    }

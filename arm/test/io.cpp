/**
 * @brief implement for file read and write utils
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "include/io.h"

#include <alphasparse.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

FILE *alpha_open(const char *filename, const char *modes) {
  FILE *file = fopen(filename, modes);
  if (file == NULL) {
    fprintf(stderr, "file does not exist!!!\n");
    exit(-1);
  }
  return file;
}

void alpha_close(FILE *stream) { fclose(stream); }
typedef enum {
  REAL = 0,    /*floating*/
  COMPLEX = 1, /*pairwise floating */
  INTEGER = 2, /*integer but defaulted to REAL*/
  PATTERN = 3,  /* binary */
  NULL_DATA_TYPE = 4
} datatype_t;
typedef enum {
  GENERAL = 0,
  HERMITIAN = 1,
  SYMMETRIC = 2,
  SKEW_SYMMETRIC = 3,
  NULL_MATRIX_TYPE = 4
} matrix_type_t;

void result_write(const char *path, const size_t ele_num, size_t ele_size,
                  const void *data) {
  FILE *ans = fopen(path, "w");
  if (ans == NULL) {
    printf("ans file open error!!!\n");
    exit(-1);
  }
  fwrite(&ele_num, sizeof(size_t), 1, ans);
  fwrite(data, ele_size, ele_num, ans);
  fclose(ans);
}

static int get_file_type(const char *file) {
  char buffer[BUFFER_SIZE];
  size_t file_size = strlen(file);
  if (!strcmp(file, "")) {
    fprintf(stderr, "filename is empty\n");
    exit(-1);
  }
  if (file_size > BUFFER_SIZE) {
    fprintf(stderr, "BUFFER SIZE TOO SMALL\n");
    exit(-1);
  }
  memcpy(buffer, file, file_size + 1);
  char *p, *mtx_name = NULL;
  p = strtok(buffer, "/");
  while (p) {
    mtx_name = p;
    p = strtok(NULL, "/");
  }
  int len_mtx = 0;
  p = strtok(mtx_name, ".");
  p = strtok(NULL, " ");
  if (strcmp(p, "mtx") == 0) {
    // printf("input matrix file is mtx type\n");
    return USE_MTX;
  } else if (strcmp(p, "bin") == 0) {
    // printf("input matrix file is binary type\n");
    return USE_BIN;
  } else {
    fprintf(stderr, "failed to inspect file type, use binary default\n");
    return USE_BIN;
  }
  return -1;
}

// return the file pointer after parse the matrix header
static FILE *parse_mtx_header(const char *file, ALPHA_INT64 *m_p,
                              ALPHA_INT64 *n_p, ALPHA_INT64 *lines,
                              datatype_t *data_type, matrix_type_t *matrix_type,
                              ALPHA_INT *is_sym) {
  FILE *fp = alpha_open(file, "r");
  char buffer[BUFFER_SIZE];
  char *data_type_str;
  const char REA[] = "real";
  const char CMP[] = "complex";
  const char INT[] = "integer";
  const char PAT[] = "pattern";

  char *matrix_type_str;
  const char SYM[] = "symmetric";
  const char GEN[] = "general";
  const char HER[] = "Hermitian";
  const char SKS[] = "skew-symmetric";

  int firstLine = 1;
  *is_sym = 0;
  char *token = NULL;
  while (fgets(buffer, BUFFER_SIZE, fp)) {
    // %%MatrixMarket matrix coordinate <datatype> <matrix_type>
    if (firstLine) {
      if (buffer[0] == '%') {
        token = strtok(buffer, " \n");  //%%MatrixMarket
        if (!token) {
          fprintf(stderr, "mtx file error");
        }
        token = strtok(NULL, " \n");  // matrix
        if (!token) {
          fprintf(stderr, "mtx file error");
        }
        token = strtok(NULL, " \n");  // coordinate OR array
        if (!token) {
          fprintf(stderr, "mtx file error");
        }
        data_type_str = strtok(NULL, " \n");    //<datatype>
        matrix_type_str = strtok(NULL, " \n");  //<matrix_type>

        if (!strcmp(data_type_str, REA))
          *data_type = REAL;
        else if (!strcmp(data_type_str, CMP))
          *data_type = COMPLEX;
        else if (!strcmp(data_type_str, INT))
          *data_type = INTEGER;
        else if (!strcmp(data_type_str, PAT))
          *data_type = PATTERN;
        else {
          *data_type = REAL;
        }
        if (*data_type == COMPLEX) {
          *data_type = REAL;
        }
        if (!strcmp(matrix_type_str, SYM))
          *matrix_type = SYMMETRIC;
        else if (!strcmp(matrix_type_str, GEN))
          *matrix_type = GENERAL;
        else if (!strcmp(matrix_type_str, HER))
          *matrix_type = HERMITIAN;
        else if (!strcmp(matrix_type_str, SKS))
          *matrix_type = SKEW_SYMMETRIC;
        else {
          *matrix_type = GENERAL;
        }
        if (*matrix_type != GENERAL) {
          *is_sym = 1;
        }
        // printf("%s,%s----------%d,%d\n",data_type_str,matrix_type_str,data_type,matrix_type);
      }
      firstLine = 0;
    }
    if (buffer[0] != '%') break;
  }
  sscanf(buffer, "%ld %ld %ld\n", m_p, n_p, lines);
  // printf("*mm *np *nnz %ld,%ld,%ld\n",*m_p,*n_p,*lines);
  return fp;
}
static void alpha_read_coo_s_pad_mtx(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index, float **values) {
  char buffer[BUFFER_SIZE];
  ALPHA_INT64 m = 0, n = 0, nnz = 0, real_nnz = 0, double_nnz = 0;
  datatype_t data_type = NULL_DATA_TYPE;
  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  int issym = 0;
  FILE *fp =
      parse_mtx_header(file, &m, &n, &nnz, &data_type, &matrix_type, &issym);
  char *token;
  real_nnz = 0;
  double_nnz = nnz << 1;
  if (pad) {
    *m_p = (ALPHA_INT)((m + pad - 1) / pad) * pad;
    *n_p = (ALPHA_INT)((n + pad - 1) / pad) * pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }

  ALPHA_INT *fake_row_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  ALPHA_INT *fake_col_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  float *fake_values = (float *)alpha_malloc((uint64_t)double_nnz * sizeof(float));
  for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++) {
    ALPHA_INT64 row, col;
    float val = 1.f;
    fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val = (float)atof(token);
    }
    fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
    fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
      fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
      fake_values[real_nnz] = val;
    }
  }
  // printf("file %s, m %ld, n %ld,real_nnz is %ld,data_type %d, matrix_type %d,
  // issym %d\n",
  //      file,
  //      m,
  //      n,
  //      real_nnz,
  //      data_type,
  //      matrix_type,
  //      issym);
  *row_index = (ALPHA_INT *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *col_index = (ALPHA_INT *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *values = (float *)alpha_malloc((uint64_t)real_nnz * sizeof(float));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(ALPHA_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, (uint64_t)sizeof(ALPHA_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(float) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

static void alpha_read_coo_d_pad_mtx(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index, double **values) {
  char buffer[BUFFER_SIZE];
  ALPHA_INT64 m = 0, n = 0, nnz = 0, real_nnz = 0, double_nnz = 0;
  datatype_t data_type = NULL_DATA_TYPE;
  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  int issym = 0;
  FILE *fp =
      parse_mtx_header(file, &m, &n, &nnz, &data_type, &matrix_type, &issym);
  char *token;
  real_nnz = 0;
  double_nnz = nnz << 1;
  if (pad) {
    *m_p = (ALPHA_INT)((m + pad - 1) / pad) * pad;
    *n_p = (ALPHA_INT)((n + pad - 1) / pad) * pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }

  ALPHA_INT *fake_row_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  ALPHA_INT *fake_col_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  double *fake_values = (double *)alpha_malloc((uint64_t)double_nnz * sizeof(double));
  for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++) {
    ALPHA_INT64 row, col;
    double val = 1.f;
    fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val = (double)atof(token);
    }
    fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
    fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
      fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
      fake_values[real_nnz] = val;
    }
  }
  *row_index = (ALPHA_INT *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *col_index = (ALPHA_INT *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *values = (double *)alpha_malloc((uint64_t)real_nnz * sizeof(double));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(ALPHA_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, (uint64_t)sizeof(ALPHA_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(double) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

static void alpha_read_coo_c_pad_mtx(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index,
                                     ALPHA_Complex8 **values) {
  char buffer[BUFFER_SIZE];
  ALPHA_INT64 m = 0, n = 0, nnz = 0, real_nnz = 0, double_nnz = 0;
  datatype_t data_type = NULL_DATA_TYPE;
  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  int issym = 0;
  FILE *fp =
      parse_mtx_header(file, &m, &n, &nnz, &data_type, &matrix_type, &issym);
  char *token;
  real_nnz = 0;
  double_nnz = nnz << 1;
  if (pad) {
    *m_p = (ALPHA_INT)((m + pad - 1) / pad) * pad;
    *n_p = (ALPHA_INT)((n + pad - 1) / pad) * pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }

  ALPHA_INT *fake_row_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  ALPHA_INT *fake_col_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  ALPHA_Complex8 *fake_values =
      (ALPHA_Complex8 *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_Complex8));
  for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++) {
    ALPHA_INT64 row, col;
    ALPHA_Complex8 val = {1.f, .0f};
    fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val.real = (float)atof(token);
    }
    if (data_type == COMPLEX) {
      token = strtok(NULL, " ");
      if (token != NULL) {
        val.imag = (float)atof(token);
      }
    } else {
      val.imag = val.real;
    }
    fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
    fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
      fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
      fake_values[real_nnz] = val;
      if (matrix_type == HERMITIAN) {
        fake_values[real_nnz].imag = -fake_values[real_nnz].imag;
      }
    }
  }
  *row_index = (ALPHA_INT *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *col_index =(ALPHA_INT *) alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *values = (ALPHA_Complex8 *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_Complex8));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(ALPHA_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, sizeof(ALPHA_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(ALPHA_Complex8) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

static void alpha_read_coo_z_pad_mtx(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index,
                                     ALPHA_Complex16 **values) {
  char buffer[BUFFER_SIZE];
  ALPHA_INT64 m = 0, n = 0, nnz = 0, real_nnz = 0, double_nnz = 0;
  datatype_t data_type = NULL_DATA_TYPE;
  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  int issym = 0;
  FILE *fp =
      parse_mtx_header(file, &m, &n, &nnz, &data_type, &matrix_type, &issym);
  char *token;
  real_nnz = 0;
  double_nnz = nnz << 1;
  if (pad) {
    *m_p = (ALPHA_INT)((m + pad - 1) / pad) * pad;
    *n_p = (ALPHA_INT)((n + pad - 1) / pad) * pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }

  ALPHA_INT *fake_row_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  ALPHA_INT *fake_col_index =
      (ALPHA_INT *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_INT));
  ALPHA_Complex16 *fake_values =
      (ALPHA_Complex16 *)alpha_malloc((uint64_t)double_nnz * sizeof(ALPHA_Complex16));
  for (ALPHA_INT64 i = 0; i < nnz; i++, real_nnz++) {
    ALPHA_INT64 row, col;
    ALPHA_Complex16 val = {1.f, .0f};
    fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val.real = (double)atof(token);
    }
    if (data_type == COMPLEX) {
      token = strtok(NULL, " ");
      if (token != NULL) {
        val.imag = (double)atof(token);
      }
    } else {
      val.imag = val.real;
    }
    fake_row_index[real_nnz] = (ALPHA_INT)row - 1;
    fake_col_index[real_nnz] = (ALPHA_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (ALPHA_INT)col - 1;
      fake_col_index[real_nnz] = (ALPHA_INT)row - 1;
      fake_values[real_nnz] = val;
      if (matrix_type == HERMITIAN) {
        fake_values[real_nnz].imag = -fake_values[real_nnz].imag;
      }
    }
  }
  *row_index = (ALPHA_INT *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *col_index = (ALPHA_INT *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_INT));
  *values = (ALPHA_Complex16 *)alpha_malloc((uint64_t)real_nnz * sizeof(ALPHA_Complex16));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(ALPHA_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, (uint64_t)sizeof(ALPHA_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(ALPHA_Complex16) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

// typedef struct {
// {
//   int64 rows; //总行数
//   int64 cols; //总列数
//   int64 nnzs; //有效nnz个数
//   int64 real_nnz; //总共的nnz个数
//   int64 field_per_nnz; //complex:2, integer/float:1, pattern 0, real:1
//   int64 num_type; //float:0, integer 1;
//   int64 mtx_sym;  //general:0, sym:1, Hermitian:2
//   int64 reserved;
// }
// } bin_header;

// typedef struct
// {
//   int32 row;
//   int32 col;
//   num_type val[filed_per_nnz];
// };
static void alpha_read_coo_s_pad_bin(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT m_pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index, float **values) {
  bin_header header;
  size_t size_header = sizeof(header);
  int fd = open(file, O_RDWR);
  long start = lseek(fd, 0, SEEK_SET);
  lseek(fd, 0, SEEK_END);
  long file_size = lseek(fd, 0, SEEK_END) - start;
  // printf("file size is %ld bytes, %lf GB\n", file_size, file_size * 1.f / (1l
  // << 30));
  char *const file_content =
      (char *)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  // printf("mmaped success\n");
  int64_t file_offset = 0;
  memcpy(&header, file_content, size_header);
  // printf("file_content copied\n");
  file_offset += size_header;
  ALPHA_INT64 m, n, real_nnz;
  // printf(
  //     "matrix has %ld rows, %ld cols, lines %ld,real_nnz %ld, field_per_nnz
  //     %ld, num_type %ld, " "mtx_sym %ld\n", header.rows, header.cols,
  //     header.nnzs, header.real_nnz, header.field_per_nnz, header.num_type,
  //     header.mtx_sym);
  m = (ALPHA_INT64)header.rows;
  n = (ALPHA_INT64)header.cols;
  real_nnz = (ALPHA_INT64)header.real_nnz;
  if (m_pad > 0) {
    *m_p = (ALPHA_INT)((m + m_pad - 1) / m_pad) * m_pad;
    *n_p = (ALPHA_INT)((n + m_pad - 1) / m_pad) * m_pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }
  *nnz_p = (ALPHA_INT)real_nnz;
  ALPHA_INT *_row_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  ALPHA_INT *_col_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  float *_values = (float *)alpha_malloc(sizeof(float) * real_nnz);

  *row_index = _row_index;
  *col_index = _col_index;
  *values = _values;

  ALPHA_INT coord_i = 0;
  ALPHA_INT nnz_i = 0;
  char *file_stream = file_content + size_header;
  size_t bytes_per_nnz =
      sizeof(int64_t) * header.field_per_nnz + sizeof(int32_t) * 2;
  // printf("field_per_nnz %ld, bytes_per_nnz %ld\n", header.field_per_nnz,
  // bytes_per_nnz);
  if (header.field_per_nnz == 0) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double val = 1.f;
      // printf("(%d,%d) %.18f\n", row, col, val);
      _values[nnz_i++] = (float)val;
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i++] = (float)val;
      }
    }
  } else if (header.field_per_nnz == 1) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18f\n", row, col, *val);
      file_stream += sizeof(double);
      _values[nnz_i++] = (float)(*val);
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i++] = (float)(*val);
      }
    }
  } else if (header.field_per_nnz == 2) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18lf,%.18f\n", row, col, *val,*(val + 1));
      file_stream += sizeof(double) * 2;
      _values[nnz_i] = (float)(*val);
      // skip the second field
      // _values[nnz_i] = (float)(*(val + 1));
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i++] = (float)(*val);
      }
    }
  } else {
    fprintf(stderr, "unsupported binary");
    exit(-1);
  }
}

static void alpha_read_coo_d_pad_bin(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT m_pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index, double **values) {
  bin_header header;
  size_t size_header = sizeof(header);
  int fd = open(file, O_RDWR);
  long start = lseek(fd, 0, SEEK_SET);
  lseek(fd, 0, SEEK_END);
  long file_size = lseek(fd, 0, SEEK_END) - start;
  // printf("file size in byte is %ld fd is %d\n", file_size, fd);
  char *const file_content =
      (char *)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  int64_t file_offset = 0;
  memcpy(&header, file_content, size_header);
  file_offset += size_header;
  ALPHA_INT64 m, n, real_nnz;
  // printf("matrix has %ld rows, %ld cols, lines %ld,real_nnz %ld,
  // field_per_nnz %ld, num_type %ld, mtx_sym %ld\n",
  //        header.rows,
  //        header.cols,
  //        header.nnzs,
  //        header.real_nnz,
  //        header.field_per_nnz,
  //        header.num_type,
  //        header.mtx_sym);
  m = (ALPHA_INT64)header.rows;
  n = (ALPHA_INT64)header.cols;
  real_nnz = (ALPHA_INT64)header.real_nnz;
  if (m_pad > 0) {
    *m_p = (ALPHA_INT)((m + m_pad - 1) / m_pad) * m_pad;
    *n_p = (ALPHA_INT)((n + m_pad - 1) / m_pad) * m_pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }
  *nnz_p = (ALPHA_INT)real_nnz;
  ALPHA_INT *_row_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  ALPHA_INT *_col_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  double *_values = (double *)alpha_malloc(sizeof(double) * real_nnz);

  *row_index = _row_index;
  *col_index = _col_index;
  *values = _values;

  ALPHA_INT coord_i = 0;
  ALPHA_INT nnz_i = 0;
  // printf("sizeof header %ld\n",size_header);
  char *file_stream = file_content + size_header;
  size_t bytes_per_nnz =
      sizeof(int64_t) * header.field_per_nnz + sizeof(int32_t) * 2;

  if (header.field_per_nnz == 0) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double val = 1.f;
      // printf("(%d,%d) %.18f\n", row, col, val);
      _values[nnz_i++] = val;
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i++] = val;
      }
    }
  } else if (header.field_per_nnz == 1) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18f\n", row, col, *val);
      file_stream += sizeof(double);
      _values[nnz_i++] = (*val);
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i++] = (*val);
      }
    }
  } else if (header.field_per_nnz == 2) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18lf,%.18f\n", row, col, *val,*(val + 1));
      file_stream += sizeof(double) * 2;
      _values[nnz_i] = (*val);
      // skip the second field
      // _values[nnz_i] = (*(val + 1));
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i++] = (*val);
      }
    }
  } else {
    fprintf(stderr, "unsupported binary");
    exit(-1);
  }
}

static void alpha_read_coo_c_pad_bin(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT m_pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index,
                                     ALPHA_Complex8 **values) {
  bin_header header;
  size_t size_header = sizeof(header);
  int fd = open(file, O_RDWR);
  long start = lseek(fd, 0, SEEK_SET);
  lseek(fd, 0, SEEK_END);
  long file_size = lseek(fd, 0, SEEK_END) - start;
  char *const file_content =
      (char *)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  int64_t file_offset = 0;
  memcpy(&header, file_content, size_header);
  file_offset += size_header;
  ALPHA_INT64 m, n, real_nnz;
  // printf("matrix has %ld rows, %ld cols, lines %ld,real_nnz %ld,
  // field_per_nnz %ld, num_type %ld, mtx_sym %ld\n",
  //        header.rows,
  //        header.cols,
  //        header.nnzs,
  //        header.real_nnz,
  //        header.field_per_nnz,
  //        header.num_type,
  //        header.mtx_sym);
  m = (ALPHA_INT64)header.rows;
  n = (ALPHA_INT64)header.cols;
  real_nnz = (ALPHA_INT64)header.real_nnz;
  if (m_pad > 0) {
    *m_p = (ALPHA_INT)((m + m_pad - 1) / m_pad) * m_pad;
    *n_p = (ALPHA_INT)((n + m_pad - 1) / m_pad) * m_pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }
  *nnz_p = (ALPHA_INT)real_nnz;
  ALPHA_INT *_row_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  ALPHA_INT *_col_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  ALPHA_Complex8 *_values =
      (ALPHA_Complex8 *)alpha_malloc(sizeof(ALPHA_Complex8) * real_nnz);

  *row_index = _row_index;
  *col_index = _col_index;
  *values = _values;

  ALPHA_INT coord_i = 0;
  ALPHA_INT nnz_i = 0;
  char *file_stream = file_content + size_header;
  size_t bytes_per_nnz =
      sizeof(int64_t) * header.field_per_nnz + sizeof(int32_t) * 2;

  if (header.field_per_nnz == 0) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      float val = 1.f;
      // printf("(%d,%d) %.18f\n", row, col, val);
      _values[nnz_i].real = val;
      _values[nnz_i++].imag = val;
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i].real = val;
        _values[nnz_i++].imag = val;
      }
    }
  } else if (header.field_per_nnz == 1) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18f\n", row, col, *val);
      file_stream += sizeof(double);
      _values[nnz_i].real = (float)(*val);
      _values[nnz_i++].imag = (float)(*val);
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i].real = (float)(*val);
        _values[nnz_i++].imag = (float)(*val);
      }
    }
  } else if (header.field_per_nnz == 2) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18lf,%.18f\n", row, col, *val,*(val + 1));
      file_stream += sizeof(double) * 2;
      _values[nnz_i].real = (float)(*val);
      _values[nnz_i++].imag = (float)(*(val + 1));
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i].real = (float)(*val);
        _values[nnz_i++].imag = (float)(*(val + 1));
      }
    }
  } else {
    fprintf(stderr, "unsupported binary");
    exit(-1);
  }
}

static void alpha_read_coo_z_pad_bin(const char *file, ALPHA_INT *m_p,
                                     ALPHA_INT *n_p, ALPHA_INT m_pad,
                                     ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                                     ALPHA_INT **col_index,
                                     ALPHA_Complex16 **values) {
  bin_header header;
  size_t size_header = sizeof(header);
  int fd = open(file, O_RDWR);
  long start = lseek(fd, 0, SEEK_SET);
  lseek(fd, 0, SEEK_END);
  long file_size = lseek(fd, 0, SEEK_END) - start;
  char *const file_content =
      (char *)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
  int64_t file_offset = 0;
  memcpy(&header, file_content, size_header);
  file_offset += size_header;
  ALPHA_INT64 m, n, real_nnz;
  // printf("matrix has %ld rows, %ld cols, lines %ld,real_nnz %ld,
  // field_per_nnz %ld, num_type %ld, mtx_sym %ld\n",
  //        header.rows,
  //        header.cols,
  //        header.nnzs,
  //        header.real_nnz,
  //        header.field_per_nnz,
  //        header.num_type,
  //        header.mtx_sym);
  m = (ALPHA_INT64)header.rows;
  n = (ALPHA_INT64)header.cols;
  real_nnz = (ALPHA_INT64)header.real_nnz;
  if (m_pad > 0) {
    *m_p = (ALPHA_INT)((m + m_pad - 1) / m_pad) * m_pad;
    *n_p = (ALPHA_INT)((n + m_pad - 1) / m_pad) * m_pad;
  } else {
    *m_p = (ALPHA_INT)m;
    *n_p = (ALPHA_INT)n;
  }
  *nnz_p = (ALPHA_INT)real_nnz;
  ALPHA_INT *_row_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  ALPHA_INT *_col_index =
      (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * real_nnz);
  ALPHA_Complex16 *_values =
      (ALPHA_Complex16 *)alpha_malloc(sizeof(ALPHA_Complex16) * real_nnz);

  *row_index = _row_index;
  *col_index = _col_index;
  *values = _values;

  ALPHA_INT coord_i = 0;
  ALPHA_INT nnz_i = 0;
  char *file_stream = file_content + size_header;
  size_t bytes_per_nnz =
      sizeof(int64_t) * header.field_per_nnz + sizeof(int32_t) * 2;

  if (header.field_per_nnz == 0) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double val = 1.f;
      // printf("(%d,%d) %.18f\n", row, col, val);
      _values[nnz_i].real = val;
      _values[nnz_i++].imag = val;
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i].real = val;
        _values[nnz_i++].imag = val;
      }
    }
  } else if (header.field_per_nnz == 1) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18f\n", row, col, *val);
      file_stream += sizeof(double);
      _values[nnz_i].real = (*val);
      _values[nnz_i++].imag = (*val);
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i].real = (*val);
        _values[nnz_i++].imag = (*val);
      }
    }
  } else if (header.field_per_nnz == 2) {
    for (; file_offset < file_size; file_offset += bytes_per_nnz) {
      int *coo = (int *)file_stream;
      int32_t row = coo[0] - 1;
      int32_t col = coo[1] - 1;
      _row_index[coord_i] = row;
      _col_index[coord_i++] = col;
      file_stream += sizeof(int32_t) * 2;
      double *val = (double *)file_stream;
      // printf("(%d,%d) %.18lf,%.18f\n", row, col, *val,*(val + 1));
      file_stream += sizeof(double) * 2;
      _values[nnz_i].real = (*val);
      _values[nnz_i++].imag = (*(val + 1));
      if (header.mtx_sym != 0 && row != col) {
        _row_index[coord_i] = col;
        _col_index[coord_i++] = row;
        _values[nnz_i].real = (*val);
        _values[nnz_i++].imag = (*(val + 1));
      }
    }
  } else {
    fprintf(stderr, "unsupported binary");
    exit(-1);
  }
}

void alpha_read_coo_pad(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                        ALPHA_INT pad, ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                        ALPHA_INT **col_index, float **values) {
  int file_type = get_file_type(file);
  if (file_type == USE_MTX) {
    alpha_read_coo_s_pad_mtx(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  } else {
    alpha_read_coo_s_pad_bin(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  }
}
void alpha_read_coo_pad_d(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                          ALPHA_INT pad, ALPHA_INT *nnz_p,
                          ALPHA_INT **row_index, ALPHA_INT **col_index,
                          double **values) {
  int file_type = get_file_type(file);
  if (file_type == USE_MTX) {
    alpha_read_coo_d_pad_mtx(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  } else {
    alpha_read_coo_d_pad_bin(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  }
}
void alpha_read_coo_pad_c(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                          ALPHA_INT pad, ALPHA_INT *nnz_p,
                          ALPHA_INT **row_index, ALPHA_INT **col_index,
                          ALPHA_Complex8 **values) {
  int file_type = get_file_type(file);
  if (file_type == USE_MTX) {
    alpha_read_coo_c_pad_mtx(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  } else {
    alpha_read_coo_c_pad_bin(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  }
}
void alpha_read_coo_pad_z(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                          ALPHA_INT pad, ALPHA_INT *nnz_p,
                          ALPHA_INT **row_index, ALPHA_INT **col_index,
                          ALPHA_Complex16 **values) {
  int file_type = get_file_type(file);
  if (file_type == USE_MTX) {
    alpha_read_coo_z_pad_mtx(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  } else {
    alpha_read_coo_z_pad_bin(file, m_p, n_p, pad, nnz_p, row_index, col_index,
                             values);
  }
}
void alpha_read_coo(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                    ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                    ALPHA_INT **col_index, float **values) {
  alpha_read_coo_pad(file, m_p, n_p, 0, nnz_p, row_index, col_index, values);
}
void alpha_read_coo_d(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                      ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                      ALPHA_INT **col_index, double **values) {
  alpha_read_coo_pad_d(file, m_p, n_p, 0, nnz_p, row_index, col_index, values);
}
void alpha_read_coo_c(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                      ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                      ALPHA_INT **col_index, ALPHA_Complex8 **values) {
  alpha_read_coo_pad_c(file, m_p, n_p, 0, nnz_p, row_index, col_index, values);
}
void alpha_read_coo_z(const char *file, ALPHA_INT *m_p, ALPHA_INT *n_p,
                      ALPHA_INT *nnz_p, ALPHA_INT **row_index,
                      ALPHA_INT **col_index, ALPHA_Complex16 **values) {
  alpha_read_coo_pad_z(file, m_p, n_p, 0, nnz_p, row_index, col_index, values);
}

// TODO : 删除从file 导出特征的接口, 原因是和文件IO打交道太慢了

// void alpha_dump_feature_file(const char *file, double *feature)
// {
//     ALPHA_INT rows, cols, nnz, min_nnz_row, max_nnz_row, diags;
//     double sparsity, avr_nnz_row, var_nnz_row;
//     double dia_padding_ratio, ell_padding_ratio, diag_ratio;
//     alpha_dump_nnz_feature(file, &rows, &cols, &nnz, &sparsity, &avr_nnz_row,
//     &min_nnz_row, &max_nnz_row, &var_nnz_row, &diags, &diag_ratio,
//     &dia_padding_ratio, &ell_padding_ratio); feature[0] = rows; feature[1] =
//     cols; feature[2] = nnz; feature[3] = sparsity; feature[4] = avr_nnz_row;
//     feature[5] = min_nnz_row;
//     feature[6] = max_nnz_row;
//     feature[7] = var_nnz_row;
//     feature[8] = diags;
//     feature[9] = diag_ratio;
//     feature[10] = dia_padding_ratio;
//     feature[11] = ell_padding_ratio;
//     // feature[12] = avr_nnz_col;
//     // feature[13] = min_nnz_col;
//     // feature[14] = max_nnz_col;
//     // feature[15] = var_nnz_col;
// }
#ifdef __MKL__

void mkl_read_coo(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p,
                  MKL_INT **row_index, MKL_INT **col_index, float **values) {
  FILE *fp = alpha_open(file, "r");
  char buffer[BUFFER_SIZE];
  char *token;
  char *data_type_str;
  datatype_t data_type = NULL_DATA_TYPE;
  const char REA[] = "real";
  const char CMP[] = "complex";
  const char INT[] = "integer";
  const char PAT[] = "pattern";

  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  char *matrix_type_str;
  const char SYM[] = "symmetric";
  const char GEN[] = "general";
  const char HER[] = "Hermitian";
  const char SKS[] = "skew-symmetric";

  int issym = 0;
  int firstLine = 1;
  while (fgets(buffer, BUFFER_SIZE, fp)) {
    // %%MatrixMarket matrix coordinate <datatype> <matrix_type>
    if (firstLine) {
      if (buffer[0] == '%') {
        token = strtok(buffer, " \n");          //%%MatrixMarket
        token = strtok(NULL, " \n");            // matrix
        token = strtok(NULL, " \n");            // coordinate OR array
        data_type_str = strtok(NULL, " \n");    //<datatype>
        matrix_type_str = strtok(NULL, " \n");  //<matrix_type>

        if (!strcmp(data_type_str, REA))
          data_type = REAL;
        else if (!strcmp(data_type_str, CMP))
          data_type = COMPLEX;
        else if (!strcmp(data_type_str, INT))
          data_type = INTEGER;
        else if (!strcmp(data_type_str, PAT))
          data_type = PATTERN;
        else {
          data_type = REAL;
        }
        if (data_type == COMPLEX) {
          data_type = REAL;
        }
        if (!strcmp(matrix_type_str, SYM))
          matrix_type = SYMMETRIC;
        else if (!strcmp(matrix_type_str, GEN))
          matrix_type = GENERAL;
        else if (!strcmp(matrix_type_str, HER))
          matrix_type = HERMITIAN;
        else if (!strcmp(matrix_type_str, SKS))
          matrix_type = SKEW_SYMMETRIC;
        else {
          matrix_type = GENERAL;
        }
        if (matrix_type != GENERAL) {
          issym = 1;
        }
        // printf("%s,%s----------%d,%d\n",data_type_str,matrix_type_str,data_type,matrix_type);
      }
      firstLine = 0;
    }
    if (buffer[0] != '%') break;
  }
  MKL_INT64 m, n, nnz, real_nnz, double_nnz;
  sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
  real_nnz = 0;
  double_nnz = nnz << 1;
  *m_p = (MKL_INT)m;
  *n_p = (MKL_INT)n;
  MKL_INT *fake_row_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  MKL_INT *fake_col_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  float *fake_values = alpha_malloc((uint64_t)double_nnz * sizeof(float));
  for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++) {
    MKL_INT64 row, col;
    float val = 1.f;
    char *r = fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val = (float)atof(token);
    }
    fake_row_index[real_nnz] = (MKL_INT)row - 1;
    fake_col_index[real_nnz] = (MKL_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (MKL_INT)col - 1;
      fake_col_index[real_nnz] = (MKL_INT)row - 1;
      fake_values[real_nnz] = val;
    }
  }
  *row_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *col_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *values = alpha_malloc((uint64_t)real_nnz * sizeof(float));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(float) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

void mkl_read_coo_d(const char *file, MKL_INT *m_p, MKL_INT *n_p,
                    MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index,
                    double **values) {
  FILE *fp = alpha_open(file, "r");
  char buffer[BUFFER_SIZE];
  char *token;
  char *data_type_str;
  datatype_t data_type = NULL_DATA_TYPE;
  const char REA[] = "real";
  const char CMP[] = "complex";
  const char INT[] = "integer";
  const char PAT[] = "pattern";

  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  char *matrix_type_str;
  const char SYM[] = "symmetric";
  const char GEN[] = "general";
  const char HER[] = "Hermitian";
  const char SKS[] = "skew-symmetric";

  int issym = 0;
  int firstLine = 1;
  while (fgets(buffer, BUFFER_SIZE, fp)) {
    // %%MatrixMarket matrix coordinate <datatype> <matrix_type>
    if (firstLine) {
      if (buffer[0] == '%') {
        token = strtok(buffer, " \n");          //%%MatrixMarket
        token = strtok(NULL, " \n");            // matrix
        token = strtok(NULL, " \n");            // coordinate OR array
        data_type_str = strtok(NULL, " \n");    //<datatype>
        matrix_type_str = strtok(NULL, " \n");  //<matrix_type>

        if (!strcmp(data_type_str, REA))
          data_type = REAL;
        else if (!strcmp(data_type_str, CMP))
          data_type = COMPLEX;
        else if (!strcmp(data_type_str, INT))
          data_type = INTEGER;
        else if (!strcmp(data_type_str, PAT))
          data_type = PATTERN;
        else {
          data_type = REAL;
        }
        if (data_type == COMPLEX) {
          data_type = REAL;
        }
        if (!strcmp(matrix_type_str, SYM))
          matrix_type = SYMMETRIC;
        else if (!strcmp(matrix_type_str, GEN))
          matrix_type = GENERAL;
        else if (!strcmp(matrix_type_str, HER))
          matrix_type = HERMITIAN;
        else if (!strcmp(matrix_type_str, SKS))
          matrix_type = SKEW_SYMMETRIC;
        else {
          matrix_type = GENERAL;
        }
        if (matrix_type != GENERAL) {
          issym = 1;
        }
        // printf("%s,%s----------%d,%d\n",data_type_str,matrix_type_str,data_type,matrix_type);
      }
      firstLine = 0;
    }
    if (buffer[0] != '%') break;
  }
  MKL_INT64 m, n, nnz, real_nnz, double_nnz;
  sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
  real_nnz = 0;
  double_nnz = nnz << 1;
  *m_p = (MKL_INT)m;
  *n_p = (MKL_INT)n;
  MKL_INT *fake_row_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  MKL_INT *fake_col_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  double *fake_values = alpha_malloc((uint64_t)double_nnz * sizeof(double));
  for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++) {
    MKL_INT64 row, col;
    double val = 1.f;
    char *r = fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val = (double)atof(token);
    }
    fake_row_index[real_nnz] = (MKL_INT)row - 1;
    fake_col_index[real_nnz] = (MKL_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (MKL_INT)col - 1;
      fake_col_index[real_nnz] = (MKL_INT)row - 1;
      fake_values[real_nnz] = val;
    }
  }
  *row_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *col_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *values = alpha_malloc((uint64_t)real_nnz * sizeof(double));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(double) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

void mkl_read_coo_c(const char *file, MKL_INT *m_p, MKL_INT *n_p,
                    MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index,
                    MKL_Complex8 **values) {
  FILE *fp = alpha_open(file, "r");
  char buffer[BUFFER_SIZE];
  char *token;
  char *data_type_str;
  datatype_t data_type = NULL_DATA_TYPE;
  const char REA[] = "real";
  const char CMP[] = "complex";
  const char INT[] = "integer";
  const char PAT[] = "pattern";

  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  char *matrix_type_str;
  const char SYM[] = "symmetric";
  const char GEN[] = "general";
  const char HER[] = "Hermitian";
  const char SKS[] = "skew-symmetric";

  int issym = 0;
  int firstLine = 1;
  while (fgets(buffer, BUFFER_SIZE, fp)) {
    // %%MatrixMarket matrix coordinate <datatype> <matrix_type>
    if (firstLine) {
      if (buffer[0] == '%') {
        token = strtok(buffer, " \n");          //%%MatrixMarket
        token = strtok(NULL, " \n");            // matrix
        token = strtok(NULL, " \n");            // coordinate OR array
        data_type_str = strtok(NULL, " \n");    //<datatype>
        matrix_type_str = strtok(NULL, " \n");  //<matrix_type>

        if (!strcmp(data_type_str, REA))
          data_type = REAL;
        else if (!strcmp(data_type_str, CMP))
          data_type = COMPLEX;
        else if (!strcmp(data_type_str, INT))
          data_type = INTEGER;
        else if (!strcmp(data_type_str, PAT))
          data_type = PATTERN;
        else {
          data_type = REAL;
        }
        if (!strcmp(matrix_type_str, SYM))
          matrix_type = SYMMETRIC;
        else if (!strcmp(matrix_type_str, GEN))
          matrix_type = GENERAL;
        else if (!strcmp(matrix_type_str, HER))
          matrix_type = HERMITIAN;
        else if (!strcmp(matrix_type_str, SKS))
          matrix_type = SKEW_SYMMETRIC;
        else {
          matrix_type = GENERAL;
        }
        if (matrix_type != GENERAL) {
          issym = 1;
        }
        // printf("%s,%s----------%d,%d\n", data_type_str, matrix_type_str,
        // data_type, matrix_type);
      }
      firstLine = 0;
    }
    if (buffer[0] != '%') break;
  }
  MKL_INT64 m, n, nnz, real_nnz, double_nnz;
  sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
  real_nnz = 0;
  double_nnz = nnz << 1;
  *m_p = (MKL_INT)m;
  *n_p = (MKL_INT)n;
  MKL_INT *fake_row_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  MKL_INT *fake_col_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  MKL_Complex8 *fake_values =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_Complex8));
  for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++) {
    MKL_INT64 row, col;
    MKL_Complex8 val = {1.f, .0f};
    fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val.real = (float)atof(token);
    }
    if (data_type == COMPLEX) {
      token = strtok(NULL, " ");
      if (token != NULL) {
        val.imag = (float)atof(token);
      }
    }
    fake_row_index[real_nnz] = (MKL_INT)row - 1;
    fake_col_index[real_nnz] = (MKL_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (MKL_INT)col - 1;
      fake_col_index[real_nnz] = (MKL_INT)row - 1;
      fake_values[real_nnz] = val;
      if (matrix_type == HERMITIAN) {
        fake_values[real_nnz].imag = -fake_values[real_nnz].imag;
      }
    }
  }
  *row_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *col_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *values = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_Complex8));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(MKL_Complex8) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

void mkl_read_coo_z(const char *file, MKL_INT *m_p, MKL_INT *n_p,
                    MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index,
                    MKL_Complex16 **values) {
  FILE *fp = alpha_open(file, "r");
  char buffer[BUFFER_SIZE];
  char *token;
  char *data_type_str;
  datatype_t data_type = NULL_DATA_TYPE;
  const char REA[] = "real";
  const char CMP[] = "complex";
  const char INT[] = "integer";
  const char PAT[] = "pattern";

  matrix_type_t matrix_type = NULL_MATRIX_TYPE;
  char *matrix_type_str;
  const char SYM[] = "symmetric";
  const char GEN[] = "general";
  const char HER[] = "Hermitian";
  const char SKS[] = "skew-symmetric";

  int issym = 0;
  int firstLine = 1;
  while (fgets(buffer, BUFFER_SIZE, fp)) {
    // %%MatrixMarket matrix coordinate <datatype> <matrix_type>
    if (firstLine) {
      if (buffer[0] == '%') {
        token = strtok(buffer, " \n");          //%%MatrixMarket
        token = strtok(NULL, " \n");            // matrix
        token = strtok(NULL, " \n");            // coordinate OR array
        data_type_str = strtok(NULL, " \n");    //<datatype>
        matrix_type_str = strtok(NULL, " \n");  //<matrix_type>

        if (!strcmp(data_type_str, REA))
          data_type = REAL;
        else if (!strcmp(data_type_str, CMP))
          data_type = COMPLEX;
        else if (!strcmp(data_type_str, INT))
          data_type = INTEGER;
        else if (!strcmp(data_type_str, PAT))
          data_type = PATTERN;
        else {
          data_type = REAL;
        }
        if (!strcmp(matrix_type_str, SYM))
          matrix_type = SYMMETRIC;
        else if (!strcmp(matrix_type_str, GEN))
          matrix_type = GENERAL;
        else if (!strcmp(matrix_type_str, HER))
          matrix_type = HERMITIAN;
        else if (!strcmp(matrix_type_str, SKS))
          matrix_type = SKEW_SYMMETRIC;
        else {
          matrix_type = GENERAL;
        }
        if (matrix_type != GENERAL) {
          issym = 1;
        }
        // printf("%s,%s----------%d,%d\n", data_type_str, matrix_type_str,
        // data_type, matrix_type);
      }
      firstLine = 0;
    }
    if (buffer[0] != '%') break;
  }
  MKL_INT64 m, n, nnz, real_nnz, double_nnz;
  sscanf(buffer, "%lld %lld %lld\n", &m, &n, &nnz);
  real_nnz = 0;
  double_nnz = nnz << 1;
  *m_p = (MKL_INT)m;
  *n_p = (MKL_INT)n;
  MKL_INT *fake_row_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  MKL_INT *fake_col_index =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_INT));
  MKL_Complex16 *fake_values =
      alpha_malloc((uint64_t)double_nnz * sizeof(MKL_Complex16));
  for (MKL_INT64 i = 0; i < nnz; i++, real_nnz++) {
    MKL_INT64 row, col;
    MKL_Complex16 val = {1.f, .0f};
    fgets(buffer, BUFFER_SIZE, fp);
    token = strtok(buffer, " ");
    row = atol(token);
    token = strtok(NULL, " ");
    col = atol(token);
    token = strtok(NULL, " ");
    if (token != NULL) {
      val.real = (double)atof(token);
    }
    if (data_type == COMPLEX) {
      token = strtok(NULL, " ");
      if (token != NULL) {
        val.imag = (double)atof(token);
      }
    }
    fake_row_index[real_nnz] = (MKL_INT)row - 1;
    fake_col_index[real_nnz] = (MKL_INT)col - 1;
    fake_values[real_nnz] = val;
    if (row != col && issym) {
      real_nnz++;
      fake_row_index[real_nnz] = (MKL_INT)col - 1;
      fake_col_index[real_nnz] = (MKL_INT)row - 1;
      fake_values[real_nnz] = val;
      if (matrix_type == HERMITIAN) {
        fake_values[real_nnz].imag = -fake_values[real_nnz].imag;
      }
    }
  }
  *row_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *col_index = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_INT));
  *values = alpha_malloc((uint64_t)real_nnz * sizeof(MKL_Complex16));
  *nnz_p = real_nnz;
  memcpy(*row_index, fake_row_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*col_index, fake_col_index, (uint64_t)sizeof(MKL_INT) * real_nnz);
  memcpy(*values, fake_values, (uint64_t)sizeof(MKL_Complex16) * real_nnz);
  alpha_free(fake_row_index);
  alpha_free(fake_col_index);
  alpha_free(fake_values);
  alpha_close(fp);
}

#endif

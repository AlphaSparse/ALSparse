#include <alphasparse/types.h>

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPMAT_HYB spmat_hyb_s_t
#else
#define ALPHA_SPMAT_HYB spmat_hyb_d_t
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPMAT_HYB spmat_hyb_c_t
#else
#define ALPHA_SPMAT_HYB spmat_hyb_z_t
#endif
#endif

typedef struct {
  ALPHA_INT rows;       // number of rows (integer).
  ALPHA_INT cols;       // number of columns (integer).
  ALPHA_INT nnz;        // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;  // maximum number of non-zero elements per row of the ELL part (integer)
  float *ell_val;     // array of m times ell_width elements containing the ELL part data (floating
                      // point).
  ALPHA_INT *ell_col_ind;  // array of m times ell_width elements containing the ELL part column
                         // indices (integer).
  float *coo_val;        // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;  // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;  // array of nnz elements containing the COO part column indices (integer).

  float     *d_ell_val;
  ALPHA_INT *d_ell_col_ind;
  float     *d_coo_val;
  ALPHA_INT *d_coo_row_val;
  ALPHA_INT *d_coo_col_val;
} spmat_hyb_s_t;

typedef struct {
  ALPHA_INT rows;       // number of rows (integer).
  ALPHA_INT cols;       // number of columns (integer).
  ALPHA_INT nnz;        // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;  // maximum number of non-zero elements per row of the ELL part (integer)
  double *ell_val;    // array of m times ell_width elements containing the ELL part data (floating
                      // point).
  ALPHA_INT *ell_col_ind;  // array of m times ell_width elements containing the ELL part column
                         // indices (integer).
  double *coo_val;       // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;  // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;  // array of nnz elements containing the COO part column indices (integer).
  
  double    *d_ell_val;
  ALPHA_INT *d_ell_col_ind;
  double    *d_coo_val;
  ALPHA_INT *d_coo_row_val;
  ALPHA_INT *d_coo_col_val;
} spmat_hyb_d_t;

typedef struct {
  ALPHA_INT rows;           // number of rows (integer).
  ALPHA_INT cols;           // number of columns (integer).
  ALPHA_INT nnz;            // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;      // maximum number of non-zero elements per row of the ELL part (integer)
  ALPHA_Complex8 *ell_val;  // array of m times ell_width elements containing the ELL part data
                          // (floating point).
  ALPHA_INT *ell_col_ind;   // array of m times ell_width elements containing the ELL part column
                          // indices (integer).
  ALPHA_Complex8 *coo_val;  // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;   // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;   // array of nnz elements containing the COO part column indices (integer).
  
  ALPHA_Complex8 *d_ell_val;
  ALPHA_INT      *d_ell_col_ind;
  ALPHA_Complex8 *d_coo_val;
  ALPHA_INT      *d_coo_row_val;
  ALPHA_INT      *d_coo_col_val;
} spmat_hyb_c_t;

typedef struct {
  ALPHA_INT rows;            // number of rows (integer).
  ALPHA_INT cols;            // number of columns (integer).
  ALPHA_INT nnz;             // number of non-zero elements of the COO part (integer)
  ALPHA_INT ell_width;       // maximum number of non-zero elements per row of the ELL part (integer)
  ALPHA_Complex16 *ell_val;  // array of m times ell_width elements containing the ELL part data
                           // (floating point).
  ALPHA_INT *ell_col_ind;    // array of m times ell_width elements containing the ELL part column
                           // indices (integer).
  ALPHA_Complex16 *coo_val;  // array of nnz elements containing the COO part data (floating point).
  ALPHA_INT *coo_row_val;    // array of nnz elements containing the COO part row indices (integer).
  ALPHA_INT *coo_col_val;  // array of nnz elements containing the COO part column indices (integer).
  
  ALPHA_Complex16 *d_ell_val;
  ALPHA_INT       *d_ell_col_ind;
  ALPHA_Complex16 *d_coo_val;
  ALPHA_INT       *d_coo_row_val;
  ALPHA_INT       *d_coo_col_val;
} spmat_hyb_z_t;

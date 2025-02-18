#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"
#include "export_csr.hpp"

template <typename TYPE>
alphasparseStatus_t alphasparse_export_csr_template(const alphasparse_matrix_t source,
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT **rows_start,
                          ALPHA_INT **rows_end,
                          ALPHA_INT **col_indx,
                          TYPE **values)
{
    return export_csr<TYPE>(source,              
                            indexing,            
                            rows,                
                            cols,                
                            rows_start,          
                            rows_end,            
                            col_indx,            
                            values);             
}


#define C_IMPL(ONAME, TYPE)                                     \     
alphasparseStatus_t ONAME(const alphasparse_matrix_t source,    \
                          alphasparseIndexBase_t *indexing,     \ 
                          ALPHA_INT *rows,                      \
                          ALPHA_INT *cols,                      \
                          ALPHA_INT **rows_start,               \
                          ALPHA_INT **rows_end,                 \
                          ALPHA_INT **col_indx,                 \
                          TYPE **values)                        \
{                                                               \
    return alphasparse_export_csr_template(source,              \
                                           indexing,            \
                                           rows,                \
                                           cols,                \
                                           rows_start,          \
                                           rows_end,            \
                                           col_indx,            \
                                           values);             \
}

C_IMPL(alphasparse_s_export_csr, float);
C_IMPL(alphasparse_d_export_csr, double);
C_IMPL(alphasparse_c_export_csr, ALPHA_Complex8);
C_IMPL(alphasparse_z_export_csr, ALPHA_Complex16);
#undef C_IMPL
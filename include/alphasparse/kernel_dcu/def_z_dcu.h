#pragma once

// level 1
#define dcu_axpyi dcu_z_axpyi
#define dcu_gthr dcu_z_gthr
#define dcu_gthrz dcu_z_gthrz
#define dcu_roti dcu_z_roti
#define dcu_sctr dcu_z_sctr
#define dcu_doti dcu_z_doti
#define dcu_dotci dcu_z_dotci

// coo
#define dcu_set_value_coo dcu_set_value_z_coo
#define dcu_update_values_coo dcu_update_values_z_coo

#define dcu_add_coo dcu_add_z_coo
#define dcu_add_coo_trans dcu_add_z_coo_trans

#define dcu_gemv_coo dcu_gemv_z_coo
#define dcu_gemv_coo_trans dcu_gemv_z_coo_trans
#define dcu_symv_coo_n_lo dcu_symv_z_coo_n_lo
#define dcu_symv_coo_u_lo dcu_symv_z_coo_u_lo
#define dcu_symv_coo_n_hi dcu_symv_z_coo_n_hi
#define dcu_symv_coo_u_hi dcu_symv_z_coo_u_hi
#define dcu_trmv_coo_n_lo dcu_trmv_z_coo_n_lo
#define dcu_trmv_coo_u_lo dcu_trmv_z_coo_u_lo
#define dcu_trmv_coo_n_hi dcu_trmv_z_coo_n_hi
#define dcu_trmv_coo_u_hi dcu_trmv_z_coo_u_hi
#define dcu_trmv_coo_n_lo_trans dcu_trmv_z_coo_n_lo_trans
#define dcu_trmv_coo_u_lo_trans dcu_trmv_z_coo_u_lo_trans
#define dcu_trmv_coo_n_hi_trans dcu_trmv_z_coo_n_hi_trans
#define dcu_trmv_coo_u_hi_trans dcu_trmv_z_coo_u_hi_trans
#define dcu_diagmv_coo_n dcu_diagmv_z_coo_n
#define dcu_diagmv_coo_u dcu_diagmv_z_coo_u

#define dcu_gemm_coo_row dcu_gemm_z_coo_row
#define dcu_gemm_coo_col dcu_gemm_z_coo_col
#define dcu_gemm_coo_row_transA dcu_gemm_z_coo_row_transA
#define dcu_gemm_coo_col_transA dcu_gemm_z_coo_col_transA
#define dcu_gemm_coo_row_transB dcu_gemm_z_coo_row_transB
#define dcu_gemm_coo_col_transB dcu_gemm_z_coo_col_transB
#define dcu_gemm_coo_row_transAB dcu_gemm_z_coo_row_transAB
#define dcu_gemm_coo_col_transAB dcu_gemm_z_coo_col_transAB
#define dcu_symm_coo_n_lo_row dcu_symm_z_coo_n_lo_row
#define dcu_symm_coo_u_lo_row dcu_symm_z_coo_u_lo_row
#define dcu_symm_coo_n_hi_row dcu_symm_z_coo_n_hi_row
#define dcu_symm_coo_u_hi_row dcu_symm_z_coo_u_hi_row
#define dcu_symm_coo_n_lo_col dcu_symm_z_coo_n_lo_col
#define dcu_symm_coo_u_lo_col dcu_symm_z_coo_u_lo_col
#define dcu_symm_coo_n_hi_col dcu_symm_z_coo_n_hi_col
#define dcu_symm_coo_u_hi_col dcu_symm_z_coo_u_hi_col
#define dcu_trmm_coo_n_lo_row dcu_trmm_z_coo_n_lo_row
#define dcu_trmm_coo_u_lo_row dcu_trmm_z_coo_u_lo_row
#define dcu_trmm_coo_n_hi_row dcu_trmm_z_coo_n_hi_row
#define dcu_trmm_coo_u_hi_row dcu_trmm_z_coo_u_hi_row
#define dcu_trmm_coo_n_lo_col dcu_trmm_z_coo_n_lo_col
#define dcu_trmm_coo_u_lo_col dcu_trmm_z_coo_u_lo_col
#define dcu_trmm_coo_n_hi_col dcu_trmm_z_coo_n_hi_col
#define dcu_trmm_coo_u_hi_col dcu_trmm_z_coo_u_hi_col
#define dcu_trmm_coo_n_lo_row_trans dcu_trmm_z_coo_n_lo_row_trans
#define dcu_trmm_coo_u_lo_row_trans dcu_trmm_z_coo_u_lo_row_trans
#define dcu_trmm_coo_n_hi_row_trans dcu_trmm_z_coo_n_hi_row_trans
#define dcu_trmm_coo_u_hi_row_trans dcu_trmm_z_coo_u_hi_row_trans
#define dcu_trmm_coo_n_lo_col_trans dcu_trmm_z_coo_n_lo_col_trans
#define dcu_trmm_coo_u_lo_col_trans dcu_trmm_z_coo_u_lo_col_trans
#define dcu_trmm_coo_n_hi_col_trans dcu_trmm_z_coo_n_hi_col_trans
#define dcu_trmm_coo_u_hi_col_trans dcu_trmm_z_coo_u_hi_col_trans
#define dcu_diagmm_coo_n_row dcu_diagmm_z_coo_n_row
#define dcu_diagmm_coo_u_row dcu_diagmm_z_coo_u_row
#define dcu_diagmm_coo_n_col dcu_diagmm_z_coo_n_col
#define dcu_diagmm_coo_u_col dcu_diagmm_z_coo_u_col

#define dcu_spmmd_coo_row dcu_spmmd_z_coo_row
#define dcu_spmmd_coo_col dcu_spmmd_z_coo_col
#define dcu_spmmd_coo_row_trans dcu_spmmd_z_coo_row_trans
#define dcu_spmmd_coo_col_trans dcu_spmmd_z_coo_col_trans

#define dcu_spmm_coo dcu_spmm_z_coo
#define dcu_spmm_coo_trans dcu_spmm_z_coo_trans

#define dcu_trsv_coo_n_lo dcu_trsv_z_coo_n_lo
#define dcu_trsv_coo_u_lo dcu_trsv_z_coo_u_lo
#define dcu_trsv_coo_n_hi dcu_trsv_z_coo_n_hi
#define dcu_trsv_coo_u_hi dcu_trsv_z_coo_u_hi
#define dcu_trsv_coo_n_lo_trans dcu_trsv_z_coo_n_lo_trans
#define dcu_trsv_coo_u_lo_trans dcu_trsv_z_coo_u_lo_trans
#define dcu_trsv_coo_n_hi_trans dcu_trsv_z_coo_n_hi_trans
#define dcu_trsv_coo_u_hi_trans dcu_trsv_z_coo_u_hi_trans
#define dcu_diagsv_coo_n dcu_diagsv_z_coo_n
#define dcu_diagsv_coo_u dcu_diagsv_z_coo_u

#define dcu_trsm_coo_n_lo_row dcu_trsm_z_coo_n_lo_row
#define dcu_trsm_coo_u_lo_row dcu_trsm_z_coo_u_lo_row
#define dcu_trsm_coo_n_hi_row dcu_trsm_z_coo_n_hi_row
#define dcu_trsm_coo_u_hi_row dcu_trsm_z_coo_u_hi_row
#define dcu_trsm_coo_n_lo_col dcu_trsm_z_coo_n_lo_col
#define dcu_trsm_coo_u_lo_col dcu_trsm_z_coo_u_lo_col
#define dcu_trsm_coo_n_hi_col dcu_trsm_z_coo_n_hi_col
#define dcu_trsm_coo_u_hi_col dcu_trsm_z_coo_u_hi_col
#define dcu_trsm_coo_n_lo_row_trans dcu_trsm_z_coo_n_lo_row_trans
#define dcu_trsm_coo_u_lo_row_trans dcu_trsm_z_coo_u_lo_row_trans
#define dcu_trsm_coo_n_hi_row_trans dcu_trsm_z_coo_n_hi_row_trans
#define dcu_trsm_coo_u_hi_row_trans dcu_trsm_z_coo_u_hi_row_trans
#define dcu_trsm_coo_n_lo_col_trans dcu_trsm_z_coo_n_lo_col_trans
#define dcu_trsm_coo_u_lo_col_trans dcu_trsm_z_coo_u_lo_col_trans
#define dcu_trsm_coo_n_hi_col_trans dcu_trsm_z_coo_n_hi_col_trans
#define dcu_trsm_coo_u_hi_col_trans dcu_trsm_z_coo_u_hi_col_trans
#define dcu_diagsm_coo_n_row dcu_diagsm_z_coo_n_row
#define dcu_diagsm_coo_u_row dcu_diagsm_z_coo_u_row
#define dcu_diagsm_coo_n_col dcu_diagsm_z_coo_n_col
#define dcu_diagsm_coo_u_col dcu_diagsm_z_coo_u_col

// csr
#define dcu_set_value_csr dcu_set_value_z_csr
#define dcu_update_values_csr dcu_update_values_z_csr

#define dcu_add_csr dcu_add_z_csr
#define dcu_add_csr_trans dcu_add_z_csr_trans

#define dcu_gemv_csr dcu_gemv_z_csr
#define dcu_gemv_csr_trans dcu_gemv_z_csr_trans
#define dcu_gemv_csr_conj dcu_gemv_z_csr_conj
#define dcu_symv_csr_n_lo dcu_symv_z_csr_n_lo
#define dcu_symv_csr_u_lo dcu_symv_z_csr_u_lo
#define dcu_symv_csr_n_hi dcu_symv_z_csr_n_hi
#define dcu_symv_csr_u_hi dcu_symv_z_csr_u_hi
#define dcu_trmv_csr_n_lo dcu_trmv_z_csr_n_lo
#define dcu_trmv_csr_u_lo dcu_trmv_z_csr_u_lo
#define dcu_trmv_csr_n_hi dcu_trmv_z_csr_n_hi
#define dcu_trmv_csr_u_hi dcu_trmv_z_csr_u_hi
#define dcu_trmv_csr_n_lo_trans dcu_trmv_z_csr_n_lo_trans
#define dcu_trmv_csr_u_lo_trans dcu_trmv_z_csr_u_lo_trans
#define dcu_trmv_csr_n_hi_trans dcu_trmv_z_csr_n_hi_trans
#define dcu_trmv_csr_u_hi_trans dcu_trmv_z_csr_u_hi_trans
#define dcu_diagmv_csr_n dcu_diagmv_z_csr_n
#define dcu_diagmv_csr_u dcu_diagmv_z_csr_u

#define dcu_gemm_csr_row dcu_gemm_z_csr_row
#define dcu_gemm_csr_row_transA dcu_gemm_z_csr_row_transA
#define dcu_gemm_csr_row_transB dcu_gemm_z_csr_row_transB
#define dcu_gemm_csr_row_transAB dcu_gemm_z_csr_row_transAB
#define dcu_gemm_csr_col dcu_gemm_z_csr_col
#define dcu_gemm_csr_col_transA dcu_gemm_z_csr_col_transA
#define dcu_gemm_csr_col_transB dcu_gemm_z_csr_col_transB
#define dcu_gemm_csr_col_transAB dcu_gemm_z_csr_col_transAB
#define dcu_symm_csr_n_lo_row dcu_symm_z_csr_n_lo_row
#define dcu_symm_csr_u_lo_row dcu_symm_z_csr_u_lo_row
#define dcu_symm_csr_n_hi_row dcu_symm_z_csr_n_hi_row
#define dcu_symm_csr_u_hi_row dcu_symm_z_csr_u_hi_row
#define dcu_symm_csr_n_lo_col dcu_symm_z_csr_n_lo_col
#define dcu_symm_csr_u_lo_col dcu_symm_z_csr_u_lo_col
#define dcu_symm_csr_n_hi_col dcu_symm_z_csr_n_hi_col
#define dcu_symm_csr_u_hi_col dcu_symm_z_csr_u_hi_col
#define dcu_trmm_csr_n_lo_row dcu_trmm_z_csr_n_lo_row
#define dcu_trmm_csr_u_lo_row dcu_trmm_z_csr_u_lo_row
#define dcu_trmm_csr_n_hi_row dcu_trmm_z_csr_n_hi_row
#define dcu_trmm_csr_u_hi_row dcu_trmm_z_csr_u_hi_row
#define dcu_trmm_csr_n_lo_col dcu_trmm_z_csr_n_lo_col
#define dcu_trmm_csr_u_lo_col dcu_trmm_z_csr_u_lo_col
#define dcu_trmm_csr_n_hi_col dcu_trmm_z_csr_n_hi_col
#define dcu_trmm_csr_u_hi_col dcu_trmm_z_csr_u_hi_col
#define dcu_trmm_csr_n_lo_row_trans dcu_trmm_z_csr_n_lo_row_trans
#define dcu_trmm_csr_u_lo_row_trans dcu_trmm_z_csr_u_lo_row_trans
#define dcu_trmm_csr_n_hi_row_trans dcu_trmm_z_csr_n_hi_row_trans
#define dcu_trmm_csr_u_hi_row_trans dcu_trmm_z_csr_u_hi_row_trans
#define dcu_trmm_csr_n_lo_col_trans dcu_trmm_z_csr_n_lo_col_trans
#define dcu_trmm_csr_u_lo_col_trans dcu_trmm_z_csr_u_lo_col_trans
#define dcu_trmm_csr_n_hi_col_trans dcu_trmm_z_csr_n_hi_col_trans
#define dcu_trmm_csr_u_hi_col_trans dcu_trmm_z_csr_u_hi_col_trans
#define dcu_diagmm_csr_n_row dcu_diagmm_z_csr_n_row
#define dcu_diagmm_csr_u_row dcu_diagmm_z_csr_u_row
#define dcu_diagmm_csr_n_col dcu_diagmm_z_csr_n_col
#define dcu_diagmm_csr_u_col dcu_diagmm_z_csr_u_col

#define dcu_gemmi_csr dcu_gemmi_z_csr
#define dcu_gemmi_csr_transA dcu_gemmi_z_csr_transA
#define dcu_gemmi_csr_transB dcu_gemmi_z_csr_transB
#define dcu_gemmi_csr_transAB dcu_gemmi_z_csr_transAB

#define dcu_spmmd_csr_row dcu_spmmd_z_csr_row
#define dcu_spmmd_csr_col dcu_spmmd_z_csr_col
#define dcu_spmmd_csr_row_trans dcu_spmmd_z_csr_row_trans
#define dcu_spmmd_csr_col_trans dcu_spmmd_z_csr_col_trans

#define dcu_spmm_csr dcu_spmm_z_csr
#define dcu_spmm_csr_trans dcu_spmm_z_csr_trans

#define dcu_spgemm_csr dcu_spgemm_z_csr
#define dcu_spgemm_csr_transA dcu_spgemm_z_csr_transA
#define dcu_spgemm_csr_transB dcu_spgemm_z_csr_transB
#define dcu_spgemm_csr_transAB dcu_spgemm_z_csr_transAB

#define dcu_trsv_csr_n_lo dcu_trsv_z_csr_n_lo
#define dcu_trsv_csr_u_lo dcu_trsv_z_csr_u_lo
#define dcu_trsv_csr_n_hi dcu_trsv_z_csr_n_hi
#define dcu_trsv_csr_u_hi dcu_trsv_z_csr_u_hi
#define dcu_trsv_csr_n_lo_trans dcu_trsv_z_csr_n_lo_trans
#define dcu_trsv_csr_u_lo_trans dcu_trsv_z_csr_u_lo_trans
#define dcu_trsv_csr_n_hi_trans dcu_trsv_z_csr_n_hi_trans
#define dcu_trsv_csr_u_hi_trans dcu_trsv_z_csr_u_hi_trans
#define dcu_diagsv_csr_n dcu_diagsv_z_csr_n
#define dcu_diagsv_csr_u dcu_diagsv_z_csr_u

#define dcu_trsm_csr_n_lo dcu_trsm_z_csr_n_lo
#define dcu_trsm_csr_u_lo dcu_trsm_z_csr_u_lo
#define dcu_trsm_csr_n_hi dcu_trsm_z_csr_n_hi
#define dcu_trsm_csr_u_hi dcu_trsm_z_csr_u_hi
#define dcu_trsm_csr_n_lo_transA dcu_trsm_z_csr_n_lo_transA
#define dcu_trsm_csr_u_lo_transA dcu_trsm_z_csr_u_lo_transA
#define dcu_trsm_csr_n_hi_transA dcu_trsm_z_csr_n_hi_transA
#define dcu_trsm_csr_u_hi_transA dcu_trsm_z_csr_u_hi_transA
#define dcu_trsm_csr_n_lo_col_transA dcu_trsm_z_csr_n_lo_col_transA
#define dcu_trsm_csr_u_lo_col_transA dcu_trsm_z_csr_u_lo_col_transA
#define dcu_trsm_csr_n_hi_col_transA dcu_trsm_z_csr_n_hi_col_transA
#define dcu_trsm_csr_u_hi_col_transA dcu_trsm_z_csr_u_hi_col_transA

#define dcu_trsm_csr_n_lo_transB dcu_trsm_z_csr_n_lo_transB
#define dcu_trsm_csr_u_lo_transB dcu_trsm_z_csr_u_lo_transB
#define dcu_trsm_csr_n_hi_transB dcu_trsm_z_csr_n_hi_transB
#define dcu_trsm_csr_u_hi_transB dcu_trsm_z_csr_u_hi_transB
#define dcu_trsm_csr_n_lo_transAB dcu_trsm_z_csr_n_lo_transAB
#define dcu_trsm_csr_u_lo_transAB dcu_trsm_z_csr_u_lo_transAB
#define dcu_trsm_csr_n_hi_transAB dcu_trsm_z_csr_n_hi_transAB
#define dcu_trsm_csr_u_hi_transAB dcu_trsm_z_csr_u_hi_transAB
#define dcu_trsm_csr_n_lo_col_transAB dcu_trsm_z_csr_n_lo_col_transAB
#define dcu_trsm_csr_u_lo_col_transAB dcu_trsm_z_csr_u_lo_col_transAB
#define dcu_trsm_csr_n_hi_col_transAB dcu_trsm_z_csr_n_hi_col_transAB
#define dcu_trsm_csr_u_hi_col_transAB dcu_trsm_z_csr_u_hi_col_transAB

// csc
#define dcu_set_value_csc dcu_set_value_z_csc
#define dcu_update_values_csc dcu_update_values_z_csc

#define dcu_add_csc dcu_add_z_csc
#define dcu_add_csc_trans dcu_add_z_csc_trans

#define dcu_gemv_csc dcu_gemv_z_csc
#define dcu_gemv_csc_trans dcu_gemv_z_csc_trans
#define dcu_symv_csc_n_lo dcu_symv_z_csc_n_lo
#define dcu_symv_csc_u_lo dcu_symv_z_csc_u_lo
#define dcu_symv_csc_n_hi dcu_symv_z_csc_n_hi
#define dcu_symv_csc_u_hi dcu_symv_z_csc_u_hi
#define dcu_trmv_csc_n_lo dcu_trmv_z_csc_n_lo
#define dcu_trmv_csc_u_lo dcu_trmv_z_csc_u_lo
#define dcu_trmv_csc_n_hi dcu_trmv_z_csc_n_hi
#define dcu_trmv_csc_u_hi dcu_trmv_z_csc_u_hi
#define dcu_trmv_csc_n_lo_trans dcu_trmv_z_csc_n_lo_trans
#define dcu_trmv_csc_u_lo_trans dcu_trmv_z_csc_u_lo_trans
#define dcu_trmv_csc_n_hi_trans dcu_trmv_z_csc_n_hi_trans
#define dcu_trmv_csc_u_hi_trans dcu_trmv_z_csc_u_hi_trans
#define dcu_diagmv_csc_n dcu_diagmv_z_csc_n
#define dcu_diagmv_csc_u dcu_diagmv_z_csc_u

#define dcu_gemm_csc_row dcu_gemm_z_csc_row
#define dcu_gemm_csc_col dcu_gemm_z_csc_col
#define dcu_gemm_csc_row_trans dcu_gemm_z_csc_row_trans
#define dcu_gemm_csc_col_trans dcu_gemm_z_csc_col_trans
#define dcu_symm_csc_n_lo_row dcu_symm_z_csc_n_lo_row
#define dcu_symm_csc_u_lo_row dcu_symm_z_csc_u_lo_row
#define dcu_symm_csc_n_hi_row dcu_symm_z_csc_n_hi_row
#define dcu_symm_csc_u_hi_row dcu_symm_z_csc_u_hi_row
#define dcu_symm_csc_n_lo_col dcu_symm_z_csc_n_lo_col
#define dcu_symm_csc_u_lo_col dcu_symm_z_csc_u_lo_col
#define dcu_symm_csc_n_hi_col dcu_symm_z_csc_n_hi_col
#define dcu_symm_csc_u_hi_col dcu_symm_z_csc_u_hi_col
#define dcu_trmm_csc_n_lo_row dcu_trmm_z_csc_n_lo_row
#define dcu_trmm_csc_u_lo_row dcu_trmm_z_csc_u_lo_row
#define dcu_trmm_csc_n_hi_row dcu_trmm_z_csc_n_hi_row
#define dcu_trmm_csc_u_hi_row dcu_trmm_z_csc_u_hi_row
#define dcu_trmm_csc_n_lo_col dcu_trmm_z_csc_n_lo_col
#define dcu_trmm_csc_u_lo_col dcu_trmm_z_csc_u_lo_col
#define dcu_trmm_csc_n_hi_col dcu_trmm_z_csc_n_hi_col
#define dcu_trmm_csc_u_hi_col dcu_trmm_z_csc_u_hi_col
#define dcu_trmm_csc_n_lo_row_trans dcu_trmm_z_csc_n_lo_row_trans
#define dcu_trmm_csc_u_lo_row_trans dcu_trmm_z_csc_u_lo_row_trans
#define dcu_trmm_csc_n_hi_row_trans dcu_trmm_z_csc_n_hi_row_trans
#define dcu_trmm_csc_u_hi_row_trans dcu_trmm_z_csc_u_hi_row_trans
#define dcu_trmm_csc_n_lo_col_trans dcu_trmm_z_csc_n_lo_col_trans
#define dcu_trmm_csc_u_lo_col_trans dcu_trmm_z_csc_u_lo_col_trans
#define dcu_trmm_csc_n_hi_col_trans dcu_trmm_z_csc_n_hi_col_trans
#define dcu_trmm_csc_u_hi_col_trans dcu_trmm_z_csc_u_hi_col_trans
#define dcu_diagmm_csc_n_row dcu_diagmm_z_csc_n_row
#define dcu_diagmm_csc_u_row dcu_diagmm_z_csc_u_row
#define dcu_diagmm_csc_n_col dcu_diagmm_z_csc_n_col
#define dcu_diagmm_csc_u_col dcu_diagmm_z_csc_u_col

#define dcu_spmmd_csc_row dcu_spmmd_z_csc_row
#define dcu_spmmd_csc_col dcu_spmmd_z_csc_col
#define dcu_spmmd_csc_row_trans dcu_spmmd_z_csc_row_trans
#define dcu_spmmd_csc_col_trans dcu_spmmd_z_csc_col_trans

#define dcu_spmm_csc dcu_spmm_z_csc
#define dcu_spmm_csc_trans dcu_spmm_z_csc_trans

#define dcu_trsv_csc_n_lo dcu_trsv_z_csc_n_lo
#define dcu_trsv_csc_u_lo dcu_trsv_z_csc_u_lo
#define dcu_trsv_csc_n_hi dcu_trsv_z_csc_n_hi
#define dcu_trsv_csc_u_hi dcu_trsv_z_csc_u_hi
#define dcu_trsv_csc_n_lo_trans dcu_trsv_z_csc_n_lo_trans
#define dcu_trsv_csc_u_lo_trans dcu_trsv_z_csc_u_lo_trans
#define dcu_trsv_csc_n_hi_trans dcu_trsv_z_csc_n_hi_trans
#define dcu_trsv_csc_u_hi_trans dcu_trsv_z_csc_u_hi_trans
#define dcu_diagsv_csc_n dcu_diagsv_z_csc_n
#define dcu_diagsv_csc_u dcu_diagsv_z_csc_u

#define dcu_trsm_csc_n_lo_row dcu_trsm_z_csc_n_lo_row
#define dcu_trsm_csc_u_lo_row dcu_trsm_z_csc_u_lo_row
#define dcu_trsm_csc_n_hi_row dcu_trsm_z_csc_n_hi_row
#define dcu_trsm_csc_u_hi_row dcu_trsm_z_csc_u_hi_row
#define dcu_trsm_csc_n_lo_col dcu_trsm_z_csc_n_lo_col
#define dcu_trsm_csc_u_lo_col dcu_trsm_z_csc_u_lo_col
#define dcu_trsm_csc_n_hi_col dcu_trsm_z_csc_n_hi_col
#define dcu_trsm_csc_u_hi_col dcu_trsm_z_csc_u_hi_col
#define dcu_trsm_csc_n_lo_row_trans dcu_trsm_z_csc_n_lo_row_trans
#define dcu_trsm_csc_u_lo_row_trans dcu_trsm_z_csc_u_lo_row_trans
#define dcu_trsm_csc_n_hi_row_trans dcu_trsm_z_csc_n_hi_row_trans
#define dcu_trsm_csc_u_hi_row_trans dcu_trsm_z_csc_u_hi_row_trans
#define dcu_trsm_csc_n_lo_col_trans dcu_trsm_z_csc_n_lo_col_trans
#define dcu_trsm_csc_u_lo_col_trans dcu_trsm_z_csc_u_lo_col_trans
#define dcu_trsm_csc_n_hi_col_trans dcu_trsm_z_csc_n_hi_col_trans
#define dcu_trsm_csc_u_hi_col_trans dcu_trsm_z_csc_u_hi_col_trans
#define dcu_diagsm_csc_n_row dcu_diagsm_z_csc_n_row
#define dcu_diagsm_csc_u_row dcu_diagsm_z_csc_u_row
#define dcu_diagsm_csc_n_col dcu_diagsm_z_csc_n_col
#define dcu_diagsm_csc_u_col dcu_diagsm_z_csc_u_col

// bsr
#define dcu_set_value_bsr dcu_set_value_z_bsr
#define dcu_update_values_bsr dcu_update_values_z_bsr

#define dcu_add_bsr dcu_add_z_bsr
#define dcu_add_bsr_trans dcu_add_z_bsr_trans

#define dcu_gemv_bsr dcu_gemv_z_bsr
#define dcu_gemv_bsr_trans dcu_gemv_z_bsr_trans
#define dcu_symv_bsr_n_lo dcu_symv_z_bsr_n_lo
#define dcu_symv_bsr_u_lo dcu_symv_z_bsr_u_lo
#define dcu_symv_bsr_n_hi dcu_symv_z_bsr_n_hi
#define dcu_symv_bsr_u_hi dcu_symv_z_bsr_u_hi
#define dcu_trmv_bsr_n_lo dcu_trmv_z_bsr_n_lo
#define dcu_trmv_bsr_u_lo dcu_trmv_z_bsr_u_lo
#define dcu_trmv_bsr_n_hi dcu_trmv_z_bsr_n_hi
#define dcu_trmv_bsr_u_hi dcu_trmv_z_bsr_u_hi
#define dcu_trmv_bsr_n_lo_trans dcu_trmv_z_bsr_n_lo_trans
#define dcu_trmv_bsr_u_lo_trans dcu_trmv_z_bsr_u_lo_trans
#define dcu_trmv_bsr_n_hi_trans dcu_trmv_z_bsr_n_hi_trans
#define dcu_trmv_bsr_u_hi_trans dcu_trmv_z_bsr_u_hi_trans
#define dcu_diagmv_bsr_n dcu_diagmv_z_bsr_n
#define dcu_diagmv_bsr_u dcu_diagmv_z_bsr_u

#define dcu_gemm_bsr dcu_gemm_z_bsr
#define dcu_gemm_bsr_transA dcu_gemm_z_bsr_transA
#define dcu_gemm_bsr_transB dcu_gemm_z_bsr_transB
#define dcu_gemm_bsr_transAB dcu_gemm_z_bsr_transAB
#define dcu_symm_bsr_n_lo_row dcu_symm_z_bsr_n_lo_row
#define dcu_symm_bsr_u_lo_row dcu_symm_z_bsr_u_lo_row
#define dcu_symm_bsr_n_hi_row dcu_symm_z_bsr_n_hi_row
#define dcu_symm_bsr_u_hi_row dcu_symm_z_bsr_u_hi_row
#define dcu_symm_bsr_n_lo_col dcu_symm_z_bsr_n_lo_col
#define dcu_symm_bsr_u_lo_col dcu_symm_z_bsr_u_lo_col
#define dcu_symm_bsr_n_hi_col dcu_symm_z_bsr_n_hi_col
#define dcu_symm_bsr_u_hi_col dcu_symm_z_bsr_u_hi_col
#define dcu_trmm_bsr_n_lo_row dcu_trmm_z_bsr_n_lo_row
#define dcu_trmm_bsr_u_lo_row dcu_trmm_z_bsr_u_lo_row
#define dcu_trmm_bsr_n_hi_row dcu_trmm_z_bsr_n_hi_row
#define dcu_trmm_bsr_u_hi_row dcu_trmm_z_bsr_u_hi_row
#define dcu_trmm_bsr_n_lo_col dcu_trmm_z_bsr_n_lo_col
#define dcu_trmm_bsr_u_lo_col dcu_trmm_z_bsr_u_lo_col
#define dcu_trmm_bsr_n_hi_col dcu_trmm_z_bsr_n_hi_col
#define dcu_trmm_bsr_u_hi_col dcu_trmm_z_bsr_u_hi_col
#define dcu_trmm_bsr_n_lo_row_trans dcu_trmm_z_bsr_n_lo_row_trans
#define dcu_trmm_bsr_u_lo_row_trans dcu_trmm_z_bsr_u_lo_row_trans
#define dcu_trmm_bsr_n_hi_row_trans dcu_trmm_z_bsr_n_hi_row_trans
#define dcu_trmm_bsr_u_hi_row_trans dcu_trmm_z_bsr_u_hi_row_trans
#define dcu_trmm_bsr_n_lo_col_trans dcu_trmm_z_bsr_n_lo_col_trans
#define dcu_trmm_bsr_u_lo_col_trans dcu_trmm_z_bsr_u_lo_col_trans
#define dcu_trmm_bsr_n_hi_col_trans dcu_trmm_z_bsr_n_hi_col_trans
#define dcu_trmm_bsr_u_hi_col_trans dcu_trmm_z_bsr_u_hi_col_trans
#define dcu_diagmm_bsr_n_row dcu_diagmm_z_bsr_n_row
#define dcu_diagmm_bsr_u_row dcu_diagmm_z_bsr_u_row
#define dcu_diagmm_bsr_n_col dcu_diagmm_z_bsr_n_col
#define dcu_diagmm_bsr_u_col dcu_diagmm_z_bsr_u_col

#define dcu_spmmd_bsr_row dcu_spmmd_z_bsr_row
#define dcu_spmmd_bsr_col dcu_spmmd_z_bsr_col
#define dcu_spmmd_bsr_row_trans dcu_spmmd_z_bsr_row_trans
#define dcu_spmmd_bsr_col_trans dcu_spmmd_z_bsr_col_trans

#define dcu_spmm_bsr dcu_spmm_z_bsr
#define dcu_spmm_bsr_trans dcu_spmm_z_bsr_trans

#define dcu_trsv_bsr_n_lo dcu_trsv_z_bsr_n_lo
#define dcu_trsv_bsr_u_lo dcu_trsv_z_bsr_u_lo
#define dcu_trsv_bsr_n_hi dcu_trsv_z_bsr_n_hi
#define dcu_trsv_bsr_u_hi dcu_trsv_z_bsr_u_hi
#define dcu_trsv_bsr_n_lo_trans dcu_trsv_z_bsr_n_lo_trans
#define dcu_trsv_bsr_u_lo_trans dcu_trsv_z_bsr_u_lo_trans
#define dcu_trsv_bsr_n_hi_trans dcu_trsv_z_bsr_n_hi_trans
#define dcu_trsv_bsr_u_hi_trans dcu_trsv_z_bsr_u_hi_trans
#define dcu_diagsv_bsr_n dcu_diagsv_z_bsr_n
#define dcu_diagsv_bsr_u dcu_diagsv_z_bsr_u

#define dcu_trsm_bsr_n_lo_row dcu_trsm_z_bsr_n_lo_row
#define dcu_trsm_bsr_u_lo_row dcu_trsm_z_bsr_u_lo_row
#define dcu_trsm_bsr_n_hi_row dcu_trsm_z_bsr_n_hi_row
#define dcu_trsm_bsr_u_hi_row dcu_trsm_z_bsr_u_hi_row
#define dcu_trsm_bsr_n_lo_col dcu_trsm_z_bsr_n_lo_col
#define dcu_trsm_bsr_u_lo_col dcu_trsm_z_bsr_u_lo_col
#define dcu_trsm_bsr_n_hi_col dcu_trsm_z_bsr_n_hi_col
#define dcu_trsm_bsr_u_hi_col dcu_trsm_z_bsr_u_hi_col
#define dcu_trsm_bsr_n_lo_row_trans dcu_trsm_z_bsr_n_lo_row_trans
#define dcu_trsm_bsr_u_lo_row_trans dcu_trsm_z_bsr_u_lo_row_trans
#define dcu_trsm_bsr_n_hi_row_trans dcu_trsm_z_bsr_n_hi_row_trans
#define dcu_trsm_bsr_u_hi_row_trans dcu_trsm_z_bsr_u_hi_row_trans
#define dcu_trsm_bsr_n_lo_col_trans dcu_trsm_z_bsr_n_lo_col_trans
#define dcu_trsm_bsr_u_lo_col_trans dcu_trsm_z_bsr_u_lo_col_trans
#define dcu_trsm_bsr_n_hi_col_trans dcu_trsm_z_bsr_n_hi_col_trans
#define dcu_trsm_bsr_u_hi_col_trans dcu_trsm_z_bsr_u_hi_col_trans
#define dcu_diagsm_bsr_n_row dcu_diagsm_z_bsr_n_row
#define dcu_diagsm_bsr_u_row dcu_diagsm_z_bsr_u_row
#define dcu_diagsm_bsr_n_col dcu_diagsm_z_bsr_n_col
#define dcu_diagsm_bsr_u_col dcu_diagsm_z_bsr_u_col

// ell
#define dcu_set_value_ell dcu_set_value_z_ell
#define dcu_update_values_ell dcu_update_values_z_ell

#define dcu_add_ell dcu_add_z_ell
#define dcu_add_ell_trans dcu_add_z_ell_trans

#define dcu_gemv_ell dcu_gemv_z_ell
#define dcu_gemv_ell_trans dcu_gemv_z_ell_trans
#define dcu_symv_ell_n_lo dcu_symv_z_ell_n_lo
#define dcu_symv_ell_u_lo dcu_symv_z_ell_u_lo
#define dcu_symv_ell_n_hi dcu_symv_z_ell_n_hi
#define dcu_symv_ell_u_hi dcu_symv_z_ell_u_hi
#define dcu_trmv_ell_n_lo dcu_trmv_z_ell_n_lo
#define dcu_trmv_ell_u_lo dcu_trmv_z_ell_u_lo
#define dcu_trmv_ell_n_hi dcu_trmv_z_ell_n_hi
#define dcu_trmv_ell_u_hi dcu_trmv_z_ell_u_hi
#define dcu_trmv_ell_n_lo_trans dcu_trmv_z_ell_n_lo_trans
#define dcu_trmv_ell_u_lo_trans dcu_trmv_z_ell_u_lo_trans
#define dcu_trmv_ell_n_hi_trans dcu_trmv_z_ell_n_hi_trans
#define dcu_trmv_ell_u_hi_trans dcu_trmv_z_ell_u_hi_trans
#define dcu_diagmv_ell_n dcu_diagmv_z_ell_n
#define dcu_diagmv_ell_u dcu_diagmv_z_ell_u

#define dcu_gemm_ell_row dcu_gemm_z_ell_row
#define dcu_gemm_ell_col dcu_gemm_z_ell_col
#define dcu_gemm_ell_row_trans dcu_gemm_z_ell_row_trans
#define dcu_gemm_ell_col_trans dcu_gemm_z_ell_col_trans
#define dcu_symm_ell_n_lo_row dcu_symm_z_ell_n_lo_row
#define dcu_symm_ell_u_lo_row dcu_symm_z_ell_u_lo_row
#define dcu_symm_ell_n_hi_row dcu_symm_z_ell_n_hi_row
#define dcu_symm_ell_u_hi_row dcu_symm_z_ell_u_hi_row
#define dcu_symm_ell_n_lo_col dcu_symm_z_ell_n_lo_col
#define dcu_symm_ell_u_lo_col dcu_symm_z_ell_u_lo_col
#define dcu_symm_ell_n_hi_col dcu_symm_z_ell_n_hi_col
#define dcu_symm_ell_u_hi_col dcu_symm_z_ell_u_hi_col
#define dcu_trmm_ell_n_lo_row dcu_trmm_z_ell_n_lo_row
#define dcu_trmm_ell_u_lo_row dcu_trmm_z_ell_u_lo_row
#define dcu_trmm_ell_n_hi_row dcu_trmm_z_ell_n_hi_row
#define dcu_trmm_ell_u_hi_row dcu_trmm_z_ell_u_hi_row
#define dcu_trmm_ell_n_lo_col dcu_trmm_z_ell_n_lo_col
#define dcu_trmm_ell_u_lo_col dcu_trmm_z_ell_u_lo_col
#define dcu_trmm_ell_n_hi_col dcu_trmm_z_ell_n_hi_col
#define dcu_trmm_ell_u_hi_col dcu_trmm_z_ell_u_hi_col
#define dcu_trmm_ell_n_lo_row_trans dcu_trmm_z_ell_n_lo_row_trans
#define dcu_trmm_ell_u_lo_row_trans dcu_trmm_z_ell_u_lo_row_trans
#define dcu_trmm_ell_n_hi_row_trans dcu_trmm_z_ell_n_hi_row_trans
#define dcu_trmm_ell_u_hi_row_trans dcu_trmm_z_ell_u_hi_row_trans
#define dcu_trmm_ell_n_lo_col_trans dcu_trmm_z_ell_n_lo_col_trans
#define dcu_trmm_ell_u_lo_col_trans dcu_trmm_z_ell_u_lo_col_trans
#define dcu_trmm_ell_n_hi_col_trans dcu_trmm_z_ell_n_hi_col_trans
#define dcu_trmm_ell_u_hi_col_trans dcu_trmm_z_ell_u_hi_col_trans
#define dcu_diagmm_ell_n_row dcu_diagmm_z_ell_n_row
#define dcu_diagmm_ell_u_row dcu_diagmm_z_ell_u_row
#define dcu_diagmm_ell_n_col dcu_diagmm_z_ell_n_col
#define dcu_diagmm_ell_u_col dcu_diagmm_z_ell_u_col

#define dcu_spmmd_ell_row dcu_spmmd_z_ell_row
#define dcu_spmmd_ell_col dcu_spmmd_z_ell_col
#define dcu_spmmd_ell_row_trans dcu_spmmd_z_ell_row_trans
#define dcu_spmmd_ell_col_trans dcu_spmmd_z_ell_col_trans

#define dcu_spmm_ell dcu_spmm_z_ell
#define dcu_spmm_ell_trans dcu_spmm_z_ell_trans

#define dcu_trsv_ell_n_lo dcu_trsv_z_ell_n_lo
#define dcu_trsv_ell_u_lo dcu_trsv_z_ell_u_lo
#define dcu_trsv_ell_n_hi dcu_trsv_z_ell_n_hi
#define dcu_trsv_ell_u_hi dcu_trsv_z_ell_u_hi
#define dcu_trsv_ell_n_lo_trans dcu_trsv_z_ell_n_lo_trans
#define dcu_trsv_ell_u_lo_trans dcu_trsv_z_ell_u_lo_trans
#define dcu_trsv_ell_n_hi_trans dcu_trsv_z_ell_n_hi_trans
#define dcu_trsv_ell_u_hi_trans dcu_trsv_z_ell_u_hi_trans
#define dcu_diagsv_ell_n dcu_diagsv_z_ell_n
#define dcu_diagsv_ell_u dcu_diagsv_z_ell_u

#define dcu_trsm_ell_n_lo_row dcu_trsm_z_ell_n_lo_row
#define dcu_trsm_ell_u_lo_row dcu_trsm_z_ell_u_lo_row
#define dcu_trsm_ell_n_hi_row dcu_trsm_z_ell_n_hi_row
#define dcu_trsm_ell_u_hi_row dcu_trsm_z_ell_u_hi_row
#define dcu_trsm_ell_n_lo_col dcu_trsm_z_ell_n_lo_col
#define dcu_trsm_ell_u_lo_col dcu_trsm_z_ell_u_lo_col
#define dcu_trsm_ell_n_hi_col dcu_trsm_z_ell_n_hi_col
#define dcu_trsm_ell_u_hi_col dcu_trsm_z_ell_u_hi_col
#define dcu_trsm_ell_n_lo_row_trans dcu_trsm_z_ell_n_lo_row_trans
#define dcu_trsm_ell_u_lo_row_trans dcu_trsm_z_ell_u_lo_row_trans
#define dcu_trsm_ell_n_hi_row_trans dcu_trsm_z_ell_n_hi_row_trans
#define dcu_trsm_ell_u_hi_row_trans dcu_trsm_z_ell_u_hi_row_trans
#define dcu_trsm_ell_n_lo_col_trans dcu_trsm_z_ell_n_lo_col_trans
#define dcu_trsm_ell_u_lo_col_trans dcu_trsm_z_ell_u_lo_col_trans
#define dcu_trsm_ell_n_hi_col_trans dcu_trsm_z_ell_n_hi_col_trans
#define dcu_trsm_ell_u_hi_col_trans dcu_trsm_z_ell_u_hi_col_trans
#define dcu_diagsm_ell_n_row dcu_diagsm_z_ell_n_row
#define dcu_diagsm_ell_u_row dcu_diagsm_z_ell_u_row
#define dcu_diagsm_ell_n_col dcu_diagsm_z_ell_n_col
#define dcu_diagsm_ell_u_col dcu_diagsm_z_ell_u_col

// hyb
#define dcu_set_value_hyb dcu_set_value_z_hyb
#define dcu_update_values_hyb dcu_update_values_z_hyb

#define dcu_add_hyb dcu_add_z_hyb
#define dcu_add_hyb_trans dcu_add_z_hyb_trans

#define dcu_gemv_hyb dcu_gemv_z_hyb
#define dcu_gemv_hyb_trans dcu_gemv_z_hyb_trans
#define dcu_symv_hyb_n_lo dcu_symv_z_hyb_n_lo
#define dcu_symv_hyb_u_lo dcu_symv_z_hyb_u_lo
#define dcu_symv_hyb_n_hi dcu_symv_z_hyb_n_hi
#define dcu_symv_hyb_u_hi dcu_symv_z_hyb_u_hi
#define dcu_trmv_hyb_n_lo dcu_trmv_z_hyb_n_lo
#define dcu_trmv_hyb_u_lo dcu_trmv_z_hyb_u_lo
#define dcu_trmv_hyb_n_hi dcu_trmv_z_hyb_n_hi
#define dcu_trmv_hyb_u_hi dcu_trmv_z_hyb_u_hi
#define dcu_trmv_hyb_n_lo_trans dcu_trmv_z_hyb_n_lo_trans
#define dcu_trmv_hyb_u_lo_trans dcu_trmv_z_hyb_u_lo_trans
#define dcu_trmv_hyb_n_hi_trans dcu_trmv_z_hyb_n_hi_trans
#define dcu_trmv_hyb_u_hi_trans dcu_trmv_z_hyb_u_hi_trans
#define dcu_diagmv_hyb_n dcu_diagmv_z_hyb_n
#define dcu_diagmv_hyb_u dcu_diagmv_z_hyb_u

#define dcu_gemm_hyb_row dcu_gemm_z_hyb_row
#define dcu_gemm_hyb_col dcu_gemm_z_hyb_col
#define dcu_gemm_hyb_row_trans dcu_gemm_z_hyb_row_trans
#define dcu_gemm_hyb_col_trans dcu_gemm_z_hyb_col_trans
#define dcu_symm_hyb_n_lo_row dcu_symm_z_hyb_n_lo_row
#define dcu_symm_hyb_u_lo_row dcu_symm_z_hyb_u_lo_row
#define dcu_symm_hyb_n_hi_row dcu_symm_z_hyb_n_hi_row
#define dcu_symm_hyb_u_hi_row dcu_symm_z_hyb_u_hi_row
#define dcu_symm_hyb_n_lo_col dcu_symm_z_hyb_n_lo_col
#define dcu_symm_hyb_u_lo_col dcu_symm_z_hyb_u_lo_col
#define dcu_symm_hyb_n_hi_col dcu_symm_z_hyb_n_hi_col
#define dcu_symm_hyb_u_hi_col dcu_symm_z_hyb_u_hi_col
#define dcu_trmm_hyb_n_lo_row dcu_trmm_z_hyb_n_lo_row
#define dcu_trmm_hyb_u_lo_row dcu_trmm_z_hyb_u_lo_row
#define dcu_trmm_hyb_n_hi_row dcu_trmm_z_hyb_n_hi_row
#define dcu_trmm_hyb_u_hi_row dcu_trmm_z_hyb_u_hi_row
#define dcu_trmm_hyb_n_lo_col dcu_trmm_z_hyb_n_lo_col
#define dcu_trmm_hyb_u_lo_col dcu_trmm_z_hyb_u_lo_col
#define dcu_trmm_hyb_n_hi_col dcu_trmm_z_hyb_n_hi_col
#define dcu_trmm_hyb_u_hi_col dcu_trmm_z_hyb_u_hi_col
#define dcu_trmm_hyb_n_lo_row_trans dcu_trmm_z_hyb_n_lo_row_trans
#define dcu_trmm_hyb_u_lo_row_trans dcu_trmm_z_hyb_u_lo_row_trans
#define dcu_trmm_hyb_n_hi_row_trans dcu_trmm_z_hyb_n_hi_row_trans
#define dcu_trmm_hyb_u_hi_row_trans dcu_trmm_z_hyb_u_hi_row_trans
#define dcu_trmm_hyb_n_lo_col_trans dcu_trmm_z_hyb_n_lo_col_trans
#define dcu_trmm_hyb_u_lo_col_trans dcu_trmm_z_hyb_u_lo_col_trans
#define dcu_trmm_hyb_n_hi_col_trans dcu_trmm_z_hyb_n_hi_col_trans
#define dcu_trmm_hyb_u_hi_col_trans dcu_trmm_z_hyb_u_hi_col_trans
#define dcu_diagmm_hyb_n_row dcu_diagmm_z_hyb_n_row
#define dcu_diagmm_hyb_u_row dcu_diagmm_z_hyb_u_row
#define dcu_diagmm_hyb_n_col dcu_diagmm_z_hyb_n_col
#define dcu_diagmm_hyb_u_col dcu_diagmm_z_hyb_u_col

#define dcu_spmmd_hyb_row dcu_spmmd_z_hyb_row
#define dcu_spmmd_hyb_col dcu_spmmd_z_hyb_col
#define dcu_spmmd_hyb_row_trans dcu_spmmd_z_hyb_row_trans
#define dcu_spmmd_hyb_col_trans dcu_spmmd_z_hyb_col_trans

#define dcu_spmm_hyb dcu_spmm_z_hyb
#define dcu_spmm_hyb_trans dcu_spmm_z_hyb_trans

#define dcu_trsv_hyb_n_lo dcu_trsv_z_hyb_n_lo
#define dcu_trsv_hyb_u_lo dcu_trsv_z_hyb_u_lo
#define dcu_trsv_hyb_n_hi dcu_trsv_z_hyb_n_hi
#define dcu_trsv_hyb_u_hi dcu_trsv_z_hyb_u_hi
#define dcu_trsv_hyb_n_lo_trans dcu_trsv_z_hyb_n_lo_trans
#define dcu_trsv_hyb_u_lo_trans dcu_trsv_z_hyb_u_lo_trans
#define dcu_trsv_hyb_n_hi_trans dcu_trsv_z_hyb_n_hi_trans
#define dcu_trsv_hyb_u_hi_trans dcu_trsv_z_hyb_u_hi_trans
#define dcu_diagsv_hyb_n dcu_diagsv_z_hyb_n
#define dcu_diagsv_hyb_u dcu_diagsv_z_hyb_u

#define dcu_trsm_hyb_n_lo_row dcu_trsm_z_hyb_n_lo_row
#define dcu_trsm_hyb_u_lo_row dcu_trsm_z_hyb_u_lo_row
#define dcu_trsm_hyb_n_hi_row dcu_trsm_z_hyb_n_hi_row
#define dcu_trsm_hyb_u_hi_row dcu_trsm_z_hyb_u_hi_row
#define dcu_trsm_hyb_n_lo_col dcu_trsm_z_hyb_n_lo_col
#define dcu_trsm_hyb_u_lo_col dcu_trsm_z_hyb_u_lo_col
#define dcu_trsm_hyb_n_hi_col dcu_trsm_z_hyb_n_hi_col
#define dcu_trsm_hyb_u_hi_col dcu_trsm_z_hyb_u_hi_col
#define dcu_trsm_hyb_n_lo_row_trans dcu_trsm_z_hyb_n_lo_row_trans
#define dcu_trsm_hyb_u_lo_row_trans dcu_trsm_z_hyb_u_lo_row_trans
#define dcu_trsm_hyb_n_hi_row_trans dcu_trsm_z_hyb_n_hi_row_trans
#define dcu_trsm_hyb_u_hi_row_trans dcu_trsm_z_hyb_u_hi_row_trans
#define dcu_trsm_hyb_n_lo_col_trans dcu_trsm_z_hyb_n_lo_col_trans
#define dcu_trsm_hyb_u_lo_col_trans dcu_trsm_z_hyb_u_lo_col_trans
#define dcu_trsm_hyb_n_hi_col_trans dcu_trsm_z_hyb_n_hi_col_trans
#define dcu_trsm_hyb_u_hi_col_trans dcu_trsm_z_hyb_u_hi_col_trans
#define dcu_diagsm_hyb_n_row dcu_diagsm_z_hyb_n_row
#define dcu_diagsm_hyb_u_row dcu_diagsm_z_hyb_u_row
#define dcu_diagsm_hyb_n_col dcu_diagsm_z_hyb_n_col
#define dcu_diagsm_hyb_u_col dcu_diagsm_z_hyb_u_col

// gebsr
#define dcu_set_value_gebsr dcu_set_value_z_gebsr
#define dcu_update_values_gebsr dcu_update_values_z_gebsr

#define dcu_add_gebsr dcu_add_z_gebsr
#define dcu_add_gebsr_trans dcu_add_z_gebsr_trans

#define dcu_gemv_gebsr dcu_gemv_z_gebsr
#define dcu_gemv_gebsr_trans dcu_gemv_z_gebsr_trans
#define dcu_symv_gebsr_n_lo dcu_symv_z_gebsr_n_lo
#define dcu_symv_gebsr_u_lo dcu_symv_z_gebsr_u_lo
#define dcu_symv_gebsr_n_hi dcu_symv_z_gebsr_n_hi
#define dcu_symv_gebsr_u_hi dcu_symv_z_gebsr_u_hi
#define dcu_trmv_gebsr_n_lo dcu_trmv_z_gebsr_n_lo
#define dcu_trmv_gebsr_u_lo dcu_trmv_z_gebsr_u_lo
#define dcu_trmv_gebsr_n_hi dcu_trmv_z_gebsr_n_hi
#define dcu_trmv_gebsr_u_hi dcu_trmv_z_gebsr_u_hi
#define dcu_trmv_gebsr_n_lo_trans dcu_trmv_z_gebsr_n_lo_trans
#define dcu_trmv_gebsr_u_lo_trans dcu_trmv_z_gebsr_u_lo_trans
#define dcu_trmv_gebsr_n_hi_trans dcu_trmv_z_gebsr_n_hi_trans
#define dcu_trmv_gebsr_u_hi_trans dcu_trmv_z_gebsr_u_hi_trans
#define dcu_diagmv_gebsr_n dcu_diagmv_z_gebsr_n
#define dcu_diagmv_gebsr_u dcu_diagmv_z_gebsr_u

#define dcu_gemm_gebsr dcu_gemm_z_gebsr
#define dcu_gemm_gebsr_transA dcu_gemm_z_gebsr_transA
#define dcu_gemm_gebsr_transB dcu_gemm_z_gebsr_transB
#define dcu_gemm_gebsr_transAB dcu_gemm_z_gebsr_transAB
#define dcu_symm_gebsr_n_lo_row dcu_symm_z_gebsr_n_lo_row
#define dcu_symm_gebsr_u_lo_row dcu_symm_z_gebsr_u_lo_row
#define dcu_symm_gebsr_n_hi_row dcu_symm_z_gebsr_n_hi_row
#define dcu_symm_gebsr_u_hi_row dcu_symm_z_gebsr_u_hi_row
#define dcu_symm_gebsr_n_lo_col dcu_symm_z_gebsr_n_lo_col
#define dcu_symm_gebsr_u_lo_col dcu_symm_z_gebsr_u_lo_col
#define dcu_symm_gebsr_n_hi_col dcu_symm_z_gebsr_n_hi_col
#define dcu_symm_gebsr_u_hi_col dcu_symm_z_gebsr_u_hi_col
#define dcu_trmm_gebsr_n_lo_row dcu_trmm_z_gebsr_n_lo_row
#define dcu_trmm_gebsr_u_lo_row dcu_trmm_z_gebsr_u_lo_row
#define dcu_trmm_gebsr_n_hi_row dcu_trmm_z_gebsr_n_hi_row
#define dcu_trmm_gebsr_u_hi_row dcu_trmm_z_gebsr_u_hi_row
#define dcu_trmm_gebsr_n_lo_col dcu_trmm_z_gebsr_n_lo_col
#define dcu_trmm_gebsr_u_lo_col dcu_trmm_z_gebsr_u_lo_col
#define dcu_trmm_gebsr_n_hi_col dcu_trmm_z_gebsr_n_hi_col
#define dcu_trmm_gebsr_u_hi_col dcu_trmm_z_gebsr_u_hi_col
#define dcu_trmm_gebsr_n_lo_row_trans dcu_trmm_z_gebsr_n_lo_row_trans
#define dcu_trmm_gebsr_u_lo_row_trans dcu_trmm_z_gebsr_u_lo_row_trans
#define dcu_trmm_gebsr_n_hi_row_trans dcu_trmm_z_gebsr_n_hi_row_trans
#define dcu_trmm_gebsr_u_hi_row_trans dcu_trmm_z_gebsr_u_hi_row_trans
#define dcu_trmm_gebsr_n_lo_col_trans dcu_trmm_z_gebsr_n_lo_col_trans
#define dcu_trmm_gebsr_u_lo_col_trans dcu_trmm_z_gebsr_u_lo_col_trans
#define dcu_trmm_gebsr_n_hi_col_trans dcu_trmm_z_gebsr_n_hi_col_trans
#define dcu_trmm_gebsr_u_hi_col_trans dcu_trmm_z_gebsr_u_hi_col_trans
#define dcu_diagmm_gebsr_n_row dcu_diagmm_z_gebsr_n_row
#define dcu_diagmm_gebsr_u_row dcu_diagmm_z_gebsr_u_row
#define dcu_diagmm_gebsr_n_col dcu_diagmm_z_gebsr_n_col
#define dcu_diagmm_gebsr_u_col dcu_diagmm_z_gebsr_u_col

#define dcu_spmmd_gebsr_row dcu_spmmd_z_gebsr_row
#define dcu_spmmd_gebsr_col dcu_spmmd_z_gebsr_col
#define dcu_spmmd_gebsr_row_trans dcu_spmmd_z_gebsr_row_trans
#define dcu_spmmd_gebsr_col_trans dcu_spmmd_z_gebsr_col_trans

#define dcu_spmm_gebsr dcu_spmm_z_gebsr
#define dcu_spmm_gebsr_trans dcu_spmm_z_gebsr_trans

#define dcu_trsv_gebsr_n_lo dcu_trsv_z_gebsr_n_lo
#define dcu_trsv_gebsr_u_lo dcu_trsv_z_gebsr_u_lo
#define dcu_trsv_gebsr_n_hi dcu_trsv_z_gebsr_n_hi
#define dcu_trsv_gebsr_u_hi dcu_trsv_z_gebsr_u_hi
#define dcu_trsv_gebsr_n_lo_trans dcu_trsv_z_gebsr_n_lo_trans
#define dcu_trsv_gebsr_u_lo_trans dcu_trsv_z_gebsr_u_lo_trans
#define dcu_trsv_gebsr_n_hi_trans dcu_trsv_z_gebsr_n_hi_trans
#define dcu_trsv_gebsr_u_hi_trans dcu_trsv_z_gebsr_u_hi_trans
#define dcu_diagsv_gebsr_n dcu_diagsv_z_gebsr_n
#define dcu_diagsv_gebsr_u dcu_diagsv_z_gebsr_u

#define dcu_trsm_gebsr_n_lo_row dcu_trsm_z_gebsr_n_lo_row
#define dcu_trsm_gebsr_u_lo_row dcu_trsm_z_gebsr_u_lo_row
#define dcu_trsm_gebsr_n_hi_row dcu_trsm_z_gebsr_n_hi_row
#define dcu_trsm_gebsr_u_hi_row dcu_trsm_z_gebsr_u_hi_row
#define dcu_trsm_gebsr_n_lo_col dcu_trsm_z_gebsr_n_lo_col
#define dcu_trsm_gebsr_u_lo_col dcu_trsm_z_gebsr_u_lo_col
#define dcu_trsm_gebsr_n_hi_col dcu_trsm_z_gebsr_n_hi_col
#define dcu_trsm_gebsr_u_hi_col dcu_trsm_z_gebsr_u_hi_col
#define dcu_trsm_gebsr_n_lo_row_trans dcu_trsm_z_gebsr_n_lo_row_trans
#define dcu_trsm_gebsr_u_lo_row_trans dcu_trsm_z_gebsr_u_lo_row_trans
#define dcu_trsm_gebsr_n_hi_row_trans dcu_trsm_z_gebsr_n_hi_row_trans
#define dcu_trsm_gebsr_u_hi_row_trans dcu_trsm_z_gebsr_u_hi_row_trans
#define dcu_trsm_gebsr_n_lo_col_trans dcu_trsm_z_gebsr_n_lo_col_trans
#define dcu_trsm_gebsr_u_lo_col_trans dcu_trsm_z_gebsr_u_lo_col_trans
#define dcu_trsm_gebsr_n_hi_col_trans dcu_trsm_z_gebsr_n_hi_col_trans
#define dcu_trsm_gebsr_u_hi_col_trans dcu_trsm_z_gebsr_u_hi_col_trans
#define dcu_diagsm_gebsr_n_row dcu_diagsm_z_gebsr_n_row
#define dcu_diagsm_gebsr_u_row dcu_diagsm_z_gebsr_u_row
#define dcu_diagsm_gebsr_n_col dcu_diagsm_z_gebsr_n_col
#define dcu_diagsm_gebsr_u_col dcu_diagsm_z_gebsr_u_col

#define dcu_geam_csr dcu_geam_z_csr

// csr5
#define dcu_gemv_csr5 dcu_gemv_z_csr5

// cooaos
#define dcu_gemv_cooaos dcu_gemv_z_cooaos

// sell_csigma
#define dcu_gemv_sell_csigma dcu_gemv_z_sell_csigma

// ellr
#define dcu_gemv_ellr dcu_gemv_z_ellr

// dia
#define dcu_gemv_dia dcu_gemv_z_dia

#pragma once

/**
 * @brief header for csr matrix related private interfaces
 */

#include "../spmat.h"

alphasparseStatus_t csr_s_order(spmat_csr_s_t *mat);
alphasparseStatus_t csr_d_order(spmat_csr_d_t *mat);
alphasparseStatus_t csr_c_order(spmat_csr_c_t *mat);
alphasparseStatus_t csr_z_order(spmat_csr_z_t *mat);

// !insert new2coo float before this line
alphasparseStatus_t destroy_csr(spmat_csr_s_t *A);
alphasparseStatus_t transpose_s_csr(const spmat_csr_s_t *s, spmat_csr_s_t **d);
alphasparseStatus_t convert_coo_s_csr(const spmat_csr_s_t *source,
                                       spmat_coo_s_t **dest);
alphasparseStatus_t convert_csr_s_csr(const spmat_csr_s_t *source,
                                       spmat_csr_s_t **dest);

// !insert new2coo float before this line
alphasparseStatus_t destroy_csr(spmat_csr_d_t *A);
alphasparseStatus_t transpose_d_csr(const spmat_csr_d_t *s, spmat_csr_d_t **d);
alphasparseStatus_t convert_coo_d_csr(const spmat_csr_d_t *source,
                                       spmat_coo_d_t **dest);
alphasparseStatus_t convert_csr_d_csr(const spmat_csr_d_t *source,
                                       spmat_csr_d_t **dest);

// !insert new2coo double before this line
alphasparseStatus_t destroy_csr(spmat_csr_c_t *A);
alphasparseStatus_t transpose_c_csr(const spmat_csr_c_t *s, spmat_csr_c_t **d);
alphasparseStatus_t transpose_conj_c_csr(const spmat_csr_c_t *s,
                                          spmat_csr_c_t **d);
alphasparseStatus_t convert_coo_c_csr(const spmat_csr_c_t *source,
                                       spmat_coo_c_t **dest);
alphasparseStatus_t convert_csr_c_csr(const spmat_csr_c_t *source,
                                       spmat_csr_c_t **dest);

// !insert new2coo Complex8 before this line
alphasparseStatus_t destroy_csr(spmat_csr_z_t *A);
alphasparseStatus_t transpose_z_csr(const spmat_csr_z_t *s, spmat_csr_z_t **d);
alphasparseStatus_t transpose_conj_z_csr(const spmat_csr_z_t *s,
                                          spmat_csr_z_t **d);
alphasparseStatus_t convert_coo_z_csr(const spmat_csr_z_t *source,
                                       spmat_coo_z_t **dest);
alphasparseStatus_t convert_csr_z_csr(const spmat_csr_z_t *source,
                                       spmat_csr_z_t **dest);

// !insert new2coo Complex16 before this line

alphasparseStatus_t create_gen_from_special_s_csr(
    const spmat_csr_s_t *source, spmat_csr_s_t **dest,
    struct alpha_matrix_descr descr_in);
alphasparseStatus_t create_gen_from_special_d_csr(
    const spmat_csr_d_t *source, spmat_csr_d_t **dest,
    struct alpha_matrix_descr descr_in);
alphasparseStatus_t create_gen_from_special_c_csr(
    const spmat_csr_c_t *source, spmat_csr_c_t **dest,
    struct alpha_matrix_descr descr_in);
alphasparseStatus_t create_gen_from_special_z_csr(
    const spmat_csr_z_t *source, spmat_csr_z_t **dest,
    struct alpha_matrix_descr descr_in);

#ifdef __CUDA__
alphasparseStatus_t csr_h_order(spmat_csr_h_t *mat);
alphasparseStatus_t destroy_h_csr(spmat_csr_h_t *A);
alphasparseStatus_t transpose_h_csr(const spmat_csr_h_t *s, spmat_csr_h_t **d);
alphasparseStatus_t convert_coo_h_csr(const spmat_csr_h_t *source,
                                       spmat_coo_h_t **dest);
alphasparseStatus_t convert_csr_h_csr(const spmat_csr_h_t *source,
                                       spmat_csr_h_t **dest);
alphasparseStatus_t create_gen_from_special_h_csr(
    const spmat_csr_h_t *source, spmat_csr_h_t **dest,
    struct alpha_matrix_descr descr_in);
#endif
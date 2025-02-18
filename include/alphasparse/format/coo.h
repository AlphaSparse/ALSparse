#pragma once

/**
 * @brief header for coo matrix related private interfaces
 */

#include "../spmat.h"

alphasparseStatus_t coo_s_order(spmat_coo_s_t *mat);
alphasparseStatus_t coo_d_order(spmat_coo_d_t *mat);
alphasparseStatus_t coo_c_order(spmat_coo_c_t *mat);
alphasparseStatus_t coo_z_order(spmat_coo_z_t *mat);

alphasparseStatus_t destroy_coo(spmat_coo_s_t *A);
alphasparseStatus_t transpose_s_coo(const spmat_coo_s_t *s, spmat_coo_s_t **d);
// template <typename I, typename J, typename T, typename U>
// alphasparseStatus_t convert_csr_coo(const T *source, U **dest);
// !insert new2coo float before this line

alphasparseStatus_t destroy_coo(spmat_coo_d_t *A);
alphasparseStatus_t transpose_d_coo(const spmat_coo_d_t *s, spmat_coo_d_t **d);
;
// !insert new2coo double before this line

alphasparseStatus_t destroy_coo(spmat_coo_c_t *A);
alphasparseStatus_t transpose_c_coo(const spmat_coo_c_t *s, spmat_coo_c_t **d);
alphasparseStatus_t transpose_conj_c_coo(const spmat_coo_c_t *s,
                                          spmat_coo_c_t **d);

// !insert new2coo Complex8 before this line
alphasparseStatus_t destroy_coo(spmat_coo_z_t *A);
alphasparseStatus_t transpose_z_coo(const spmat_coo_z_t *s, spmat_coo_z_t **d);
alphasparseStatus_t transpose_conj_z_coo(const spmat_coo_z_t *s,
                                          spmat_coo_z_t **d);

// !insert new2coo Complex16 before this line
alphasparseStatus_t create_gen_from_special_s_coo(
    const spmat_coo_s_t *source, spmat_coo_s_t **dest,
    struct alpha_matrix_descr descr_in);
alphasparseStatus_t create_gen_from_special_d_coo(
    const spmat_coo_d_t *source, spmat_coo_d_t **dest,
    struct alpha_matrix_descr descr_in);
alphasparseStatus_t create_gen_from_special_c_coo(
    const spmat_coo_c_t *source, spmat_coo_c_t **dest,
    struct alpha_matrix_descr descr_in);
alphasparseStatus_t create_gen_from_special_z_coo(
    const spmat_coo_z_t *source, spmat_coo_z_t **dest,
    struct alpha_matrix_descr descr_in);

#ifdef __CUDA__
alphasparseStatus_t coo_h_order(spmat_coo_h_t *mat);
alphasparseStatus_t destroy_h_coo(spmat_coo_h_t *A);
alphasparseStatus_t transpose_h_coo(const spmat_coo_h_t *s, spmat_coo_h_t **d);
alphasparseStatus_t convert_csr_h_coo(const spmat_coo_h_t *source,
                                       spmat_csr_h_t **dest);
alphasparseStatus_t create_gen_from_special_h_coo(
    const spmat_coo_h_t *source, spmat_coo_h_t **dest,
    struct alpha_matrix_descr descr_in);
#endif
set(alphasparse_source
  kernel/generic/alphasparse_axpby.cu
  kernel/generic/alphasparse_gather.cu
  kernel/generic/alphasparse_rot.cu
  kernel/generic/alphasparse_scatter.cu
  kernel/generic/alphasparse_spvv.cu
  # kernel/extra/alphasparse_csrgeam2.cu
  kernel/level2/alphasparse_spmv.cu
  # kernel/level2/alphasparse_bsrmv.cu
  # kernel/level2/alphasparse_bsrxmv.cu
  # kernel/level2/alphasparse_bsrsv2.cu
  # kernel/level2/alphasparse_spsv.cu
  # kernel/level2/alphasparse_gemvi.cu
  # kernel/level3/alphasparse_bsrmm.cu
  # kernel/level3/alphasparse_spmm.cu
  # kernel/level3/alphasparse_bsrsm.cu
  # kernel/level3/alphasparse_spsm.cu
  kernel/level3/alphasparse_sddmm.cu
  kernel/level3/alphasparse_spgemm.cu
  # # kernel/level3/alphasparse_spgemm_reuse.cu
  # kernel/precond/alphasparse_csric02.cu
  # kernel/precond/alphasparse_csrilu02.cu
  # kernel/precond/alphasparse_bsrilu02.cu
  # kernel/precond/alphasparse_bsric02.cu
  # kernel/precond/alphasparse_gpsv.cu
  # kernel/reordering/alphasparse_csrcolor.cu
  # kernel/precond/alphasparse_gtsv2.cu
  # kernel/precond/alphasparse_gtsv2_nopivot.cu
  # kernel/precond/alphasparse_gtsv2_strided_batch.cu
  # kernel/precond/alphasparse_gtsv2_interleaved_batch.cu
  PARENT_SCOPE
)

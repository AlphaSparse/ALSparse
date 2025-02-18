#ifdef __CUDA__
#include "kernel/generic/axpby_device.cuh"
#include "kernel/generic/gather_device.cuh"
#include "kernel/generic/rot_device.cuh"
#include "kernel/generic/spvv_device.cuh"
#include "kernel/generic/scatter_device.cuh"
#endif
#include "kernel/kernel_csr.h"

#include "kernel/kernel_s.h"
#include "kernel/kernel_coo_s.h"
#include "kernel/kernel_csc_s.h"
#include "kernel/kernel_bsr_s.h"
#include "kernel/kernel_sky_s.h"
#include "kernel/kernel_dia_s.h"
#include "kernel/kernel_ell_s.h"
#include "kernel/kernel_gebsr_s.h"
#include "kernel/kernel_sell_c_sigma_s.h"

#include "kernel/kernel_d.h"
#include "kernel/kernel_coo_d.h"
#include "kernel/kernel_csc_d.h"
#include "kernel/kernel_bsr_d.h"
#include "kernel/kernel_sky_d.h"
#include "kernel/kernel_dia_d.h"
#include "kernel/kernel_ell_d.h"
#include "kernel/kernel_gebsr_d.h"
#include "kernel/kernel_sell_c_sigma_d.h"

#include "kernel/kernel_c.h"
#include "kernel/kernel_coo_c.h"
#include "kernel/kernel_csc_c.h"
#include "kernel/kernel_bsr_c.h"
#include "kernel/kernel_sky_c.h"
#include "kernel/kernel_dia_c.h"
#include "kernel/kernel_ell_c.h"
#include "kernel/kernel_gebsr_c.h"
#include "kernel/kernel_sell_c_sigma_c.h"

#include "kernel/kernel_z.h"
#include "kernel/kernel_coo_z.h"
#include "kernel/kernel_csc_z.h"
#include "kernel/kernel_bsr_z.h"
#include "kernel/kernel_sky_z.h"
#include "kernel/kernel_dia_z.h"
#include "kernel/kernel_ell_z.h"
#include "kernel/kernel_gebsr_z.h"
#include "kernel/kernel_sell_c_sigma_z.h"

// #ifndef COMPLEX
// #ifndef DOUBLE
// #include "kernel/def_s.h"
// #else
// #include "kernel/def_d.h"
// #endif
// #else
// #ifndef DOUBLE
// #include "kernel/def_c.h"
// #else
// #include "kernel/def_z.h"
// #endif
// #endif


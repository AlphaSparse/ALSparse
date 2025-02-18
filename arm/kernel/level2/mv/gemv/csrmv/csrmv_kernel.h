void gemv_unroll_f32(const float alpha,   // s0
                     const float beta,    // s1
                     const int num_rows,  // x0
                     const int* Ap,       // row_start //x1
                     const int* Aj,       // col_idx   //x2
                     const float* Ax,     // value   //x3
                     const float* x,      // x4
                     float* y);            // x5
extern "C"  void gemv_3insert_sload_vhadd_f32(const float alpha,   // s0
                     const float beta,    // s1
                     const int num_rows,  // x0
                     const int* Ap,       // row_start //x1
                     const int* Aj,       // col_idx   //x2
                     const float* Ax,     // value   //x3
                     const float* x,      // x4
                     float* y);            // x5
void gemv_3insert_sload_revadd_f32(const float alpha,   // s0
                     const float beta,    // s1
                     const int num_rows,  // x0
                     const int* Ap,       // row_start //x1
                     const int* Aj,       // col_idx   //x2
                     const float* Ax,     // value   //x3
                     const float* x,      // x4
                     float* y);            // x5

void gemv_4insert_sload_vhadd_f32(const float alpha,   // s0
                     const float beta,    // s1
                     const int num_rows,  // x0
                     const int* Ap,       // row_start //x1
                     const int* Aj,       // col_idx   //x2
                     const float* Ax,     // value   //x3
                     const float* x,      // x4
                     float* y);            // x5
void gemv_4insert_sload_revadd_f32(const float alpha,   // s0
                     const float beta,    // s1
                     const int num_rows,  // x0
                     const int* Ap,       // row_start //x1
                     const int* Aj,       // col_idx   //x2
                     const float* Ax,     // value   //x3
                     const float* x,      // x4
                     float* y);            // x5

void gemv_unroll16_3insert_sload_revadd_f32(const float alpha,   // s0
                     const float beta,    // s1
                     const int num_rows,  // x0
                     const int* Ap,       // row_start //x1
                     const int* Aj,       // col_idx   //x2
                     const float* Ax,     // value   //x3
                     const float* x,      // x4
                     float* y);            // x5


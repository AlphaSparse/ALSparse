#include "include/norm.h"
 #include <malloc.h>
double CalEpisilon_d()
{
    static double eps;
    const double half = 0.5;
    volatile double maxval, fl = 0.5;
    do
    {
        eps = fl;
        fl *= half;
        maxval = 1.0 + fl; 
    } while (maxval != 1.0);
    return (eps * 10);    
}

float CalEpisilon_s()
{
    static float eps;
    const float half = 0.5;
    volatile float maxval, fl = 0.5;
    do
    {
        eps = fl;
        fl *= half;
        maxval = 1.0 + fl; 
    } while (maxval != 1.0);
    return (eps * 10);    
}

float InfiniteNorm_s(const ALPHA_INT n, const float *xa, const float *xb)
{
    float amax = 0.f;
    for (ALPHA_INT i = 0; i < n; i++)
    {
        if(xb != NULL)
            {if(fabs(xa[i] - xb[i]) > amax) amax = fabs(xa[i] - xb[i]);}
        else amax = fabs(xa[i]) > amax ? fabs(xa[i]) : amax;
    }
    return amax;
}

float GeNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const float *val)
{
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*n);
    memset(sum, '\0', sizeof(float)*n);
    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        sum[col_index[i]] += fabs(val[i]);
    }
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float TrSyNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const float *val, struct alpha_matrix_descr descr)
{
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*n);
    memset(sum, '\0', sizeof(float)*n);
    if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
    {
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] > col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i]);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                    sum[row_index[i]] += fabs(val[i]);
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    sum[col_index[i]] += fabs(val[i]);
                else
                    sum[col_index[i]] += 1.0f;
            }
            else  continue;
        }
    }
    else
    {// descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER 
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] < col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i]);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                    sum[row_index[i]] += fabs(val[i]);
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    sum[col_index[i]] += fabs(val[i]);
                else
                    sum[col_index[i]] += 1.0f;
            }
            else  continue;
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float DiNorm1_s(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const float *val, struct alpha_matrix_descr descr)
{
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*n);
    memset(sum, '\0', sizeof(float)*n);

    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        if(row_index[i] == col_index[i])
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                sum[col_index[i]] += fabs(val[i]);
            else
                sum[col_index[i]] += 1.0f;
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float DesNorm1_s(const ALPHA_INT rows, const ALPHA_INT cols, const float *val, const ALPHA_INT ldv, const alphasparse_layout_t layout)
{//compute the norm 1 of a dense matrix, parameters(rows, cols, dense matrix, leading dimension)
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*cols);
    memset(sum, '\0', sizeof(float)*cols);
    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            for(ALPHA_INT c = 0; c < cols; c++)
            {
                sum[c] += fabs(val[index2(r, c, ldv)]);
            }
        }
    }
    else
    {//layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR        
        for(ALPHA_INT c = 0; c < cols; c++)
        {
            for(ALPHA_INT r = 0; r < rows; r++)
            {
                sum[c] += fabs(val[index2(c, r, ldv)]);
            }
        }
    }
    
    for(ALPHA_INT i = 0; i < cols; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float DesDiffNorm1_s(const ALPHA_INT rows, const ALPHA_INT cols, const float *xa, const ALPHA_INT lda, const float *xb, const ALPHA_INT ldb, alphasparse_layout_t layout)
{//compute the infinite norm of two dense matrix 
    float amax = 0.f;
    for(ALPHA_INT c = 0; c < cols; c++)
    {
        float t = 0.0f;
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                t += fabsf(xa[index2(r, c, lda)] - xb[index2(r, c, ldb)]);
            else
                t += fabsf(xa[index2(c, r, lda)] - xb[index2(c, r, ldb)]);
        }
        amax = alpha_max(t, amax); 
    }

    return amax;
}

double InfiniteNorm_d(const ALPHA_INT n, const double *xa, const double *xb)
{
    double amax = 0.f;
    for (ALPHA_INT i = 0; i < n; i++)
    {
        if(xb != NULL)
            {if(fabs(xa[i] - xb[i]) > amax) amax = fabs(xa[i] - xb[i]);}
        else amax = fabs(xa[i]) > amax ? fabs(xa[i]) : amax;
    }
    return amax;
}

double GeNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const double *val)
{
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*n);
    memset(sum, '\0', sizeof(double)*n);
    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        sum[col_index[i]] += fabs(val[i]);
    }
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double TrSyNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const double *val, struct alpha_matrix_descr descr)
{
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*n);
    memset(sum, '\0', sizeof(double)*n);
    if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
    {
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] > col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i]);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                    sum[row_index[i]] += fabs(val[i]);
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    sum[col_index[i]] += fabs(val[i]);
                else
                    sum[col_index[i]] += 1.0f;
            }
            else  continue;
        }
    }
    else
    {// descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER 
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] < col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i]);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                    sum[row_index[i]] += fabs(val[i]);
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    sum[col_index[i]] += fabs(val[i]);
                else
                    sum[col_index[i]] += 1.0f;
            }
            else  continue;
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double DiNorm1_d(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const double *val, struct alpha_matrix_descr descr)
{
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*n);
    memset(sum, '\0', sizeof(double)*n);

    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        if(row_index[i] == col_index[i])
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                sum[col_index[i]] += fabs(val[i]);
            else
                sum[col_index[i]] += 1.0f;
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double DesNorm1_d(ALPHA_INT rows, ALPHA_INT cols, const double *val, ALPHA_INT ldv, alphasparse_layout_t layout)
{//compute the norm 1 of a dense matrix, parameters(rows, cols, dense matrix, leading dimension)
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*cols);
    memset(sum, '\0', sizeof(double)*cols);
    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            for(ALPHA_INT c = 0; c < cols; c++)
            {
                sum[c] += fabs(val[index2(r, c, ldv)]);
            }
        }
    }
    else
    {//layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR        
        for(ALPHA_INT c = 0; c < cols; c++)
        {
            for(ALPHA_INT r = 0; r < rows; r++)
            {
                sum[c] += fabs(val[index2(c, r, ldv)]);
            }
        }
    }
    
    for(ALPHA_INT i = 0; i < cols; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double DesDiffNorm1_d(const ALPHA_INT rows, const ALPHA_INT cols, const double *xa, const ALPHA_INT lda, const double *xb, const ALPHA_INT ldb, alphasparse_layout_t layout)
{//compute the infinite norm of two dense matrix 
    double amax = 0.f;
    for(ALPHA_INT c = 0; c < cols; c++)
    {
        double t = 0.0f;
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                t += fabs(xa[index2(r, c, lda)] - xb[index2(r, c, ldb)]);
            else
                t += fabs(xa[index2(c, r, lda)] - xb[index2(c, r, ldb)]);
        }
        amax = t > amax ? t : amax; 
    }

    return amax;
}

float InfiniteNorm_c(const ALPHA_INT n, const ALPHA_Complex8 *xa, const ALPHA_Complex8 *xb)
{
    float amax = 0.f;
    for (ALPHA_INT i = 0; i < n; i++)
    {
        if(xb != NULL)
        {
            float fr = fabs(xa[i].real - xb[i].real);
            float fi = fabs(xa[i].imag - xb[i].imag);
            amax = fr + fi > amax ? fr + fi : amax;
        }
        else amax = fabs(xa[i].real) + fabs(xa[i].imag) > amax ? fabs(xa[i].real) + fabs(xa[i].imag) : amax;
    }
    return amax;
}

float GeNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const ALPHA_Complex8 *val)
{
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*n);
    memset(sum, '\0', sizeof(float)*n);
    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        sum[col_index[i]] += fabs(val[i].real);
        sum[col_index[i]] += fabs(val[i].imag);
    }
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float TrSyNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex8 *val, struct alpha_matrix_descr descr)
{
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*n);
    memset(sum, '\0', sizeof(float)*n);
    if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
    {
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] > col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i].real);
                sum[col_index[i]] += fabs(val[i].imag);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                {
                    sum[row_index[i]] += fabs(val[i].real);
                    sum[row_index[i]] += fabs(val[i].imag);
                }
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    sum[col_index[i]] += fabs(val[i].real);
                    sum[col_index[i]] += fabs(val[i].imag);
                }
                else
                {
                    sum[col_index[i]] += 1.0f;
                    sum[col_index[i]] += 1.0f;
                }
            }
            else  continue;
        }
    }
    else
    {// descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER 
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] < col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i].real);
                sum[col_index[i]] += fabs(val[i].imag);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                {
                    sum[row_index[i]] += fabs(val[i].real);
                    sum[row_index[i]] += fabs(val[i].imag);
                }
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    sum[col_index[i]] += fabs(val[i].real);
                    sum[col_index[i]] += fabs(val[i].imag);
                }
                else
                {
                    sum[col_index[i]] += 1.0f;
                    sum[col_index[i]] += 1.0f;
                }
            }
            else  continue;
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float DiNorm1_c(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex8 *val, struct alpha_matrix_descr descr)
{
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*n);
    memset(sum, '\0', sizeof(float)*n);

    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        if(row_index[i] == col_index[i])
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                sum[col_index[i]] += fabs(val[i].real);
                sum[col_index[i]] += fabs(val[i].imag);
            }
            else
            {
                sum[col_index[i]] += 1.0f;
                sum[col_index[i]] += 1.0f;
            }
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float DesNorm1_c(ALPHA_INT rows, ALPHA_INT cols, const ALPHA_Complex8 *val, ALPHA_INT ldv, alphasparse_layout_t layout)
{//compute the norm 1 of a dense matrix, parameters(rows, cols, dense matrix, leading dimension)
    float norm1 = 0.f;
    float * sum = (float *)malloc(sizeof(float)*cols);
    memset(sum, '\0', sizeof(float)*cols);
    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            for(ALPHA_INT c = 0; c < cols; c++)
            {
                sum[c] += fabs(val[index2(r, c, ldv)].real);
                sum[c] += fabs(val[index2(r, c, ldv)].imag);
            }
        }
    }
    else
    {//layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR        
        for(ALPHA_INT c = 0; c < cols; c++)
        {
            for(ALPHA_INT r = 0; r < rows; r++)
            {
                sum[c] += fabs(val[index2(c, r, ldv)].real);
                sum[c] += fabs(val[index2(c, r, ldv)].imag);
            }
        }
    }
    
    for(ALPHA_INT i = 0; i < cols; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

float DesDiffNorm1_c(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_Complex8 *xa, const ALPHA_INT lda, const ALPHA_Complex8 *xb, const ALPHA_INT ldb, alphasparse_layout_t layout)
{//compute the infinite norm of two dense matrix 
    float amax = 0.f;
    for(ALPHA_INT c = 0; c < cols; c++)
    {
        float t = 0.0f;
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
            {
                t += fabs(xa[index2(r, c, lda)].real - xb[index2(r, c, ldb)].real);
                t += fabs(xa[index2(r, c, lda)].imag - xb[index2(r, c, ldb)].imag);
            }
            else
            {
                t += fabs(xa[index2(c, r, lda)].real - xb[index2(c, r, ldb)].real);
                t += fabs(xa[index2(c, r, lda)].imag - xb[index2(c, r, ldb)].imag);
            }
        }
        amax = t > amax ? t : amax; 
    }

    return amax;
}

double InfiniteNorm_z(const ALPHA_INT n, const ALPHA_Complex16 *xa, const ALPHA_Complex16 *xb)
{
    double amax = 0.f;
    for (ALPHA_INT i = 0; i < n; i++)
    {
        if(xb != NULL)
        {
            double fr = fabs(xa[i].real - xb[i].real);
            double fi = fabs(xa[i].imag - xb[i].imag);
            amax = fr + fi > amax ? fr + fi : amax;
        }
        else amax = fabs(xa[i].real) + fabs(xa[i].imag) > amax ? fabs(xa[i].real) + fabs(xa[i].imag) : amax;
    }
    return amax;
}

double GeNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * col_index, const ALPHA_Complex16 *val)
{
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*n);
    memset(sum, '\0', sizeof(double)*n);
    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        sum[col_index[i]] += fabs(val[i].real);
        sum[col_index[i]] += fabs(val[i].imag);
    }
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double TrSyNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex16 *val, struct alpha_matrix_descr descr)
{
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*n);
    memset(sum, '\0', sizeof(double)*n);
    if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
    {
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] > col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i].real);
                sum[col_index[i]] += fabs(val[i].imag);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                {
                    sum[row_index[i]] += fabs(val[i].real);
                    sum[row_index[i]] += fabs(val[i].imag);
                }
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    sum[col_index[i]] += fabs(val[i].real);
                    sum[col_index[i]] += fabs(val[i].imag);
                }
                else
                {
                    sum[col_index[i]] += 1.0f;
                    sum[col_index[i]] += 1.0f;
                }
            }
            else  continue;
        }
    }
    else
    {// descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER 
        for(ALPHA_INT i = 0; i < nnz; i++)
        {
            if(row_index[i] < col_index[i]) 
            {
                sum[col_index[i]] += fabs(val[i].real);
                sum[col_index[i]] += fabs(val[i].imag);
                if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
                {
                    sum[row_index[i]] += fabs(val[i].real);
                    sum[row_index[i]] += fabs(val[i].imag);
                }
            }
            else if(row_index[i] == col_index[i])
            {
                if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    sum[col_index[i]] += fabs(val[i].real);
                    sum[col_index[i]] += fabs(val[i].imag);
                }
                else
                {
                    sum[col_index[i]] += 1.0f;
                    sum[col_index[i]] += 1.0f;
                }
            }
            else  continue;
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double DiNorm1_z(ALPHA_INT n, ALPHA_INT nnz, const ALPHA_INT * row_index, const ALPHA_INT * col_index, const ALPHA_Complex16 *val, struct alpha_matrix_descr descr)
{
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*n);
    memset(sum, '\0', sizeof(double)*n);

    for(ALPHA_INT i = 0; i < nnz; i++)
    {
        if(row_index[i] == col_index[i])
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                sum[col_index[i]] += fabs(val[i].real);
                sum[col_index[i]] += fabs(val[i].imag);
            }
            else
            {
                sum[col_index[i]] += 1.0f;
                sum[col_index[i]] += 1.0f;
            }
        }
    }
    
    for(ALPHA_INT i = 0; i < n; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double DesNorm1_z(ALPHA_INT rows, ALPHA_INT cols, const ALPHA_Complex16 *val, ALPHA_INT ldv, alphasparse_layout_t layout)
{//compute the norm 1 of a dense matrix, parameters(rows, cols, dense matrix, leading dimension)
    double norm1 = 0.f;
    double * sum = (double *)malloc(sizeof(double)*cols);
    memset(sum, '\0', sizeof(double)*cols);
    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            for(ALPHA_INT c = 0; c < cols; c++)
            {
                sum[c] += fabs(val[index2(r, c, ldv)].real);
                sum[c] += fabs(val[index2(r, c, ldv)].imag);
            }
        }
    }
    else
    {//layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR        
        for(ALPHA_INT c = 0; c < cols; c++)
        {
            for(ALPHA_INT r = 0; r < rows; r++)
            {
                sum[c] += fabs(val[index2(c, r, ldv)].real);
                sum[c] += fabs(val[index2(c, r, ldv)].imag);
            }
        }
    }
    
    for(ALPHA_INT i = 0; i < cols; i++)
    {
        if(sum[i] > norm1) norm1 = sum[i];
    }
    free(sum);
    return norm1;
}

double DesDiffNorm1_z(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_Complex16 *xa, const ALPHA_INT lda, const ALPHA_Complex16 *xb, const ALPHA_INT ldb, alphasparse_layout_t layout)
{//compute the infinite norm of two dense matrix 
    double amax = 0.f;
    for(ALPHA_INT c = 0; c < cols; c++)
    {
        double t = 0.0f;
        for(ALPHA_INT r = 0; r < rows; r++)
        {
            if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
            {
                t += fabs(xa[index2(r, c, lda)].real - xb[index2(r, c, ldb)].real);
                t += fabs(xa[index2(r, c, lda)].imag - xb[index2(r, c, ldb)].imag);
            }
            else
            {
                t += fabs(xa[index2(c, r, lda)].real - xb[index2(c, r, ldb)].real);
                t += fabs(xa[index2(c, r, lda)].imag - xb[index2(c, r, ldb)].imag);
            }
        }
        amax = t > amax ? t : amax; 
    }

    return amax;
}
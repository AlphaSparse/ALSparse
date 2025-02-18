#pragma once

#include "types.h"
// #include "complex_compute.h"
// #include "real_compute.h"

// #define ALPHA_ZERO 0.
// #define ALPHA_ONE  1.

#ifdef __DCU__
#define ALPHA_DEVICE __device__
#else
#define ALPHA_DEVICE 
#endif

float inline ALPHA_DEVICE alpha_setone(float a)
{
    return 1.0f;
}
double inline ALPHA_DEVICE alpha_setone(double a)
{
    return 1.0;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_setone(ALPHA_Complex8 a)
{
    ALPHA_Complex8 z;
    (z).real = 1.f; 
    (z).imag = 0.f; 
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_setone(ALPHA_Complex16 a)
{
    ALPHA_Complex16 z;
    (z).real = 1.; 
    (z).imag = 0.; 
    return z;
}

float inline ALPHA_DEVICE alpha_setzero(float a)
{
    return 0.0f;
}
double inline ALPHA_DEVICE alpha_setzero(double a)
{
    return 0.0;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_setzero(ALPHA_Complex8 a)
{
    ALPHA_Complex8 z;
    (z).real = 0.f; 
    (z).imag = 0.f; 
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_setzero(ALPHA_Complex16 a)
{
    ALPHA_Complex16 z;
    (z).real = 0.; 
    (z).imag = 0.; 
    return z;
}

bool inline ALPHA_DEVICE alpha_iszero(float val)
{
    return (0.0f <= (val) && (val) <= 1e-10) || (-1e-10 <= (val) && (val) <= 0.0f);
}
bool inline ALPHA_DEVICE alpha_iszero(double val)
{
    return (0. <= (val) && (val) <= 1e-10) || (-1e-10 <= (val) && (val) <= 0.);
}
bool inline ALPHA_DEVICE alpha_iszero(ALPHA_Complex8 val)
{
    return alpha_iszero((val).real) && alpha_iszero((val).imag);
}
bool inline ALPHA_DEVICE alpha_iszero(ALPHA_Complex16 val)
{
    return alpha_iszero((val).real) && alpha_iszero((val).imag);
}

bool inline ALPHA_DEVICE alpha_isone(float val)
{
    return alpha_iszero(val-1.f);
}
bool inline ALPHA_DEVICE alpha_isone(double val)
{
    return alpha_iszero(val-1.);
}
bool inline ALPHA_DEVICE alpha_isone(ALPHA_Complex8 val)
{
    return alpha_iszero((val).real-1.f) && alpha_iszero((val).imag);
}
bool inline ALPHA_DEVICE alpha_isone(ALPHA_Complex16 val)
{
    return alpha_iszero((val).real-1.0) && alpha_iszero((val).imag);
}

float inline ALPHA_DEVICE alpha_add(float a, float b)
{
    return a + b;
}
double inline ALPHA_DEVICE alpha_add(double a, double b)
{
    return a + b;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_add(ALPHA_Complex8 a, ALPHA_Complex8 b)
{
    ALPHA_Complex8 z;
    (z).real = (a).real + (b).real; 
    (z).imag = (a).imag + (b).imag; 
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_add(ALPHA_Complex16 a, ALPHA_Complex16 b)
{
    ALPHA_Complex16 z;
    (z).real = (a).real + (b).real; 
    (z).imag = (a).imag + (b).imag; 
    return z;
}

float inline ALPHA_DEVICE alpha_sub(float a, float b)
{
    return a - b;
}
double inline ALPHA_DEVICE alpha_sub(double a, double b)
{
    return a - b;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_sub(ALPHA_Complex8 a, ALPHA_Complex8 b)
{
    ALPHA_Complex8 z;
    (z).real = (a).real - (b).real; 
    (z).imag = (a).imag - (b).imag; 
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_sub(ALPHA_Complex16 a, ALPHA_Complex16 b)
{
    ALPHA_Complex16 z;
    (z).real = (a).real - (b).real; 
    (z).imag = (a).imag - (b).imag; 
    return z;
}

float inline ALPHA_DEVICE alpha_mul(float a, float b)
{
    return a * b;
}
double inline ALPHA_DEVICE alpha_mul(double a, double b)
{
    return a * b;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_mul(ALPHA_Complex8 a, ALPHA_Complex8 b)
{
    ALPHA_Complex8 z;
    float _REAL = (a).real * (b).real - (a).imag * (b).imag;        
    float _IMAG = (a).imag * (b).real + (a).real * (b).imag;        
    (z).real = _REAL;                                                   
    (z).imag = _IMAG;
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_mul(ALPHA_Complex16 a, ALPHA_Complex16 b)
{
    ALPHA_Complex16 z;
    double _REAL = (a).real * (b).real - (a).imag * (b).imag;        
    double _IMAG = (a).imag * (b).real + (a).real * (b).imag;        
    (z).real = _REAL;                                                   
    (z).imag = _IMAG;
    return z;
}

float inline ALPHA_DEVICE alpha_div(float a, float b)
{
    return a / b;
}
double inline ALPHA_DEVICE alpha_div(double a, double b)
{
    return a / b;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_div(ALPHA_Complex8 a, ALPHA_Complex8 b)
{
    ALPHA_Complex8 z;
    float _AC = a.real * b.real;   
    float _BD = a.imag * b.imag;   
    float _BC = a.imag * b.real;   
    float _AD = a.real * b.imag;   
    float _C2 = b.real * b.real; 
    float _D2 = b.imag * b.imag; 
    z.real = (_AC + _BD) / (_C2 + _D2);             
    z.imag = (_BC - _AD) / (_C2 + _D2);                
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_div(ALPHA_Complex16 a, ALPHA_Complex16 b)
{
    ALPHA_Complex16 z;
    double _AC = a.real * b.real;   
    double _BD = a.imag * b.imag;   
    double _BC = a.imag * b.real;   
    double _AD = a.real * b.imag;   
    double _C2 = b.real * b.real; 
    double _D2 = b.imag * b.imag; 
    z.real = (_AC + _BD) / (_C2 + _D2);             
    z.imag = (_BC - _AD) / (_C2 + _D2);  
    return z;
}

float inline ALPHA_DEVICE alpha_madd(float a, float b, float c)
{
    return a * b + c;
}
double inline ALPHA_DEVICE alpha_madd(double a, double b, double c)
{
    return a * b + c;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_madd(ALPHA_Complex8 a, ALPHA_Complex8 b, ALPHA_Complex8 c)
{
    ALPHA_Complex8 z;
    float _B = (a).real * (b).imag;                                          
    float _C = (a).imag * (b).real;                                          
    (z).real = ((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C + (c).real; 
    (z).imag = _B + _C + (c).imag;                                                 
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_madd(ALPHA_Complex16 a, ALPHA_Complex16 b, ALPHA_Complex16 c)
{
    ALPHA_Complex16 z;
    double _B = (a).real * (b).imag;                                          
    double _C = (a).imag * (b).real;                                          
    (z).real = ((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C + (c).real; 
    (z).imag = _B + _C + (c).imag;  
    return z;
}

float inline ALPHA_DEVICE alpha_madde(float a, float b, float c)
{
    return a + b * c;
}
double inline ALPHA_DEVICE alpha_madde(double a, double b, double c)
{
    return a + b * c;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_madde(ALPHA_Complex8 a, ALPHA_Complex8 b, ALPHA_Complex8 c)
{
    ALPHA_Complex8 z;
    float _B = (b).real * (c).imag;                                          
    float _C = (b).imag * (c).real;                                          
    (z).real = ((b).real + (b).imag) * ((c).real - (c).imag) + _B - _C + (a).real; 
    (z).imag = _B + _C + (a).imag;                                                 
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_madde(ALPHA_Complex16 a, ALPHA_Complex16 b, ALPHA_Complex16 c)
{
    ALPHA_Complex16 z;
    double _B = (b).real * (c).imag;                                          
    double _C = (b).imag * (c).real;                                          
    (z).real = ((b).real + (b).imag) * ((c).real - (c).imag) + _B - _C + (a).real; 
    (z).imag = _B + _C + (a).imag;                                                 
    return z;
}

float inline ALPHA_DEVICE alpha_msub(float a, float b, float c)
{
    return c - a * b;
}
double inline ALPHA_DEVICE alpha_msub(double a, double b, double c)
{
    return c - a * b;
}
ALPHA_Complex8 inline ALPHA_DEVICE alpha_msub(ALPHA_Complex8 a, ALPHA_Complex8 b, ALPHA_Complex8 c)
{
    ALPHA_Complex8 z;
    float _B = (a).real * (b).imag;                                          
    float _C = (a).imag * (b).real;                                          
    (z).real = (c).real - (((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C); 
    (z).imag = (c).imag - (_B + _C);                                                 
    return z;
}
ALPHA_Complex16 inline ALPHA_DEVICE alpha_msub(ALPHA_Complex16 a, ALPHA_Complex16 b, ALPHA_Complex16 c)
{
    ALPHA_Complex16 z;
    double _B = (a).real * (b).imag;                                          
    double _C = (a).imag * (b).real;                                          
    (z).real = (c).real - (((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C); 
    (z).imag = (c).imag - (_B + _C);      
    return z;
}


float inline ALPHA_DEVICE alpha_msube(float d, float a, float b)
{
    return d - (a * b);
} 

double inline ALPHA_DEVICE alpha_msube(double d, double a, double b)
{
    return d - (a * b);
} 

ALPHA_Complex8 inline ALPHA_DEVICE alpha_msube(ALPHA_Complex8 d, ALPHA_Complex8 a, ALPHA_Complex8 b)                                                   
{                                                                        
    float _B = (a).real * (b).imag;                                
    float _C = (a).imag * (b).real;  
    ALPHA_Complex8 z;                              
    (z).real = (d).real - (((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C); 
    (z).imag = (d).imag - (_B + _C);  
    return z;                                               
}

ALPHA_Complex16 inline ALPHA_DEVICE alpha_msube(ALPHA_Complex16 d, ALPHA_Complex16 a, ALPHA_Complex16 b)                                                   
{                                                                        
    double _B = (a).real * (b).imag;                                
    double _C = (a).imag * (b).real;  
    ALPHA_Complex16 z;                              
    (z).real = (d).real - (((a).real + (a).imag) * ((b).real - (b).imag) + _B - _C); 
    (z).imag = (d).imag - (_B + _C);  
    return z;                                               
}

float inline ALPHA_DEVICE cmp_conj(float a)        
{       
    return -a;
}

double inline ALPHA_DEVICE cmp_conj(double a)        
{       
    return -a;
}

ALPHA_Complex8 inline ALPHA_DEVICE cmp_conj(ALPHA_Complex8 a)        
{       
    ALPHA_Complex8 b;                  
    (b).real = (a).real;  
    (b).imag = -(a).imag; 
    return b;
}

ALPHA_Complex16 inline ALPHA_DEVICE cmp_conj(ALPHA_Complex16 a)        
{       
    ALPHA_Complex16 b;                  
    (b).real = (a).real;  
    (b).imag = -(a).imag; 
    return b;
}

#define alpha_conj cmp_conj

template <typename J>
static inline void vec_fma2(J *y, const J *x, const J val, ALPHA_INT len)
{
    ALPHA_INT i = 0;
    for (i = 0; i < len; i += 1)
    {
        y[i] = alpha_madd(x[i], val, y[i]);
    }
}

template <typename J>
static inline void vec_mul2(J *y, const J *x, const J val, ALPHA_INT len)
{
    ALPHA_INT i = 0;
    for (i = 0; i < len; i += 1)
    {
        y[i] = alpha_mul(x[i], val);
    }
}

//z=conj(x)*y
float inline ALPHA_DEVICE alpha_mul_2c(float x, float y)
{                                                                       
    return 1.0f;                                           
}

double inline ALPHA_DEVICE alpha_mul_2c(double x, double y)
{                                                                       
    return 1.0;                                           
}

ALPHA_Complex8 inline ALPHA_DEVICE alpha_mul_2c(ALPHA_Complex8 x, ALPHA_Complex8 y)
{                                                                       
    float _B = (x).real * (y).imag;                                
    float _C = -(x).imag * (y).real;
    ALPHA_Complex8 z;                               
    (z).real = ((x).real - (x).imag) * ((y).real - (y).imag) + _B - _C; 
    (z).imag = _B + _C;   
    return z;                                              
}

ALPHA_Complex16 inline ALPHA_DEVICE alpha_mul_2c(ALPHA_Complex16 x, ALPHA_Complex16 y)
{                                                                       
    double _B = (x).real * (y).imag;                                 
    double _C = -(x).imag * (y).real;
    ALPHA_Complex16 z;                                
    (z).real = ((x).real - (x).imag) * ((y).real - (y).imag) + _B - _C; 
    (z).imag = _B + _C;   
    return z;                                              
}

//z=x*conj(y)
float inline ALPHA_DEVICE alpha_mul_3c(float x, float y)
{                                                                       
    return 1.0f;                                           
}

double inline ALPHA_DEVICE alpha_mul_3c(double x, double y)
{                                                                       
    return 1.0;                                           
}

ALPHA_Complex8 inline ALPHA_DEVICE alpha_mul_3c(ALPHA_Complex8 x, ALPHA_Complex8 y)
{                                                                       
    float _B = (-(x).real) * (y).imag;                                
    float _C = (x).imag * (y).real;  
    ALPHA_Complex8 z;                               
    (z).real = ((x).real + (x).imag) * ((y).real + (y).imag) + _B - _C; 
    (z).imag = _B + _C;   
    return z;                                              
}

ALPHA_Complex16 inline ALPHA_DEVICE alpha_mul_3c(ALPHA_Complex16 x, ALPHA_Complex16 y)
{                                                                       
    double _B = (-(x).real) * (y).imag;                                
    double _C = (x).imag * (y).real;   
    ALPHA_Complex16 z;                                
    (z).real = ((x).real + (x).imag) * ((y).real + (y).imag) + _B - _C; 
    (z).imag = _B + _C;   
    return z;                                              
}

//d += conj(a) * b
float inline ALPHA_DEVICE alpha_madde_2c(float d, float a, float b)                                                
{
    return 1.0f;
}

double inline ALPHA_DEVICE alpha_madde_2c(double d, double a, double b)                                                
{
    return 1.0f;
}

ALPHA_Complex8 inline ALPHA_DEVICE alpha_madde_2c(ALPHA_Complex8 d, ALPHA_Complex8 a, ALPHA_Complex8 b)                                                
{                                                                        
    float _B = (a).real * (b).imag;                                
    float _C = -(a).imag * (b).real;     
    ALPHA_Complex8 z;                           
    (z).real = d.real + ((a).real - (a).imag) * ((b).real - (b).imag) + _B - _C; 
    (z).imag = d.imag + _B + _C;  
    return z;
}

ALPHA_Complex16 inline ALPHA_DEVICE alpha_madde_2c(ALPHA_Complex16 d, ALPHA_Complex16 a, ALPHA_Complex16 b)                                                
{                                                                        
    double _B = (a).real * (b).imag;                                
    double _C = -(a).imag * (b).real;     
    ALPHA_Complex16 z;                           
    (z).real = d.real + ((a).real - (a).imag) * ((b).real - (b).imag) + _B - _C; 
    (z).imag = d.imag + _B + _C;  
    return z;
}

//d += a * conj(b)
// #define cmp_madde_3c(d, a, b)                                                
//     {                                                                        
//         ALPHA_Float _B = -(a).real * (b).imag;                               
//         ALPHA_Float _C = (a).imag * (b).real;                                
//         (d).real += ((a).real + (a).imag) * ((b).real + (b).imag) + _B - _C; 
//         (d).imag += _B + _C;                                                 
//     }
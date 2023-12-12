#ifndef _Z_DEFINE_H_
#define _Z_DEFINE_H_

#define div_up(a, b) ((a+b-1)/b)
#define div_round_up(a, b) ((a+b-1)/b)

#define NUM_BIN 8
#define WSIZE 32

#define PWARP 4
#define PWARP_ROWS 256
#define PWARP_TSIZE 32
#define PWARP_BLOCK_SIZE (PWARP * PWARP_ROWS)

#define NUMERIC_PWARP 8
#define NUMERIC_PWARP_ROWS 128
#define NUMERIC_PWARP_TSIZE 32
#define NUMERIC_PWARP_BLOCK_SIZE (NUMERIC_PWARP * NUMERIC_PWARP_ROWS)

#define HASH_SINGLE
//#define HASH_MULTI

// cannot define WARP, since thrust source code uses WARP
//#define WARP 32

#define SYMBOLIC_SCALE_SMALL 1
#define SYMBOLIC_SCALE_LARGE 1
#define NUMERIC_SCALE_LARGE 2
#define NUMERIC_SCALE 1.5
#define THRESH_SCALE 0.8
#define HASH_SCALE 107

#endif

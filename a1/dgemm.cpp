#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>

#include "Stopwatch.h"

void dgemm(double* A, double* B, double* C) {
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        C[n*M + m] += A[k*M + m] * B[n*K + k];
      }
    }
  }
}

#define A(i,j) A[ (j)*lda + (i) ]
#define B(i,j) B[ (j)*ldb + (i) ]
#define C(i,j) C[ (j)*ldc + (i) ]

void AddDot16(int, double*, int, double*, int, double*, int);
void AddDot8x8_2(int, double*, int, double*, int, double*, int);
void AddDot4x4(int, double*, int, double*, int, double*, int);
void AddDot1x4(int, double*, int, double*, int, double*, int);
void PackMatrixA( int, double *, int, double *, int );
void PackMatrixB( int, double *, int, double * );
void InnerKernel(int, double*, int, double*, int, double*, int);

void dgemm_opt(double* A, double* B, double* C) {
  
  /** =================
   *         TODO
   *  ================= */
  int lda = M, ldb = K, ldc = M;
  int n,m,i,j;
  
  // Strip-mining with the dimension of length K
  /*for (int k = 0; k < K; k += 8*M) {
    int kb = K-k < 8*M? K-k : 8*M;
    InnerKernel(kb, &A(0,k), lda, &B(k,0), ldb, &C(0,0), ldc);
  }*/
  
  // 16x16 block
  for (m = 0; m < M; m += 16) {
    for (n = 0; n < N; n += 16) {
      AddDot16(K, &A(m,0), lda, &B(0,n), ldb, &C(m,n), ldc);
    }
  }
  
  // 8x8 block, AVX-512, packing of A
//   double packedA[8*K];
//   for (m = 0; m < M; m+=8) {
//     PackMatrixA(K, &A(m,0), lda, packedA, 8);
//     for (n = 0; n < N; n+=8) {
//       AddDot8x8_2(K, packedA, 8, &B(0,n), ldb, &C(m,n), ldc);
//     }
//   }
  
  // 8x8 block, AVX-512
//   for (m = 0; m < M; m+=8) {
//     for (n = 0; n < N; n+=8) {
//       AddDot8x8_2(K, &A(m,0), lda, &B(0,n), ldb, &C(m,n), ldc);
//     }
//   }

  // 4x4 block with packing of A
//   double packedA[4*K];
//   for (m = 0; m < M; m += 4) {
//     PackMatrixA(K, &A(m,0), lda, packedA, 4);
//     for (n = 0; n < N; n += 4) {
//       AddDot4x4(K, packedA, 4, &B(0,n), ldb, &C(m,n), ldc);
//     }
//   }

  // 4x4 block
//   for (m = 0; m < M; m += 4) {
//     for (n = 0; n < N; n += 4) {
//       AddDot4x4(K, &A(m,0), lda, &B(0,n), ldb, &C(m,n), ldc);
//     }
//   }

  // 1x4 block
//   for ( j=0; j<N; j+=4 ){
//     for ( i=0; i<M; i+=1 ){
//       AddDot1x4( K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc );
//     }
//   }
}

void InnerKernel(int k, double* A, int lda, double* B, int ldb, double* C, int ldc) {
  int n,m;
  double packedA[ M * k ], packedB[k*N];
  
  for (n = 0; n < N; n+=4) {
    //PackMatrixB( k, &B( 0, n ), ldb, &packedB[ n*k ] );
    for (m = 0; m < M; m+=4) {
      if (n == 0) PackMatrixA( k, &A( m, 0 ), lda, &packedA[ m*k ], 4 );
      AddDot4x4(k, &packedA[ m*k ], 4, &B(0,n), ldb, &C(m,n), ldc);
    }
  }
}

#include <immintrin.h>

void PackMatrixA( int k, double *A, int lda, double *a_to, int packSize )
{
  int j;

  for( j=0; j<k; j++){  /* loop over columns of A */
    double *a_ij_pntr = &A( 0, j );

#pragma vector always
    for (int i = 0; i<packSize; i++) {
      *(a_to+i) = *(a_ij_pntr+i);
    }
    a_to += packSize;
    
    //memcpy(a_to + j*packSize, &A( 0, j ), packSize);
  }
}

void PackMatrixB( int k, double* B, int ldb, double *b_to) {
  int i;
  double 
    *b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
    *b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

  for( i=0; i<k; i++){  /* loop over rows of B */
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}

typedef union
{
  __m512d v;
  double d[8];
} v8d_t;

typedef union
{
  __m256d v;
  double d[4];
} v4d_t;

void AddDot16(int KK, double* A, int lda, double* B, int ldb, double* C, int ldc) {
  int k;
  v8d_t c_x0,c_x1,c_x2,c_x3,c_x4,c_x5,c_x6,c_x7,c_x8,c_x9,c_x10,c_x11,c_x12,c_x13,c_x14,c_x15,
        c_x80,c_x81,c_x82,c_x83,c_x84,c_x85,c_x86,c_x87,c_x88,c_x89,c_x810,c_x811,c_x812,c_x813,c_x814,c_x815,
        a_xk,a_x8k,
        b_k0,b_k1,b_k2,b_k3,b_k4,b_k5,b_k6,b_k7,b_k8,b_k9,b_k10,b_k11,b_k12,b_k13,b_k14,b_k15;
  double    *b_k0_ptr = &B(0,0), *b_k1_ptr = &B(0,1), *b_k2_ptr = &B(0,2), *b_k3_ptr = &B(0,3),
            *b_k4_ptr = &B(0,4), *b_k5_ptr = &B(0,5), *b_k6_ptr = &B(0,6), *b_k7_ptr = &B(0,7),
            *b_k8_ptr = &B(0,8), *b_k9_ptr = &B(0,9), *b_k10_ptr = &B(0,10), *b_k11_ptr = &B(0,11),
            *b_k12_ptr = &B(0,12), *b_k13_ptr = &B(0,13), *b_k14_ptr = &B(0,14), *b_k15_ptr = &B(0,15);
    
  c_x0.v = _mm512_setzero_pd(); c_x1.v = _mm512_setzero_pd();
  c_x2.v = _mm512_setzero_pd(); c_x3.v = _mm512_setzero_pd();
  c_x4.v = _mm512_setzero_pd(); c_x5.v = _mm512_setzero_pd();
  c_x6.v = _mm512_setzero_pd(); c_x7.v = _mm512_setzero_pd();
  c_x8.v = _mm512_setzero_pd(); c_x9.v = _mm512_setzero_pd();
  c_x10.v = _mm512_setzero_pd(); c_x11.v = _mm512_setzero_pd();
  c_x12.v = _mm512_setzero_pd(); c_x13.v = _mm512_setzero_pd();
  c_x14.v = _mm512_setzero_pd(); c_x15.v = _mm512_setzero_pd();
  c_x80.v = _mm512_setzero_pd(); c_x81.v = _mm512_setzero_pd();
  c_x82.v = _mm512_setzero_pd(); c_x83.v = _mm512_setzero_pd();
  c_x84.v = _mm512_setzero_pd(); c_x85.v = _mm512_setzero_pd();
  c_x86.v = _mm512_setzero_pd(); c_x87.v = _mm512_setzero_pd();
  c_x88.v = _mm512_setzero_pd(); c_x89.v = _mm512_setzero_pd();
  c_x810.v = _mm512_setzero_pd(); c_x811.v = _mm512_setzero_pd();
  c_x812.v = _mm512_setzero_pd(); c_x813.v = _mm512_setzero_pd();
  c_x814.v = _mm512_setzero_pd(); c_x815.v = _mm512_setzero_pd();
  
  for (k = 0; k < KK; k++) {
    a_xk.v = _mm512_load_pd((double*) &A(0,k));
    a_x8k.v = _mm512_load_pd((double*) &A(8,k));

    b_k0.v = _mm512_set1_pd(*(b_k0_ptr+k));
    b_k1.v = _mm512_set1_pd(*(b_k1_ptr+k));
    b_k2.v = _mm512_set1_pd(*(b_k2_ptr+k));
    b_k3.v = _mm512_set1_pd(*(b_k3_ptr+k));
    b_k4.v = _mm512_set1_pd(*(b_k4_ptr+k));
    b_k5.v = _mm512_set1_pd(*(b_k5_ptr+k));
    b_k6.v = _mm512_set1_pd(*(b_k6_ptr+k));
    b_k7.v = _mm512_set1_pd(*(b_k7_ptr+k));
    b_k8.v = _mm512_set1_pd(*(b_k8_ptr+k));
    b_k9.v = _mm512_set1_pd(*(b_k9_ptr+k));
    b_k10.v = _mm512_set1_pd(*(b_k10_ptr+k));
    b_k11.v = _mm512_set1_pd(*(b_k11_ptr+k));
    b_k12.v = _mm512_set1_pd(*(b_k12_ptr+k));
    b_k13.v = _mm512_set1_pd(*(b_k13_ptr+k));
    b_k14.v = _mm512_set1_pd(*(b_k14_ptr+k));
    b_k15.v = _mm512_set1_pd(*(b_k15_ptr+k));
    
    c_x0.v = _mm512_fmadd_pd(a_xk.v, b_k0.v, c_x0.v);
    c_x1.v = _mm512_fmadd_pd(a_xk.v, b_k1.v, c_x1.v);
    c_x2.v = _mm512_fmadd_pd(a_xk.v, b_k2.v, c_x2.v);
    c_x3.v = _mm512_fmadd_pd(a_xk.v, b_k3.v, c_x3.v);
    c_x4.v = _mm512_fmadd_pd(a_xk.v, b_k4.v, c_x4.v);
    c_x5.v = _mm512_fmadd_pd(a_xk.v, b_k5.v, c_x5.v);
    c_x6.v = _mm512_fmadd_pd(a_xk.v, b_k6.v, c_x6.v);
    c_x7.v = _mm512_fmadd_pd(a_xk.v, b_k7.v, c_x7.v);
    c_x8.v = _mm512_fmadd_pd(a_xk.v, b_k8.v, c_x8.v);
    c_x9.v = _mm512_fmadd_pd(a_xk.v, b_k9.v, c_x9.v);
    c_x10.v = _mm512_fmadd_pd(a_xk.v, b_k10.v, c_x10.v);
    c_x11.v = _mm512_fmadd_pd(a_xk.v, b_k11.v, c_x11.v);
    c_x12.v = _mm512_fmadd_pd(a_xk.v, b_k12.v, c_x12.v);
    c_x13.v = _mm512_fmadd_pd(a_xk.v, b_k13.v, c_x13.v);
    c_x14.v = _mm512_fmadd_pd(a_xk.v, b_k14.v, c_x14.v);
    c_x15.v = _mm512_fmadd_pd(a_xk.v, b_k15.v, c_x15.v);
    
    c_x80.v = _mm512_fmadd_pd(a_x8k.v, b_k0.v, c_x80.v);
    c_x81.v = _mm512_fmadd_pd(a_x8k.v, b_k1.v, c_x81.v);
    c_x82.v = _mm512_fmadd_pd(a_x8k.v, b_k2.v, c_x82.v);
    c_x83.v = _mm512_fmadd_pd(a_x8k.v, b_k3.v, c_x83.v);
    c_x84.v = _mm512_fmadd_pd(a_x8k.v, b_k4.v, c_x84.v);
    c_x85.v = _mm512_fmadd_pd(a_x8k.v, b_k5.v, c_x85.v);
    c_x86.v = _mm512_fmadd_pd(a_x8k.v, b_k6.v, c_x86.v);
    c_x87.v = _mm512_fmadd_pd(a_x8k.v, b_k7.v, c_x87.v);
    c_x88.v = _mm512_fmadd_pd(a_x8k.v, b_k8.v, c_x88.v);
    c_x89.v = _mm512_fmadd_pd(a_x8k.v, b_k9.v, c_x89.v);
    c_x810.v = _mm512_fmadd_pd(a_x8k.v, b_k10.v, c_x810.v);
    c_x811.v = _mm512_fmadd_pd(a_x8k.v, b_k11.v, c_x811.v);
    c_x812.v = _mm512_fmadd_pd(a_x8k.v, b_k12.v, c_x812.v);
    c_x813.v = _mm512_fmadd_pd(a_x8k.v, b_k13.v, c_x813.v);
    c_x814.v = _mm512_fmadd_pd(a_x8k.v, b_k14.v, c_x814.v);
    c_x815.v = _mm512_fmadd_pd(a_x8k.v, b_k15.v, c_x815.v);
 }
  
  int i;
  for (i = 0; i < 8; i++) {
    C(i,0) += c_x0.d[i]; C(i,1) += c_x1.d[i]; C(i,2) += c_x2.d[i]; C(i,3) += c_x3.d[i];
    C(i,4) += c_x4.d[i]; C(i,5) += c_x5.d[i]; C(i,6) += c_x6.d[i]; C(i,7) += c_x7.d[i];
    C(i,8) += c_x8.d[i]; C(i,9) += c_x9.d[i]; C(i,10) += c_x10.d[i]; C(i,11) += c_x11.d[i];
    C(i,12) += c_x12.d[i]; C(i,13) += c_x13.d[i]; C(i,14) += c_x14.d[i]; C(i,15) += c_x15.d[i];
    C(i+8,0) += c_x80.d[i]; C(i+8,1) += c_x81.d[i]; C(i+8,2) += c_x82.d[i]; C(i+8,3) += c_x83.d[i];
    C(i+8,4) += c_x84.d[i]; C(i+8,5) += c_x85.d[i]; C(i+8,6) += c_x86.d[i]; C(i+8,7) += c_x87.d[i];
    C(i+8,8) += c_x88.d[i]; C(i+8,9) += c_x89.d[i]; C(i+8,10) += c_x810.d[i]; C(i+8,11) += c_x811.d[i];
    C(i+8,12) += c_x812.d[i]; C(i+8,13) += c_x813.d[i]; C(i+8,14) += c_x814.d[i]; C(i+8,15) += c_x815.d[i];
  }
}

void AddDot8x8_2(int KK, double* A, int lda, double* B, int ldb, double* C, int ldc) {
  int k;
  v8d_t c_x0,c_x1,c_x2,c_x3,c_x4,c_x5,c_x6,c_x7,
        a_xk,
        b_k0,b_k1,b_k2,b_k3,b_k4,b_k5,b_k6,b_k7;
  double
    *b_k0_ptr = &B(0,0), *b_k1_ptr = &B(0,1), *b_k2_ptr = &B(0,2), *b_k3_ptr = &B(0,3),
    *b_k4_ptr = &B(0,4), *b_k5_ptr = &B(0,5), *b_k6_ptr = &B(0,6), *b_k7_ptr = &B(0,7);
    
  c_x0.v = _mm512_setzero_pd();
  c_x1.v = _mm512_setzero_pd();
  c_x2.v = _mm512_setzero_pd();
  c_x3.v = _mm512_setzero_pd();
  c_x4.v = _mm512_setzero_pd();
  c_x5.v = _mm512_setzero_pd();
  c_x6.v = _mm512_setzero_pd();
  c_x7.v = _mm512_setzero_pd();
    
  for (k = 0; k < KK; k++) {
    a_xk.v = _mm512_load_pd((double*) &A(0,k));
    
    b_k0.v = _mm512_set1_pd(*(b_k0_ptr+k));
    b_k1.v = _mm512_set1_pd(*(b_k1_ptr+k));
    b_k2.v = _mm512_set1_pd(*(b_k2_ptr+k));
    b_k3.v = _mm512_set1_pd(*(b_k3_ptr+k));
    b_k4.v = _mm512_set1_pd(*(b_k4_ptr+k));
    b_k5.v = _mm512_set1_pd(*(b_k5_ptr+k));
    b_k6.v = _mm512_set1_pd(*(b_k6_ptr+k));
    b_k7.v = _mm512_set1_pd(*(b_k7_ptr+k));
    
    c_x0.v = _mm512_fmadd_pd(a_xk.v, b_k0.v, c_x0.v);
    c_x1.v = _mm512_fmadd_pd(a_xk.v, b_k1.v, c_x1.v);
    c_x2.v = _mm512_fmadd_pd(a_xk.v, b_k2.v, c_x2.v);
    c_x3.v = _mm512_fmadd_pd(a_xk.v, b_k3.v, c_x3.v);
    c_x4.v = _mm512_fmadd_pd(a_xk.v, b_k4.v, c_x4.v);
    c_x5.v = _mm512_fmadd_pd(a_xk.v, b_k5.v, c_x5.v);
    c_x6.v = _mm512_fmadd_pd(a_xk.v, b_k6.v, c_x6.v);
    c_x7.v = _mm512_fmadd_pd(a_xk.v, b_k7.v, c_x7.v);
  }
  
  int i;
  for (i = 0; i < 8; i++) {
    C(i,0) += c_x0.d[i]; C(i,1) += c_x1.d[i]; C(i,2) += c_x2.d[i]; C(i,3) += c_x3.d[i];
    C(i,4) += c_x4.d[i]; C(i,5) += c_x5.d[i]; C(i,6) += c_x6.d[i]; C(i,7) += c_x7.d[i];
  }
}

void AddDot4x4(int KK, double* A, int lda, double* B, int ldb, double* C, int ldc) {
  int k;
  v4d_t c_x0,c_x1,c_x2,c_x3,
        a_xk,
        b_k0,b_k1,b_k2,b_k3;
  double
    *b_k0_ptr = &B(0,0), *b_k1_ptr = &B(0,1), *b_k2_ptr = &B(0,2), *b_k3_ptr = &B(0,3);
    
  c_x0.v = _mm256_setzero_pd();
  c_x1.v = _mm256_setzero_pd();
  c_x2.v = _mm256_setzero_pd();
  c_x3.v = _mm256_setzero_pd();
    
  for (k = 0; k < KK; k++) {
    a_xk.v = _mm256_load_pd((double*) &A(0,k));
    
    b_k0.v = _mm256_set1_pd(*(b_k0_ptr+k));
    b_k1.v = _mm256_set1_pd(*(b_k1_ptr+k));
    b_k2.v = _mm256_set1_pd(*(b_k2_ptr+k));
    b_k3.v = _mm256_set1_pd(*(b_k3_ptr+k));
    
    c_x0.v = _mm256_fmadd_pd(a_xk.v, b_k0.v, c_x0.v);
    c_x1.v = _mm256_fmadd_pd(a_xk.v, b_k1.v, c_x1.v);
    c_x2.v = _mm256_fmadd_pd(a_xk.v, b_k2.v, c_x2.v);
    c_x3.v = _mm256_fmadd_pd(a_xk.v, b_k3.v, c_x3.v);
  }
  
  int i;
  for (i = 0; i < 4; i++) {
    C(i,0) += c_x0.d[i]; C(i,1) += c_x1.d[i]; C(i,2) += c_x2.d[i]; C(i,3) += c_x3.d[i];
  }
}

void AddDot1x4( int k, double* A, int lda,  double* B, int ldb, double* C, int ldc )
{
  /* So, this routine computes four elements of C: 
           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 
           C( i, j ), C( i, j+1 ), C( i, j+2 ), C( i, j+3 ) 
	  
     in the original matrix C.
     We next use indirect addressing */

  int p;
  register double 
    /* hold contributions to
       C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) */
       c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
    /* holds A( 0, p ) */
       a_0p_reg;
  double 
    /* Point to the current elements in the four columns of B */
    *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr; 
    
  bp0_pntr = &B(0,0);
  bp1_pntr = &B(0,1);
  bp2_pntr = &B(0,2);
  bp3_pntr = &B(0,3);

  c_00_reg = 0.0; 
  c_01_reg = 0.0; 
  c_02_reg = 0.0; 
  c_03_reg = 0.0;
 
  for ( p=0; p<k; p+=4 ){
    a_0p_reg = A(0,p);

    c_00_reg += a_0p_reg * *bp0_pntr;
    c_01_reg += a_0p_reg * *bp1_pntr;
    c_02_reg += a_0p_reg * *bp2_pntr;
    c_03_reg += a_0p_reg * *bp3_pntr;

    a_0p_reg = A(0,p+1);

    c_00_reg += a_0p_reg * *(bp0_pntr+1);
    c_01_reg += a_0p_reg * *(bp1_pntr+1);
    c_02_reg += a_0p_reg * *(bp2_pntr+1);
    c_03_reg += a_0p_reg * *(bp3_pntr+1);

    a_0p_reg = A(0,p+2);

    c_00_reg += a_0p_reg * *(bp0_pntr+2);
    c_01_reg += a_0p_reg * *(bp1_pntr+2);
    c_02_reg += a_0p_reg * *(bp2_pntr+2);
    c_03_reg += a_0p_reg * *(bp3_pntr+2);

    a_0p_reg = A(0,p+3);

    c_00_reg += a_0p_reg * *(bp0_pntr+3);
    c_01_reg += a_0p_reg * *(bp1_pntr+3);
    c_02_reg += a_0p_reg * *(bp2_pntr+3);
    c_03_reg += a_0p_reg * *(bp3_pntr+3);

    bp0_pntr+=4;
    bp1_pntr+=4;
    bp2_pntr+=4;
    bp3_pntr+=4;
  }

  C(0,0) += c_00_reg; 
  C(0,1) += c_01_reg; 
  C(0,2) += c_02_reg; 
  C(0,3) += c_03_reg;
}

int main(int argc, char** argv) {
  int repetitions = 10000;
  if (argc > 1) {
    repetitions = atoi(argv[1]);
  }
  
  /** Allocate memory */
  double* A, *B, *C, *A_test, *B_test, *C_test;
  
  posix_memalign(reinterpret_cast<void**>(&A),      ALIGNMENT, M*K*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B),      ALIGNMENT, K*N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C),      ALIGNMENT, M*N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&A_test), ALIGNMENT, M*K*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B_test), ALIGNMENT, K*N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C_test), ALIGNMENT, M*N*sizeof(double));

  for (int j = 0; j < K; ++j) {
    for (int i = 0; i < M; ++i) {
      A[j*M + i] = i + j;
    }
  }
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < K; ++i) {
      B[j*K + i] = (K-i) + (N-j);
    }
  }
  memset(C, 0, M*N*sizeof(double));
  memcpy(A_test, A, M*K*sizeof(double));
  memcpy(B_test, B, K*N*sizeof(double));
  memset(C_test, 0, M*N*sizeof(double));
  
  /** Check correctness of optimised dgemm */
  #pragma noinline
  {
    dgemm(A, B, C);
    dgemm_opt(A_test, B_test, C_test);
  }

  double error = 0.0;
  for (int i = 0; i < M*N; ++i) {
    double diff = C[i] - C_test[i];
    error += diff*diff;
  }
  error = sqrt(error);
  if (error > std::numeric_limits<double>::epsilon()) {
    printf("Optimised DGEMM is incorrect. Error: %e\n", error);
    return -1;
  }
  
  /** Test performance of optimised dgemm */
  
  #pragma noinline
  dgemm_opt(A, B, C);
  
  Stopwatch stopwatch;
  stopwatch.start();

  #pragma noinline
  for (int r = 0; r < repetitions; ++r) {
    dgemm_opt(A, B, C);
    __asm__ __volatile__("");
  }
  
  double time = stopwatch.stop();
  printf("%lf ms, %lf GFLOP/s\n", time * 1.0e3, repetitions*2.0*M*N*K/time * 1.0e-9);
  
  /** Clean up */
  
  free(A); free(B); free(C);
  free(A_test); free(B_test); free(C_test);

  return 0;
}

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

void AddDot8x8(int, double*, int, double*, int, double*, int);
void AddDot4x4(int, double*, int, double*, int, double*, int);
void AddDot1x4(int, double*, int, double*, int, double*, int);
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void InnerKernel(int, double*, int, double*, int, double*, int);

void dgemm_opt(double* A, double* B, double* C) {
  
  /** =================
   *         TODO
   *  ================= */
  int lda = M, ldb = K, ldc = M;
  
  /*for (int n = 0; n < N; n+=8) {
    for (int m = 0; m < M; m+=8) {
      AddDot8x8(K, &A(m,0), lda, &B(0,n), ldb, &C(m,n), ldc);
    }
  }*/
  
  for (int n = 0; n < N; n+=4) {
    for (int m = 0; m < M; m+=4) {
      AddDot4x4(K, &A(m,0), lda, &B(0,n), ldb, &C(m,n), ldc);
    }
  }
  
  /*int b = 8;
  
  for (int i = 0; i < M/b; i++)
    for (int j = 0; j < N/b; j++)
      for (int k = 0; k < K/b; k++) {
        // read required blocks of A and B (also C for k = 1)
        for (int ii = 1; ii <= b; ii++)
          for (int jj = 1; jj <= b; jj++)
            for (int kk = 1; kk <= b; kk++)
              C[i*b+ii + M * (j*b+jj)] += A[i∗b+ii + M * (k∗b+kk)] ∗ B[k∗b+kk + K * (j∗b+jj)];
      }*/
}

#include <immintrin.h>

typedef union
{
  __m512d v;
  double d[8];
} v8d_t;

void AddDot8x8(int KK, double* A, int lda, double* B, int ldb, double* C, int ldc) {
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
    
    c_x0.v += a_xk.v * b_k0.v;
    c_x1.v += a_xk.v * b_k1.v;
    c_x2.v += a_xk.v * b_k2.v;
    c_x3.v += a_xk.v * b_k3.v;
    c_x4.v += a_xk.v * b_k4.v;
    c_x5.v += a_xk.v * b_k5.v;
    c_x6.v += a_xk.v * b_k6.v;
    c_x7.v += a_xk.v * b_k7.v;
  }
  
  int i;
  for (i = 0; i < 8; i++) {
    C(i,0) += c_x0.d[i]; C(i,1) += c_x1.d[i]; C(i,2) += c_x2.d[i]; C(i,3) += c_x3.d[i];
    C(i,4) += c_x4.d[i]; C(i,5) += c_x5.d[i]; C(i,6) += c_x6.d[i]; C(i,7) += c_x7.d[i];
  }
}

typedef union
{
  __m256d v;
  double d[4];
} v4d_t;

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
    
    c_x0.v += a_xk.v * b_k0.v;
    c_x1.v += a_xk.v * b_k1.v;
    c_x2.v += a_xk.v * b_k2.v;
    c_x3.v += a_xk.v * b_k3.v;
  }
  
  int i;
  for (i = 0; i < 4; i++) {
    C(i,0) += c_x0.d[i]; C(i,1) += c_x1.d[i]; C(i,2) += c_x2.d[i]; C(i,3) += c_x3.d[i];
  }
}

void InnerKernel(int k, double* A, int lda, double* B, int ldb, double* C, int ldc) {
  int n,m;
  double packedA[ M * k ], packedB[k*N];
  
  for (n = 0; n < N; n+=4) {
    PackMatrixB( k, &B( 0, n ), ldb, &packedB[ n*k ] );
    for (m = 0; m < M; m+=4) {
      if (n == 0) PackMatrixA( k, &A( m, 0 ), lda, &packedA[ m*k ] );
      AddDot4x4(k, &packedA[ m*k ], 4, &B(0,n), ldb, &C(m,n), ldc);
    }
  }
}

void PackMatrixA( int k, double *A, int lda, double *a_to )
{
  int j;

  for( j=0; j<k; j++){  /* loop over columns of A */
    double 
      *a_ij_pntr = &A( 0, j );

    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);
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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

void AddDot4x4_(int k, double* A, int lda, double* B, int ldb, double* C, int ldc) {
  /* So, this routine computes a 4x4 block of matrix C
           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  
     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 
           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 
     And now we use vector registers and instructions */
  int p;
  v2df_t
    c_00_c_10_vreg, c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg,
    c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg,
    a_0p_a_1p_vreg,
    a_2p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 
  double 
    /* Point to the current elements in the four columns of B */
    *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
    
  b_p0_pntr = &B( 0, 0 );
  b_p1_pntr = &B( 0, 1 );
  b_p2_pntr = &B( 0, 2 );
  b_p3_pntr = &B( 0, 3 );

  c_00_c_10_vreg.v = _mm_setzero_pd();   
  c_01_c_11_vreg.v = _mm_setzero_pd();
  c_02_c_12_vreg.v = _mm_setzero_pd(); 
  c_03_c_13_vreg.v = _mm_setzero_pd(); 
  c_20_c_30_vreg.v = _mm_setzero_pd();   
  c_21_c_31_vreg.v = _mm_setzero_pd();  
  c_22_c_32_vreg.v = _mm_setzero_pd();   
  c_23_c_33_vreg.v = _mm_setzero_pd(); 
  
  for ( p=0; p<k; p++ ){
    a_0p_a_1p_vreg.v = _mm_load_pd( (double *) &A( 0, p ) );
    a_2p_a_3p_vreg.v = _mm_load_pd( (double *) &A( 2, p ) );

    b_p0_vreg.v = _mm_loaddup_pd( (double *) b_p0_pntr++ );   /* load and duplicate */
    b_p1_vreg.v = _mm_loaddup_pd( (double *) b_p1_pntr++ );   /* load and duplicate */
    b_p2_vreg.v = _mm_loaddup_pd( (double *) b_p2_pntr++ );   /* load and duplicate */
    b_p3_vreg.v = _mm_loaddup_pd( (double *) b_p3_pntr++ );   /* load and duplicate */

    /* First row and second rows */
    c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
    c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
    c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
    c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

    /* Third and fourth rows */
    c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
    c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
    c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
    c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
  }
  
  C( 0, 0 ) += c_00_c_10_vreg.d[0];  C( 0, 1 ) += c_01_c_11_vreg.d[0];  
  C( 0, 2 ) += c_02_c_12_vreg.d[0];  C( 0, 3 ) += c_03_c_13_vreg.d[0]; 

  C( 1, 0 ) += c_00_c_10_vreg.d[1];  C( 1, 1 ) += c_01_c_11_vreg.d[1];  
  C( 1, 2 ) += c_02_c_12_vreg.d[1];  C( 1, 3 ) += c_03_c_13_vreg.d[1]; 

  C( 2, 0 ) += c_20_c_30_vreg.d[0];  C( 2, 1 ) += c_21_c_31_vreg.d[0];  
  C( 2, 2 ) += c_22_c_32_vreg.d[0];  C( 2, 3 ) += c_23_c_33_vreg.d[0]; 

  C( 3, 0 ) += c_20_c_30_vreg.d[1];  C( 3, 1 ) += c_21_c_31_vreg.d[1];  
  C( 3, 2 ) += c_22_c_32_vreg.d[1];  C( 3, 3 ) += c_23_c_33_vreg.d[1];
}

void AddDot1x4(int k, double* A, int lda, double* B, int ldb, double* C, int ldc) {
  int p;
  register double c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg;
  double *bp0_p, *bp1_p, *bp2_p, *bp3_p;
  
  bp0_p = &B(0,0);
  bp1_p = &B(0,1);
  bp2_p = &B(0,2);
  bp3_p = &B(0,3);
  
  c_00_reg = 0.0;
  c_01_reg = 0.0;
  c_02_reg = 0.0;
  c_03_reg = 0.0;
  
  for ( p=0; p<k; p+=4 ){
    a_0p_reg = A( 0, p );

    c_00_reg += a_0p_reg * *bp0_p;
    c_01_reg += a_0p_reg * *bp1_p;
    c_02_reg += a_0p_reg * *bp2_p;
    c_03_reg += a_0p_reg * *bp3_p;

    a_0p_reg = A( 0, p+1 );

    c_00_reg += a_0p_reg * *(bp0_p+1);
    c_01_reg += a_0p_reg * *(bp1_p+1);
    c_02_reg += a_0p_reg * *(bp2_p+1);
    c_03_reg += a_0p_reg * *(bp3_p+1);

    a_0p_reg = A( 0, p+2 );

    c_00_reg += a_0p_reg * *(bp0_p+2);
    c_01_reg += a_0p_reg * *(bp1_p+2);
    c_02_reg += a_0p_reg * *(bp2_p+2);
    c_03_reg += a_0p_reg * *(bp3_p+2);

    a_0p_reg = A( 0, p+3 );

    c_00_reg += a_0p_reg * *(bp0_p+3);
    c_01_reg += a_0p_reg * *(bp1_p+3);
    c_02_reg += a_0p_reg * *(bp2_p+3);
    c_03_reg += a_0p_reg * *(bp3_p+3);
    
    bp0_p+=4;
    bp1_p+=4;
    bp2_p+=4;
    bp3_p+=4;
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
  
  dgemm_opt(A, B, C);
  
  // From here
  stopwatch.start();

  #pragma noinline
  for (int r = 0; r < repetitions; ++r) {
    dgemm(A, B, C);
    __asm__ __volatile__("");
  }
  
  time = stopwatch.stop();
  printf("%lf ms, %lf GFLOP/s\n", time * 1.0e3, repetitions*2.0*M*N*K/time * 1.0e-9);
  // To here
  
  /** Clean up */
  
  free(A); free(B); free(C);
  free(A_test); free(B_test); free(C_test);

  return 0;
}

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

#include <immintrin.h>
#include <omp.h>

#include "Stopwatch.h"

/** Syntax: M x N, ld S
 *  A M x N sub-block of of S x something matrix
 *  in column-major layout.
 *  That is C_ij := C[j*S + i], i=0,...,M-1,  j=0,...,N-1
 */
 


void dgemm(double* A, double* B, double* C, int S) {
  for (int n = 0; n < S; ++n) {
    for (int m = 0; m < S; ++m) {
      for (int k = 0; k < S; ++k) {
        C[n*S + m] += A[k*S + m] * B[n*S + k];
      }
    }
  }
}

#define A(i,j) A[(j)*lda + (i)]
#define B(i,j) B[(j)*ldb + (i)]
#define C(i,j) C[(j)*ldc + (i)]
#define X(i,j) X[(j)*ldx + (i)]

void microkernel_MN16_K128_S4096(double*, double*, double*);
void microkernel_MN16_K128_S1024(double*, double*, double*);
void microkernel_MN16_K128_S512(double*, double*, double*);
void microkernel_MN8_K256_S512(double*, double*, double*);

void printMatrix(const char* name, double *A, int m, int n, int lda) {
  int i,j;
  
  printf("Matrix %s (size %d %d, ld %d)\n", name, m, n, lda);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%f ", A[i+j*lda]);
    }
    printf("\n");
  }
}

void PackA(int KK, double* A, int lda, double* A_pack, int ldap) {
  int k, i;

  for (k = 0; k < KK; k++) { /* loop over columns */
    /*double *a_col = &A(0, k);

#pragma vector always
    for (i = 0; i < ldap; i++)
      *(A_pack + i) = *(a_col + i);
    A_pack += ldap;*/
    
    memcpy(A_pack + k * ldap, &A(0, k), ldap);
  }
}

void Pack(int cols, double* X, int ldx, double* X_pack, int ldxp) {
  int k;

  for (k = 0; k < cols; k++) /* loop over columns */
    memcpy(X_pack + k * ldxp, &X(0, k), ldxp * sizeof(double));
}

/**
 * A: M x K, ld M
 * B: K x N, ld K
 * C: M x N, ld S
 */
void microkernel(double* A, double* B, double* C, int S) {
  
  /** =================
   *         TODO
   *  ================= */
  int lda = M, ldb = K, ldc = S;
  
  microkernel_MN16_K128_S4096(A, B, C);
  //microkernel_MN16_K128_S1024(A, B, C);
  //microkernel_MN16_K128_S512(A, B, C);
  //microkernel_MN8_K256_S512(A, B, C);
  //C[0] += A[0] * B[0] + A[1] * B[1];
}

/**
 * A: MC x K, ld S
 * B:  K x S, ld K
 * C: MC x S, ld S
 */
void GEBP(double* A, double* B, double* C, double* A_pack, int S, int threadsPerTeam) {

  /** =================
   *         TODO
   *  ================= */
  int lda = S, ldb = K, ldc = S, ldap = M;
  
  //printMatrix("GEBP A", A, MC, K, lda);
  //printMatrix("GEBP B", B, K, S, ldb);

#pragma omp parallel for num_threads(MC/M) schedule(static)  
  for (int m = 0; m < MC; m += ldap) Pack(K, &A(m,0), lda, &A_pack[m*K], ldap);
  
#pragma omp parallel for num_threads(MC/M) schedule(static)
  // threadsPerTeam = S / N = 512...4096 / 16 = 32...256
  //             or = MC / M = 512 / 16 = 32
  for (int n = 0; n < S; n += N)
    for (int m = 0; m < MC; m += ldap)
      microkernel(&A_pack[m*K], &B(0,n), &C(m,n), S);
  //printMatrix("GEBP C", C, MC, S, ldc);
}

/**
 * A: S x K, ld S
 * B: K x S, ld S
 * C: S x S, ld S
 */
void GEPP(double* A, double* B, double* C, double** A_pack, double* B_pack, int S, int nTeams, int threadsPerTeam) {

  /** =================
   *         TODO
   *  ================= */
  int lda = S, ldb = S, ldc = S, ldbp = K;
  
  //printMatrix("GEPP A", A, S, K, lda);
  //printMatrix("GEPP B", B, K, S, ldb);
  
  Pack(S, B, ldb, B_pack, ldbp);
#pragma omp parallel for num_threads(nTeams) schedule(static)
  // nTeams = S / MC = 512...4096 / 512 = 1...8
  for (int m = 0; m < S; m += MC)
    GEBP(&A(m,0), B_pack, &C(m,0), A_pack[omp_get_thread_num()], S, threadsPerTeam);
  //printMatrix("GEPP C", C, S, S, ldc);
}

/**
 * A: S x S, ld S
 * B: S x S, ld S
 * C: S x S, ld S
 */
void GEMM(double* A, double* B, double* C, double** A_pack, double* B_pack, int S, int nTeams = 1, int threadsPerTeam = 1) {

  /** =================
   *         TODO
   *  ================= */
  int lda = S, ldb = S, ldc = S;
  int k;
  
  for (k = 0; k < S; k += K)
    GEPP(&A(0,k), &B(k,0), C, A_pack, B_pack, S, nTeams, threadsPerTeam);
}

int main(int argc, char** argv) {
  int S = 4096;
  bool test = true;
  int threadsPerTeam = 1;
  int nRepeat = 10;
  if (argc <= 1) {
    printf("Usage: dgemm <S> <test> <threadsPerTeam> <repetitions>");
    return -1;
  }
  if (argc > 1) {
    S = atoi(argv[1]);
  }
  if (argc > 2) {
    test = atoi(argv[2]) != 0;
  }
  if (argc > 3) {
    threadsPerTeam = atoi(argv[3]);
  }
  if (argc > 4) {
    nRepeat = atoi(argv[4]);
  }

  omp_set_nested(1);

  int nThreads, nTeams;
  #pragma omp parallel
  #pragma omp master
  {
    nThreads = omp_get_num_threads(); 
  }
  threadsPerTeam = std::min(threadsPerTeam, nThreads);
  nTeams = nThreads / threadsPerTeam;
  
  /** Allocate memory */
  double* A, *B, *C, *A_test, *B_test, *C_test, **A_pack, *B_pack, *C_aux;
  
  posix_memalign(reinterpret_cast<void**>(&A),      ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B),      ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C),      ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&A_test), ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B_test), ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C_test), ALIGNMENT, S*S*sizeof(double));

  posix_memalign(reinterpret_cast<void**>(&A_pack), ALIGNMENT, nTeams*sizeof(double*));
  for (int t = 0; t < nTeams; ++t) {
    posix_memalign(reinterpret_cast<void**>(&A_pack[t]), ALIGNMENT, MC*K*sizeof(double));
  }
  posix_memalign(reinterpret_cast<void**>(&B_pack), ALIGNMENT,  K*S*sizeof(double));

  #pragma omp parallel for
  for (int j = 0; j < S; ++j) {
    for (int i = 0; i < S; ++i) {
      A[j*S + i] = i + j;
      B[j*S + i] = (S-i) + (S-j);
      C[j*S + i] = 0.0;
    }
  }
  memcpy(A_test, A, S*S*sizeof(double));
  memcpy(B_test, B, S*S*sizeof(double));
  memset(C_test, 0, S*S*sizeof(double));
  
  /** Check correctness of optimised dgemm */
  if (test) {
    #pragma noinline
    {
      dgemm(A_test, B_test, C_test, S);
      //printMatrix("C_test", C_test, S, S, S);
      GEMM(A, B, C, A_pack, B_pack, S, nTeams, threadsPerTeam);
      //printMatrix("C", C, S, S, S);
    }

    double error = 0.0;
    for (int i = 0; i < S*S; ++i) {
      double diff = C[i] - C_test[i];
      error += diff*diff;
    }
    error = sqrt(error);
    if (error > std::numeric_limits<double>::epsilon()) {
      printf("Optimised DGEMM is incorrect. Error: %e\n", error);
      return -1;
    }
  }

  Stopwatch stopwatch;
  double time;

  /** Test performance of microkernel */

  stopwatch.start();
  for (int i = 0; i < 10000; ++i) {
    #pragma noinline
    microkernel(A, B, C, S);
    __asm__ __volatile__("");
  }
  time = stopwatch.stop();
  printf("Microkernel: %lf ms, %lf GFLOP/s\n", time*1.0e3, 10000*2.0*M*N*K/time * 1.0e-9);

  /** Test performance of GEBP */

  stopwatch.start();
  for (int i = 0; i < nRepeat; ++i) {
    #pragma noinline
    GEBP(A, B, C, A_pack[0], S, threadsPerTeam);
    __asm__ __volatile__("");
  }
  time = stopwatch.stop();
  printf("GEBP: %lf ms, %lf GFLOP/s\n", time*1.0e3, nRepeat*2.0*MC*S*K/time * 1.0e-9);

  /** Test performance of optimised GEMM */

  stopwatch.start();
  for (int i = 0; i < nRepeat; ++i) {
    #pragma noinline
    GEMM(A, B, C, A_pack, B_pack, S, nTeams, threadsPerTeam);
    __asm__ __volatile__("");
  }  
  time = stopwatch.stop();
  printf("GEMM: %lf ms, %lf GFLOP/s\n", time * 1.0e3, nRepeat*2.0*S*S*S/time * 1.0e-9);
  
  /** Clean up */
  
  free(A); free(B); free(C);
  free(A_test); free(B_test); free(C_test);
  for (int t = 0; t < nTeams; ++t) {
    free(A_pack[t]);
  }
  free(A_pack); free(B_pack);

  return 0;
}

void microkernel_MN16_K128_S4096(double* A, double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "33:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovapd 0(%%rdx), %%zmm16\n\t"
                       "vmovapd 32768(%%rdx), %%zmm17\n\t"
                       "vmovapd 65536(%%rdx), %%zmm18\n\t"
                       "vmovapd 98304(%%rdx), %%zmm19\n\t"
                       "vmovapd 131072(%%rdx), %%zmm20\n\t"
                       "vmovapd 163840(%%rdx), %%zmm21\n\t"
                       "vmovapd 196608(%%rdx), %%zmm22\n\t"
                       "vmovapd 229376(%%rdx), %%zmm23\n\t"
                       "vmovapd 262144(%%rdx), %%zmm24\n\t"
                       "vmovapd 294912(%%rdx), %%zmm25\n\t"
                       "vmovapd 327680(%%rdx), %%zmm26\n\t"
                       "vmovapd 360448(%%rdx), %%zmm27\n\t"
                       "vmovapd 393216(%%rdx), %%zmm28\n\t"
                       "vmovapd 425984(%%rdx), %%zmm29\n\t"
                       "vmovapd 458752(%%rdx), %%zmm30\n\t"
                       "vmovapd 491520(%%rdx), %%zmm31\n\t"
                       "movq $0, %%r14\n\t"
                       "34:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $1024, %%r15\n\t"
                       "movq $3072, %%rax\n\t"
                       "movq $5120, %%rbx\n\t"
                       "movq $7168, %%r11\n\t"
                       "movq %%rsi, %%r10\n\t"
                       "addq $9216, %%r10\n\t"
                       "vmovapd 0(%%rdi), %%zmm0\n\t"
                       "vmovapd 128(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 256(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 8(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 384(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 512(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 24(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 24(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 640(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 768(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 40(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 40(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 896(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $1024, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 56(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 56(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "addq $64, %%rsi\n\t"
                       "cmpq $128, %%r14\n\t"
                       "jl 34b\n\t"
                       "subq $1024, %%rsi\n\t"
                       "vmovapd %%zmm16, 0(%%rdx)\n\t"
                       "vmovapd %%zmm17, 32768(%%rdx)\n\t"
                       "vmovapd %%zmm18, 65536(%%rdx)\n\t"
                       "vmovapd %%zmm19, 98304(%%rdx)\n\t"
                       "vmovapd %%zmm20, 131072(%%rdx)\n\t"
                       "vmovapd %%zmm21, 163840(%%rdx)\n\t"
                       "vmovapd %%zmm22, 196608(%%rdx)\n\t"
                       "vmovapd %%zmm23, 229376(%%rdx)\n\t"
                       "vmovapd %%zmm24, 262144(%%rdx)\n\t"
                       "vmovapd %%zmm25, 294912(%%rdx)\n\t"
                       "vmovapd %%zmm26, 327680(%%rdx)\n\t"
                       "vmovapd %%zmm27, 360448(%%rdx)\n\t"
                       "vmovapd %%zmm28, 393216(%%rdx)\n\t"
                       "vmovapd %%zmm29, 425984(%%rdx)\n\t"
                       "vmovapd %%zmm30, 458752(%%rdx)\n\t"
                       "vmovapd %%zmm31, 491520(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $16320, %%rdi\n\t"
                       "cmpq $16, %%r12\n\t"
                       "jl 33b\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif

#ifndef NDEBUG
#ifdef _OPENMP
//#pragma omp atomic
#endif
//libxsmm_num_total_flops += 65536;
#endif
}

void microkernel_MN16_K128_S512(double* A, double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "33:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovapd 0(%%rdx), %%zmm16\n\t"
                       "vmovapd 4096(%%rdx), %%zmm17\n\t"
                       "vmovapd 8192(%%rdx), %%zmm18\n\t"
                       "vmovapd 12288(%%rdx), %%zmm19\n\t"
                       "vmovapd 16384(%%rdx), %%zmm20\n\t"
                       "vmovapd 20480(%%rdx), %%zmm21\n\t"
                       "vmovapd 24576(%%rdx), %%zmm22\n\t"
                       "vmovapd 28672(%%rdx), %%zmm23\n\t"
                       "vmovapd 32768(%%rdx), %%zmm24\n\t"
                       "vmovapd 36864(%%rdx), %%zmm25\n\t"
                       "vmovapd 40960(%%rdx), %%zmm26\n\t"
                       "vmovapd 45056(%%rdx), %%zmm27\n\t"
                       "vmovapd 49152(%%rdx), %%zmm28\n\t"
                       "vmovapd 53248(%%rdx), %%zmm29\n\t"
                       "vmovapd 57344(%%rdx), %%zmm30\n\t"
                       "vmovapd 61440(%%rdx), %%zmm31\n\t"
                       "movq $0, %%r14\n\t"
                       "34:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $1024, %%r15\n\t"
                       "movq $3072, %%rax\n\t"
                       "movq $5120, %%rbx\n\t"
                       "movq $7168, %%r11\n\t"
                       "movq %%rsi, %%r10\n\t"
                       "addq $9216, %%r10\n\t"
                       "vmovapd 0(%%rdi), %%zmm0\n\t"
                       "vmovapd 128(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 256(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 8(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 384(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 512(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 24(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 24(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 640(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 768(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 40(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 40(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 896(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $1024, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 56(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 56(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "addq $64, %%rsi\n\t"
                       "cmpq $128, %%r14\n\t"
                       "jl 34b\n\t"
                       "subq $1024, %%rsi\n\t"
                       "vmovapd %%zmm16, 0(%%rdx)\n\t"
                       "vmovapd %%zmm17, 4096(%%rdx)\n\t"
                       "vmovapd %%zmm18, 8192(%%rdx)\n\t"
                       "vmovapd %%zmm19, 12288(%%rdx)\n\t"
                       "vmovapd %%zmm20, 16384(%%rdx)\n\t"
                       "vmovapd %%zmm21, 20480(%%rdx)\n\t"
                       "vmovapd %%zmm22, 24576(%%rdx)\n\t"
                       "vmovapd %%zmm23, 28672(%%rdx)\n\t"
                       "vmovapd %%zmm24, 32768(%%rdx)\n\t"
                       "vmovapd %%zmm25, 36864(%%rdx)\n\t"
                       "vmovapd %%zmm26, 40960(%%rdx)\n\t"
                       "vmovapd %%zmm27, 45056(%%rdx)\n\t"
                       "vmovapd %%zmm28, 49152(%%rdx)\n\t"
                       "vmovapd %%zmm29, 53248(%%rdx)\n\t"
                       "vmovapd %%zmm30, 57344(%%rdx)\n\t"
                       "vmovapd %%zmm31, 61440(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $16320, %%rdi\n\t"
                       "cmpq $16, %%r12\n\t"
                       "jl 33b\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif

#ifndef NDEBUG
#ifdef _OPENMP
//#pragma omp atomic
#endif
//libxsmm_num_total_flops += 65536;
#endif
}

void microkernel_MN8_K256_S512(double* A, double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "33:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovapd 0(%%rdx), %%zmm24\n\t"
                       "vmovapd 4096(%%rdx), %%zmm25\n\t"
                       "vmovapd 8192(%%rdx), %%zmm26\n\t"
                       "vmovapd 12288(%%rdx), %%zmm27\n\t"
                       "vmovapd 16384(%%rdx), %%zmm28\n\t"
                       "vmovapd 20480(%%rdx), %%zmm29\n\t"
                       "vmovapd 24576(%%rdx), %%zmm30\n\t"
                       "vmovapd 28672(%%rdx), %%zmm31\n\t"
                       "movq $0, %%r14\n\t"
                       "34:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $2048, %%r15\n\t"
                       "movq $6144, %%rax\n\t"
                       "movq $10240, %%rbx\n\t"
                       "movq $14336, %%r11\n\t"
                       "vpxord %%zmm16, %%zmm16, %%zmm16\n\t"
                       "vpxord %%zmm17, %%zmm17, %%zmm17\n\t"
                       "vpxord %%zmm18, %%zmm18, %%zmm18\n\t"
                       "vpxord %%zmm19, %%zmm19, %%zmm19\n\t"
                       "vpxord %%zmm20, %%zmm20, %%zmm20\n\t"
                       "vpxord %%zmm21, %%zmm21, %%zmm21\n\t"
                       "vpxord %%zmm22, %%zmm22, %%zmm22\n\t"
                       "vpxord %%zmm23, %%zmm23, %%zmm23\n\t"
                       "vmovapd 0(%%rdi), %%zmm0\n\t"
                       "vmovapd 64(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 128(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vmovapd 192(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 256(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vmovapd 320(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 384(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vmovapd 448(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $512, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "addq $64, %%rsi\n\t"
                       "vaddpd %%zmm16, %%zmm24, %%zmm24\n\t"
                       "vaddpd %%zmm17, %%zmm25, %%zmm25\n\t"
                       "vaddpd %%zmm18, %%zmm26, %%zmm26\n\t"
                       "vaddpd %%zmm19, %%zmm27, %%zmm27\n\t"
                       "vaddpd %%zmm20, %%zmm28, %%zmm28\n\t"
                       "vaddpd %%zmm21, %%zmm29, %%zmm29\n\t"
                       "vaddpd %%zmm22, %%zmm30, %%zmm30\n\t"
                       "vaddpd %%zmm23, %%zmm31, %%zmm31\n\t"
                       "cmpq $256, %%r14\n\t"
                       "jl 34b\n\t"
                       "subq $2048, %%rsi\n\t"
                       "vmovapd %%zmm24, 0(%%rdx)\n\t"
                       "vmovapd %%zmm25, 4096(%%rdx)\n\t"
                       "vmovapd %%zmm26, 8192(%%rdx)\n\t"
                       "vmovapd %%zmm27, 12288(%%rdx)\n\t"
                       "vmovapd %%zmm28, 16384(%%rdx)\n\t"
                       "vmovapd %%zmm29, 20480(%%rdx)\n\t"
                       "vmovapd %%zmm30, 24576(%%rdx)\n\t"
                       "vmovapd %%zmm31, 28672(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $16320, %%rdi\n\t"
                       "cmpq $8, %%r12\n\t"
                       "jl 33b\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif

#ifndef NDEBUG
#ifdef _OPENMP
//#pragma omp atomic
#endif
//libxsmm_num_total_flops += 32768;
#endif
}

void microkernel_MN16_K128_S1024(double* A, double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "33:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovapd 0(%%rdx), %%zmm16\n\t"
                       "vmovapd 8192(%%rdx), %%zmm17\n\t"
                       "vmovapd 16384(%%rdx), %%zmm18\n\t"
                       "vmovapd 24576(%%rdx), %%zmm19\n\t"
                       "vmovapd 32768(%%rdx), %%zmm20\n\t"
                       "vmovapd 40960(%%rdx), %%zmm21\n\t"
                       "vmovapd 49152(%%rdx), %%zmm22\n\t"
                       "vmovapd 57344(%%rdx), %%zmm23\n\t"
                       "vmovapd 65536(%%rdx), %%zmm24\n\t"
                       "vmovapd 73728(%%rdx), %%zmm25\n\t"
                       "vmovapd 81920(%%rdx), %%zmm26\n\t"
                       "vmovapd 90112(%%rdx), %%zmm27\n\t"
                       "vmovapd 98304(%%rdx), %%zmm28\n\t"
                       "vmovapd 106496(%%rdx), %%zmm29\n\t"
                       "vmovapd 114688(%%rdx), %%zmm30\n\t"
                       "vmovapd 122880(%%rdx), %%zmm31\n\t"
                       "movq $0, %%r14\n\t"
                       "34:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $1024, %%r15\n\t"
                       "movq $3072, %%rax\n\t"
                       "movq $5120, %%rbx\n\t"
                       "movq $7168, %%r11\n\t"
                       "movq %%rsi, %%r10\n\t"
                       "addq $9216, %%r10\n\t"
                       "vmovapd 0(%%rdi), %%zmm0\n\t"
                       "vmovapd 128(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 256(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 8(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 384(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 512(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 24(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 24(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 640(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovapd 768(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 40(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 40(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovapd 896(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%r10)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $1024, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 56(%%r10)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 56(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "addq $64, %%rsi\n\t"
                       "cmpq $128, %%r14\n\t"
                       "jl 34b\n\t"
                       "subq $1024, %%rsi\n\t"
                       "vmovapd %%zmm16, 0(%%rdx)\n\t"
                       "vmovapd %%zmm17, 8192(%%rdx)\n\t"
                       "vmovapd %%zmm18, 16384(%%rdx)\n\t"
                       "vmovapd %%zmm19, 24576(%%rdx)\n\t"
                       "vmovapd %%zmm20, 32768(%%rdx)\n\t"
                       "vmovapd %%zmm21, 40960(%%rdx)\n\t"
                       "vmovapd %%zmm22, 49152(%%rdx)\n\t"
                       "vmovapd %%zmm23, 57344(%%rdx)\n\t"
                       "vmovapd %%zmm24, 65536(%%rdx)\n\t"
                       "vmovapd %%zmm25, 73728(%%rdx)\n\t"
                       "vmovapd %%zmm26, 81920(%%rdx)\n\t"
                       "vmovapd %%zmm27, 90112(%%rdx)\n\t"
                       "vmovapd %%zmm28, 98304(%%rdx)\n\t"
                       "vmovapd %%zmm29, 106496(%%rdx)\n\t"
                       "vmovapd %%zmm30, 114688(%%rdx)\n\t"
                       "vmovapd %%zmm31, 122880(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $16320, %%rdi\n\t"
                       "cmpq $16, %%r12\n\t"
                       "jl 33b\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif

#ifndef NDEBUG
#ifdef _OPENMP
//#pragma omp atomic
#endif
//libxsmm_num_total_flops += 65536;
#endif
}


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

void microkernel_MN16_K128(double*, double*, double*);

/**
 * A: M x K, ld M
 * B: K x N, ld K
 * C: M x N, ld S
 */
void microkernel(double* A, double* B, double* C, int S) {
  
  /** =================
   *         TODO
   *  ================= */
  
}

/**
 * A: MC x K, ld S
 * B:  K x S, ld K
 * C: MC x S, ld S
 */
void GEBP(double* A, double* B, double* C, double* A_pack, int S, int threadsPerTeam)
{
	  
  /** =================
   *         TODO
   *  ================= */
  
}

/**
 * A: S x K, ld S
 * B: K x S, ld S
 * C: S x S, ld S
 */
void GEPP(double* A, double* B, double* C, double** A_pack, double* B_pack, int S, int nTeams, int threadsPerTeam)
{
	  
  /** =================
   *         TODO
   *  ================= */
  
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
      GEMM(A, B, C, A_pack, B_pack, S, nTeams, threadsPerTeam);
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

void microkernel_MN16_K128(double* A, double* B, double* C) {
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
                       "vmovapd 128(%%rdx), %%zmm17\n\t"
                       "vmovapd 256(%%rdx), %%zmm18\n\t"
                       "vmovapd 384(%%rdx), %%zmm19\n\t"
                       "vmovapd 512(%%rdx), %%zmm20\n\t"
                       "vmovapd 640(%%rdx), %%zmm21\n\t"
                       "vmovapd 768(%%rdx), %%zmm22\n\t"
                       "vmovapd 896(%%rdx), %%zmm23\n\t"
                       "vmovapd 1024(%%rdx), %%zmm24\n\t"
                       "vmovapd 1152(%%rdx), %%zmm25\n\t"
                       "vmovapd 1280(%%rdx), %%zmm26\n\t"
                       "vmovapd 1408(%%rdx), %%zmm27\n\t"
                       "vmovapd 1536(%%rdx), %%zmm28\n\t"
                       "vmovapd 1664(%%rdx), %%zmm29\n\t"
                       "vmovapd 1792(%%rdx), %%zmm30\n\t"
                       "vmovapd 1920(%%rdx), %%zmm31\n\t"
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
                       "vmovapd %%zmm17, 128(%%rdx)\n\t"
                       "vmovapd %%zmm18, 256(%%rdx)\n\t"
                       "vmovapd %%zmm19, 384(%%rdx)\n\t"
                       "vmovapd %%zmm20, 512(%%rdx)\n\t"
                       "vmovapd %%zmm21, 640(%%rdx)\n\t"
                       "vmovapd %%zmm22, 768(%%rdx)\n\t"
                       "vmovapd %%zmm23, 896(%%rdx)\n\t"
                       "vmovapd %%zmm24, 1024(%%rdx)\n\t"
                       "vmovapd %%zmm25, 1152(%%rdx)\n\t"
                       "vmovapd %%zmm26, 1280(%%rdx)\n\t"
                       "vmovapd %%zmm27, 1408(%%rdx)\n\t"
                       "vmovapd %%zmm28, 1536(%%rdx)\n\t"
                       "vmovapd %%zmm29, 1664(%%rdx)\n\t"
                       "vmovapd %%zmm30, 1792(%%rdx)\n\t"
                       "vmovapd %%zmm31, 1920(%%rdx)\n\t"
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
#pragma omp atomic
#endif
//libxsmm_num_total_flops += 65536;
#endif
}

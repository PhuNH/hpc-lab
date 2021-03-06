#include <cstdlib>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#define N 10000
#define R 10

char *markerName = "dtrmv";

void dtrmv(double const* A, double const* x, double* y) {
#pragma omp parallel
{
  LIKWID_MARKER_START(markerName);
#pragma omp for
  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    for (int j = i; j < N; ++j) {
      sum += A[i*N + j] * x[j];
    }
    y[i] = sum;
  }
  LIKWID_MARKER_STOP(markerName);
}
}

void dtrmv_modified(double const* A, double const* x, double* y) {
#pragma omp parallel
{
  LIKWID_MARKER_START(markerName);
#pragma omp for
  for (int i = 0; i < N/2; ++i) {
    int j;
    double sum = 0.0;
    for (j = i; j < N; ++j) {
      sum += A[i*N + j] * x[j];
    }
    y[i] = sum;
    
    int k = N - 1 - i;
    sum = 0.0;
    for (j = k; j < N; ++j) {
      sum += A[k*N + j] * x[j];
    }
    y[k] = sum;
  }
  LIKWID_MARKER_STOP(markerName);
}
}

int main() {
  double *A, *x, *y;
  posix_memalign(reinterpret_cast<void**>(&A), 64, N*N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&x), 64,   N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&y), 64,   N*sizeof(double));
  
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      A[i*N + j] = i + j;
    }
    x[i] = i;
  }
  
  LIKWID_MARKER_INIT;
#pragma omp parallel
{
  LIKWID_MARKER_THREADINIT;
  LIKWID_MARKER_REGISTER(markerName);
}
  
  for (unsigned r = 0; r < R; ++r) {
    //dtrmv(A, x, y);
    dtrmv_modified(A, x, y);
  }
  
  LIKWID_MARKER_CLOSE;
  
  free(A); free(x); free(y);
  
  return 0;
}

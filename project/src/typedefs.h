#ifndef TYPEDEFS_H_
#define TYPEDEFS_H_

#include <mpi.h>
#include <cmath>
#include "constants.h"

typedef double DegreesOfFreedom[NUMBER_OF_DOFS];

struct GlobalConstants {
  double hx;
  double hy;
  int X;
  int Y;
  double maxTimestep;
  double endTime;
  
  double hxKxiT[GLOBAL_MATRIX_SIZE];
  double hyKetaT[GLOBAL_MATRIX_SIZE];
  double hxKxi[GLOBAL_MATRIX_SIZE];
  double hyKeta[GLOBAL_MATRIX_SIZE];
  
  double hxFxm0[GLOBAL_MATRIX_SIZE];
  double hxFxm1[GLOBAL_MATRIX_SIZE];
  double hyFym0[GLOBAL_MATRIX_SIZE];
  double hyFym1[GLOBAL_MATRIX_SIZE];
  double hxFxp0[GLOBAL_MATRIX_SIZE];
  double hxFxp1[GLOBAL_MATRIX_SIZE];
  double hyFyp0[GLOBAL_MATRIX_SIZE];
  double hyFyp1[GLOBAL_MATRIX_SIZE];
  
  void (*dgemm_beta_0)(const double*, const double*, double*); // NUMBER_OF_BASIS_FUNCTIONS, NUMBER_OF_QUANTITIES, NUMBER_OF_BASIS_FUNCTIONS
  void (*dgemm_beta_1)(const double*, const double*, double*); // NUMBER_OF_BASIS_FUNCTIONS, NUMBER_OF_QUANTITIES, NUMBER_OF_QUANTITIES
  
  // Size of the processors grid [int procs_y_axis,int procs_x_axis]
  int dims_proc[2];
  
  // Number of processors allocated
  int nb_procs;
};

struct LocalConstants {
	
  int rank;
  
  // Index among processor [i_proc,j_proc] with i on vertical axis, j on horizontal axis
  int coords_proc[2];
  
  // index among elements : starting element [x_elt_start,y_elt_start];
  int start_elts[2];
  
  // size elements : (x,y)
  int elts_size[2];
  
  // rank processors [UP, DOWN, LEFT, RIGHT]
  int adj_list[4];
  
  MPI_Group nbGroups[4];
};

struct Material {
  double K0;
  double rho0;
  double wavespeed;
//  inline double wavespeed() const { return sqrt(K0/rho0); }
};

struct SourceTerm {
  SourceTerm() : x(-1), y(-1) {} // -1 == invalid
  int x;
  int y;
  double phi[NUMBER_OF_BASIS_FUNCTIONS];
  double (*antiderivative)(double);
  int quantity;
};

#endif // TYPEDEFS_H_

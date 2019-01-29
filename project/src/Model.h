#ifndef MODEL_H_
#define MODEL_H_

#include "typedefs.h"

void computeTnxyTTnxy(double* Tnxy0_1, double* Tnxy01, double* Tnxy_10, double* Tnxy10, double*  TTnxy0_1, double* TTnxy01, double* TTnxy_10, double* TTnxy10);

/** Returns A in column-major storage */
void computeA(Material const& material, double A[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES]);

/** Returns B in column-major storage */
void computeB(Material const& material, double B[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES]);

/** Returns rotated flux solver in column-major storage for face-aligned coordinate system in direction (nx, ny).
  * (nx,ny) must be a unit vector, i.e. nx^2 + ny^2 = 1. */
void rotateFluxSolver(  double*      T,
                        double*      TT,
                        double const fluxSolver[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES],
                        double       rotatedFluxSolver[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] );

/** Returns A^+ in column-major storage */
void computeAplus( Material const&  local,
                   Material const&  neighbour,
                   double           Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] );

/** Returns A^- in column-major storage */
void computeAminus( Material const& local,
                    Material const& neighbour,
                    double          Aminus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] );

#endif // MODEL_H_

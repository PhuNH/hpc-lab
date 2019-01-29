#include "Model.h"

#include <cstring>
#include "GEMM.h"
#include "microkernels.h"

void computeTnxyTTnxy(double* Tnxy0_1, double* Tnxy01, double* Tnxy_10, double* Tnxy10, double*  TTnxy0_1, double* TTnxy01, double* TTnxy_10, double* TTnxy10){
  Tnxy0_1[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  Tnxy0_1[1*NUMBER_OF_QUANTITIES + 1] = 0.0;
  Tnxy0_1[1*NUMBER_OF_QUANTITIES + 2] = -1.0;
  Tnxy0_1[2*NUMBER_OF_QUANTITIES + 1] = 1.0;
  Tnxy0_1[2*NUMBER_OF_QUANTITIES + 2] = 0.0;
  
  Tnxy01[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  Tnxy01[1*NUMBER_OF_QUANTITIES + 1] = 0.0;
  Tnxy01[1*NUMBER_OF_QUANTITIES + 2] = 1.0;
  Tnxy01[2*NUMBER_OF_QUANTITIES + 1] = -1.0;
  Tnxy01[2*NUMBER_OF_QUANTITIES + 2] = 0.0;
  
  Tnxy_10[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  Tnxy_10[1*NUMBER_OF_QUANTITIES + 1] = -1.0;
  Tnxy_10[1*NUMBER_OF_QUANTITIES + 2] = 0.0;
  Tnxy_10[2*NUMBER_OF_QUANTITIES + 1] = -0.0;
  Tnxy_10[2*NUMBER_OF_QUANTITIES + 2] = -1.0;
 
  Tnxy10[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  Tnxy10[1*NUMBER_OF_QUANTITIES + 1] = 1.0;
  Tnxy10[1*NUMBER_OF_QUANTITIES + 2] = 0.0;
  Tnxy10[2*NUMBER_OF_QUANTITIES + 1] = -0.0;
  Tnxy10[2*NUMBER_OF_QUANTITIES + 2] = 1.0;
  
  TTnxy0_1[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  TTnxy0_1[1*NUMBER_OF_QUANTITIES + 1] = 0.0;
  TTnxy0_1[1*NUMBER_OF_QUANTITIES + 2] = 1.0;
  TTnxy0_1[2*NUMBER_OF_QUANTITIES + 1] = -1.0;
  TTnxy0_1[2*NUMBER_OF_QUANTITIES + 2] = 0.0;
  
  TTnxy01[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  TTnxy01[1*NUMBER_OF_QUANTITIES + 1] = 0.0;
  TTnxy01[1*NUMBER_OF_QUANTITIES + 2] = -1.0;
  TTnxy01[2*NUMBER_OF_QUANTITIES + 1] = 1.0;
  TTnxy01[2*NUMBER_OF_QUANTITIES + 2] = 0.0;
  
  TTnxy_10[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  TTnxy_10[1*NUMBER_OF_QUANTITIES + 1] = -1.0;
  TTnxy_10[1*NUMBER_OF_QUANTITIES + 2] = -0.0;
  TTnxy_10[2*NUMBER_OF_QUANTITIES + 1] = 0.0;
  TTnxy_10[2*NUMBER_OF_QUANTITIES + 2] = -1.0;
 
  TTnxy10[0*NUMBER_OF_QUANTITIES + 0] = 1.0;
  TTnxy10[1*NUMBER_OF_QUANTITIES + 1] = 1.0;
  TTnxy10[1*NUMBER_OF_QUANTITIES + 2] = -0.0;
  TTnxy10[2*NUMBER_OF_QUANTITIES + 1] = 0.0;
  TTnxy10[2*NUMBER_OF_QUANTITIES + 2] = 1.0;
}

void computeA(Material const& material, double A[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES])
{
  memset(A, 0, NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES*sizeof(double));
  A[0 * NUMBER_OF_QUANTITIES + 1] = material.K0;
  A[1 * NUMBER_OF_QUANTITIES + 0] = 1.0 / material.rho0;
}

void computeB(Material const& material, double B[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES])
{
  memset(B, 0, NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES*sizeof(double));
  B[0 * NUMBER_OF_QUANTITIES + 2] = material.K0;
  B[2 * NUMBER_OF_QUANTITIES + 0] = 1.0 / material.rho0;
}

void rotateFluxSolver(  double*      T,
                        double*      TT,
                        double const fluxSolver[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES],
                        double       rotatedFluxSolver[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] )
{
  double tmp[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
  
  memset(rotatedFluxSolver, 0, NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES*sizeof(double));
  
  microkernel_nq_3(T, fluxSolver, tmp);
  microkernel_nq_3(tmp, TT, rotatedFluxSolver);
}

void computeAplus( Material const&  local,
                   Material const&  neighbour,
                   double           Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] )
{
  memset(Aplus, 0, NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES*sizeof(double));
  
  double cm = local.wavespeed;
  double cp = neighbour.wavespeed;
  double div1 = 1.0 / (local.K0 * cp + neighbour.K0 * cm);
  double div2 = div1 / local.rho0;
  Aplus[0*NUMBER_OF_QUANTITIES + 0] = local.K0 * cm * cp * div1;
  Aplus[0*NUMBER_OF_QUANTITIES + 1] = local.K0 * local.K0 * cp * div1;
  Aplus[1*NUMBER_OF_QUANTITIES + 0] = neighbour.K0 * cm * div2;
  Aplus[1*NUMBER_OF_QUANTITIES + 1] = local.K0 * neighbour.K0 * div2;
}

void computeAminus( Material const& local,
                    Material const& neighbour,
                    double          Aminus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] )
{  
  memset(Aminus, 0, NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES*sizeof(double));
  
  double cm = local.wavespeed;
  double cp = neighbour.wavespeed;
  double div1 = 1.0 / (local.K0 * cp + neighbour.K0 * cm);
  double div2 = div1 / local.rho0;

  Aminus[0*NUMBER_OF_QUANTITIES + 0] = -local.K0 * cm * cp * div1;
  Aminus[0*NUMBER_OF_QUANTITIES + 1] = local.K0 * neighbour.K0 * cm * div1;
  Aminus[1*NUMBER_OF_QUANTITIES + 0] = local.K0 * cp * div2;
  Aminus[1*NUMBER_OF_QUANTITIES + 1] = -local.K0 * neighbour.K0 * div2;
}

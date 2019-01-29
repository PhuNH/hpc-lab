#include "Kernels.h"
#include "GEMM.h"
#include "constants.h"
#include "GlobalMatrices.h"
#include "Model.h"

void computeAder( double                  timestep,
                  GlobalConstants const&  globals,
                  Material const&         material,
                  DegreesOfFreedom const& degreesOfFreedom,
                  DegreesOfFreedom&       timeIntegrated )
{
  double tmp[NUMBER_OF_DOFS] = {}; // zero initialisation
  double derivatives[CONVERGENCE_ORDER][NUMBER_OF_DOFS] = {}; // zero initialisation
  double A[NUMBER_OF_QUANTITIES * NUMBER_OF_QUANTITIES];
  double B[NUMBER_OF_QUANTITIES * NUMBER_OF_QUANTITIES];
  
  double factor = timestep;

  for (unsigned dof = 0; dof < NUMBER_OF_DOFS; ++dof) {
    derivatives[0][dof] = degreesOfFreedom[dof];
    timeIntegrated[dof] = factor * degreesOfFreedom[dof];
  }
  
  computeA(material, A);
  computeB(material, B);
  
  for (unsigned der = 1; der < CONVERGENCE_ORDER; ++der) {  
    // tmp = Kxi^T * degreesOfFreedom
    (*globals.dgemm_beta_0)(globals.hxKxiT, derivatives[der-1], tmp);
    
    // derivatives[der] = -1/hx * tmp * A
    (*globals.dgemm_beta_1)(tmp, A, derivatives[der]);
    
    // tmp = Keta^T * degreesOfFreedom
    (*globals.dgemm_beta_0)(globals.hyKetaT, derivatives[der-1], tmp);
    
    // derivatives[der] += -1/hy * tmp * B
    (*globals.dgemm_beta_1)(tmp, B, derivatives[der]);

    factor *= timestep / (der + 1);
    for (unsigned dof = 0; dof < NUMBER_OF_DOFS; ++dof) {
      timeIntegrated[dof] += factor * derivatives[der][dof];
    }
  }
}

void computeVolumeIntegral( GlobalConstants const&  globals,
                            Material const&         material,
                            DegreesOfFreedom const& timeIntegrated,
                            DegreesOfFreedom&       degreesOfFreedom)
{
  double A[NUMBER_OF_QUANTITIES * NUMBER_OF_QUANTITIES];
  double B[NUMBER_OF_QUANTITIES * NUMBER_OF_QUANTITIES];
  double tmp[NUMBER_OF_DOFS] = {}; // zero initialisation
  
  computeA(material, A);
  computeB(material, B);
  
  // Computes tmp = Kxi * timeIntegrated
  (*globals.dgemm_beta_0)(globals.hxKxi, timeIntegrated, tmp);
  
  // Computes degreesOfFreedom += 1/hx tmp * A
  (*globals.dgemm_beta_1)(tmp, A, degreesOfFreedom);
  
  // Computes tmp = Keta * timeIntegrated
  (*globals.dgemm_beta_0)(globals.hyKeta, timeIntegrated, tmp);
  
  // Computes degreesOfFreedom += 1/hy tmp * B
  (*globals.dgemm_beta_1)(tmp, B, degreesOfFreedom);
}

void computeFlux( GlobalConstants const&  globals,
                  double const            fluxMatrix[NUMBER_OF_BASIS_FUNCTIONS*NUMBER_OF_BASIS_FUNCTIONS],
                  double const            rotatedFluxSolver[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES],
                  DegreesOfFreedom const& timeIntegrated,
                  DegreesOfFreedom        degreesOfFreedom )
{
  double tmp[NUMBER_OF_DOFS] = {}; // zero initialisation
  
  // Computes tmp = fluxMatrix * timeIntegrated
  (*globals.dgemm_beta_0)(fluxMatrix, timeIntegrated, tmp);
  
  // Computes degreesOfFreedom += factor * tmp * rotatedFluxSolver
  (*globals.dgemm_beta_1)(tmp, rotatedFluxSolver, degreesOfFreedom);
}

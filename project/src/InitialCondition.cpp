#include "InitialCondition.h"
#include "Quadrature.h"
#include "basisfunctions.h"
#include "GlobalMatrices.h"

void initialCondition(  GlobalConstants const& globals,
						LocalConstants const& locals,
                        Grid<Material>& materialGrid,
                        Grid<DegreesOfFreedom>& degreesOfFreedomGrid  )
{
  int const npoints = CONVERGENCE_ORDER+1;
  double points[npoints];
  double weights[npoints];
  
  seissol::quadrature::GaussLegendre(points, weights, npoints);
  
  for (int y = 0; y < locals.elts_size[1]; ++y) {
    for (int x = 0; x < locals.elts_size[0]; ++x) {
      DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);
      Material& material = materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]);
      
      double scaledWavespeed = sqrt(2.) * material.wavespeed / 2.;
      
      for (int i = 0; i < npoints; ++i) {
        double xi = (points[i]+1.)/2.;
        for (unsigned j = 0; j < npoints; ++j) {
          double eta = (points[j]+1.)/2.;
          double weight = weights[i] * weights[j] / 4.;

          double xp = xi*globals.hx + (x + locals.start_elts[0])*globals.hx;
          double yp = eta*globals.hy + (y + locals.start_elts[1])*globals.hy;
          double sn = sin(-2.*M_PI*xp - 2.*M_PI*yp);
          double f[] = {material.K0*sn, scaledWavespeed*sn, scaledWavespeed*sn};

          for (unsigned k = 0; k < NUMBER_OF_BASIS_FUNCTIONS; ++k) {
            double bf = (*basisFunctions[k])(xi, eta);
            for (unsigned q = 0; q < NUMBER_OF_QUANTITIES; ++q) {
              degreesOfFreedom[q*NUMBER_OF_BASIS_FUNCTIONS + k] += bf * f[q] * weight;
            }
          }
        }
      }
      
      for (int k = 0; k < NUMBER_OF_BASIS_FUNCTIONS; ++k) {
        for (int q = 0; q < NUMBER_OF_QUANTITIES; ++q) {
          degreesOfFreedom[q*NUMBER_OF_BASIS_FUNCTIONS + k] *= GlobalMatrices::Minv[k*NUMBER_OF_BASIS_FUNCTIONS + k];
        }
      }
    }
  }
}

void L2error_squared( double time,
              GlobalConstants const& globals,
			  LocalConstants const& locals,
              Grid<Material>& materialGrid,
              Grid<DegreesOfFreedom>& degreesOfFreedomGrid,
              double l2error[NUMBER_OF_QUANTITIES]  )
{
  int const npoints = CONVERGENCE_ORDER+1;
  double points[npoints];
  double weights[npoints];
  
  seissol::quadrature::GaussLegendre(points, weights, npoints);
  
  memset(l2error, 0, NUMBER_OF_QUANTITIES * sizeof(double));
  
  double area = globals.hx*globals.hy;

  for (int y = 0; y < locals.elts_size[1]; ++y) {
    for (int x = 0; x < locals.elts_size[0]; ++x) {
      DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);
      Material& material = materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]);
      
      double scaledWavespeed = sqrt(2.) * material.wavespeed / 2.;
      double omega = 2.*sqrt(2.) * M_PI * material.wavespeed;
      
      for (int i = 0; i < npoints; ++i) {
        double xi = (points[i]+1.)/2.;
        for (unsigned j = 0; j < npoints; ++j) {
          double eta = (points[j]+1.)/2.;
          double weight = weights[i] * weights[j] / 4.;

          double xp = xi*globals.hx + (x + locals.start_elts[0])*globals.hx;
          double yp = eta*globals.hy + (y + locals.start_elts[1])*globals.hy;
          double sn = sin(omega*time-2.*M_PI*xp - 2.*M_PI*yp);
          double f[] = {material.K0*sn, scaledWavespeed*sn, scaledWavespeed*sn};

          double Q[NUMBER_OF_QUANTITIES] = {};
          for (unsigned k = 0; k < NUMBER_OF_BASIS_FUNCTIONS; ++k) {
            double bf = (*basisFunctions[k])(xi, eta);
            for (unsigned q = 0; q < NUMBER_OF_QUANTITIES; ++q) {
              Q[q] += degreesOfFreedom[q*NUMBER_OF_BASIS_FUNCTIONS + k] * bf;
            }
          }
          
          for (unsigned q = 0; q < NUMBER_OF_QUANTITIES; ++q) {
            double diff = Q[q] - f[q];
            l2error[q] += diff * diff * weight * area;
          }
        }
      }      
    }
  }
}

void square_root_array(double * array, int length){
  for (unsigned q = 0; q < length; ++q) {
    array[q] = sqrt(array[q]);
  }
}

void initSourcetermPhi(double xi, double eta, SourceTerm& sourceterm) {
  for (unsigned k = 0; k < NUMBER_OF_BASIS_FUNCTIONS; ++k) {
    sourceterm.phi[k] = basisFunctions[k](xi, eta) * GlobalMatrices::Minv[k*NUMBER_OF_BASIS_FUNCTIONS + k];
  }
}

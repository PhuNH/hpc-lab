#include "Simulator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "constants.h"
#include "Kernels.h"
#include "Model.h"
#include "GlobalMatrices.h"

double determineTimestep(double hx, double hy, Grid<Material>& materialGrid)
{
  double maxWaveSpeed = 0.0;
  for (int y = 0; y < materialGrid.Y(); ++y) {
    for (int x = 0; x < materialGrid.X(); ++x) {
      maxWaveSpeed = std::max(maxWaveSpeed, materialGrid.get(x, y).wavespeed());
    }
  }
  
  return 0.25 * std::min(hx, hy)/((2*CONVERGENCE_ORDER-1) * maxWaveSpeed);
}

int simulate( GlobalConstants const&  globals,
			  LocalConstants const&   locals,
              Grid<Material>&         materialGrid,
              Grid<DegreesOfFreedom>& degreesOfFreedomGrid,
              WaveFieldWriter&        waveFieldWriter,
              SourceTerm&             sourceterm)
{
  std::ofstream m_xdmf;
  std::stringstream ss;
  ss << locals.rank;
  m_xdmf.open(("mat" + ss.str() + ".mat").c_str());

  Grid<DegreesOfFreedom> timeIntegratedGrid(locals.elts_size[1]+2, locals.elts_size[0]+2); // +2 for ghost layers
  
  double time;
  int step = 0;
  
  double **inbuf_x, **inbuf_y, **outbuf_x, **outbuf_y;
  inbuf_x = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  inbuf_y = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  outbuf_x = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  outbuf_y = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  
  for (int i = 0; i < 2; i++) {
    inbuf_x[i] = (double*) _mm_malloc(locals.elts_size[1] * sizeof(DegreesOfFreedom), ALIGNMENT);
    inbuf_y[i] = (double*) _mm_malloc(locals.elts_size[0] * sizeof(DegreesOfFreedom), ALIGNMENT);
    outbuf_x[i] = (double*) _mm_malloc(locals.elts_size[1] * sizeof(DegreesOfFreedom), ALIGNMENT);
    outbuf_y[i] = (double*) _mm_malloc(locals.elts_size[0] * sizeof(DegreesOfFreedom), ALIGNMENT);
  }
  
  for (time = 0.0; time < globals.endTime; time += globals.maxTimestep) {
    waveFieldWriter.writeTimestep(time, degreesOfFreedomGrid);
    
    double timestep = std::min(globals.maxTimestep, globals.endTime - time);
	
    for (int y = 0; y < locals.elts_size[0]; ++y) {
      for (int x = 0; x < locals.elts_size[1]; ++x) {
        double Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        double rotatedAplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        
        Material& material = materialGrid.get(x + locals.start_elts[1], y + locals.start_elts[0]); // Attention : Shifted coordinates
        DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);
        DegreesOfFreedom& timeIntegrated = timeIntegratedGrid.get(x+1, y+1);
        
        computeAder(timestep, globals, material, degreesOfFreedom, timeIntegrated);
        
        computeVolumeIntegral(globals, material, timeIntegrated, degreesOfFreedom);

        computeAplus(material, materialGrid.get(x + locals.start_elts[1], y + locals.start_elts[0] - 1), Aplus);
        rotateFluxSolver(0., -1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxm0, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[1], y + locals.start_elts[0] + 1), Aplus);
        rotateFluxSolver(0., 1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxm1, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[1] - 1, y + locals.start_elts[0]), Aplus);
        rotateFluxSolver(-1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fym0, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[1] + 1, y + locals.start_elts[0]), Aplus);
        rotateFluxSolver(1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fym1, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        if (x == 0) memcpy(&outbuf_y[0][y * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
        if (x == locals.elts_size[0] - 1) memcpy(&outbuf_y[1][y * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
        if (y == 0) memcpy(&outbuf_x[0][x * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
        if (y == locals.elts_size[1] - 1) memcpy(&outbuf_x[1][x * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
      }
    }

    if (locals.rank == 1) {
        printf("At %f %d to UP %f %f %f\n", time, locals.rank, outbuf_x[0][0], outbuf_x[0][NUMBER_OF_DOFS], outbuf_x[0][2*NUMBER_OF_DOFS]);
        printf("At %f %d to DOWN %f %f %f\n", time, locals.rank, outbuf_x[1][0], outbuf_x[1][NUMBER_OF_DOFS], outbuf_x[1][2*NUMBER_OF_DOFS]);
    }
	
	// Barrier : All the timeIntegrated variables have to be updated before data exchange
	MPI_Barrier(MPI_COMM_WORLD);
	
	/*
	Exchanging the data (necessary timeIntegrated column/row)
    */
	
	int tag = 13;
    MPI_Request reqs[8];
    MPI_Status stats[8];
	
	MPI_Isend(outbuf_x[0], locals.elts_size[1]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[UP], tag + locals.rank, MPI_COMM_WORLD, &reqs[UP]);
    MPI_Irecv(inbuf_x[0], locals.elts_size[1]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[UP], tag + locals.rank, MPI_COMM_WORLD, &reqs[UP+4]);
    MPI_Isend(outbuf_x[1], locals.elts_size[1]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[DOWN], tag + locals.adj_list[DOWN], MPI_COMM_WORLD, &reqs[DOWN]);
    MPI_Irecv(inbuf_x[1], locals.elts_size[1]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[DOWN], tag + locals.adj_list[DOWN], MPI_COMM_WORLD, &reqs[DOWN+4]);

	MPI_Isend(outbuf_y[0], locals.elts_size[0]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[LEFT], tag + locals.rank, MPI_COMM_WORLD, &reqs[LEFT]);
    MPI_Irecv(inbuf_y[0], locals.elts_size[0]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[LEFT], tag + locals.rank, MPI_COMM_WORLD, &reqs[LEFT+4]);
    MPI_Isend(outbuf_y[1], locals.elts_size[0]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[RIGHT], tag + locals.adj_list[RIGHT], MPI_COMM_WORLD, &reqs[RIGHT]);
    MPI_Irecv(inbuf_y[1], locals.elts_size[0]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[RIGHT], tag + locals.adj_list[RIGHT], MPI_COMM_WORLD, &reqs[RIGHT+4]);
	
	MPI_Waitall(8, reqs, stats);
    
    if (locals.rank == 3) {
        printf("At %f %d from UP %f %f %f\n", time, locals.rank, inbuf_x[0][0], inbuf_x[0][NUMBER_OF_DOFS], inbuf_x[0][2*NUMBER_OF_DOFS]);
        printf("At %f %d from DOWN %f %f %f\n", time, locals.rank, inbuf_x[1][0], inbuf_x[1][NUMBER_OF_DOFS], inbuf_x[1][2*NUMBER_OF_DOFS]);
    }
    
    for (int x = 1; x <= locals.elts_size[1]; ++x) {
      memcpy(timeIntegratedGrid.get(x, 0), &inbuf_x[0][(x-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
      memcpy(timeIntegratedGrid.get(x, locals.elts_size[0]+1), &inbuf_x[1][(x-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
    }
    for (int y = 1; y <= locals.elts_size[0]; ++y) {
      memcpy(timeIntegratedGrid.get(0, y), &inbuf_y[0][(y-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
      memcpy(timeIntegratedGrid.get(locals.elts_size[1]+1, y), &inbuf_y[1][(y-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
    }
    
    if (time == globals.maxTimestep) {
      m_xdmf << locals.rank << std::endl;
      for (int y = 0; y < locals.elts_size[0] + 2; ++y) {
        for (int x = 0; x < locals.elts_size[1] + 2; ++x) {
          DegreesOfFreedom& timeIntegrated = timeIntegratedGrid.get(x, y);
          m_xdmf << timeIntegrated[0] << " ";
        }
        m_xdmf << std::endl;
      }
      
      m_xdmf.close();
    }

    for (int y = 0; y < locals.elts_size[0]; ++y) {
      for (int x = 0; x < locals.elts_size[1]; ++x) {
        double Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        double rotatedAplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];

        Material& material = materialGrid.get(x + locals.start_elts[1], y + locals.start_elts[0]);
        DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);

        computeAminus(material, materialGrid.get(x + locals.start_elts[1], y + locals.start_elts[0]-1), Aplus);
        rotateFluxSolver(0., -1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxp0, rotatedAplus, timeIntegratedGrid.get(x+1, y), degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[1], y + locals.start_elts[0]+1), Aplus);
        rotateFluxSolver(0., 1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxp1, rotatedAplus, timeIntegratedGrid.get(x+1, y+2), degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[1]-1, y + locals.start_elts[0]), Aplus);
        rotateFluxSolver(-1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fyp0, rotatedAplus, timeIntegratedGrid.get(x, y+1), degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[1]+1, y + locals.start_elts[0]), Aplus);
        rotateFluxSolver(1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fyp1, rotatedAplus, timeIntegratedGrid.get(x+2, y+1), degreesOfFreedom);
      }
    }
    
    if (sourceterm.x >= locals.start_elts[1] && sourceterm.x <= locals.start_elts[1]+locals.elts_size[1] && sourceterm.y >= locals.start_elts[0] && sourceterm.y <= locals.start_elts[0]+locals.elts_size[0]) {
      double areaInv = 1. / (globals.hx*globals.hy);
      DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(sourceterm.x-locals.start_elts[1], sourceterm.y-locals.start_elts[0]);
      double timeIntegral = (*sourceterm.antiderivative)(time + timestep) - (*sourceterm.antiderivative)(time);
      for (unsigned b = 0; b < NUMBER_OF_BASIS_FUNCTIONS; ++b) {
        degreesOfFreedom[sourceterm.quantity * NUMBER_OF_BASIS_FUNCTIONS + b] += areaInv * timeIntegral * sourceterm.phi[b];
      }
    }
    
    ++step;
    if (step % 100 == 0 && locals.rank == 0) {
      std::cout << "At time / timestep: " << time << " / " << step << std::endl;
    }
  }
  
  for (int i = 0; i < 2; i++) {
    _mm_free(inbuf_x[i]);
    _mm_free(inbuf_y[i]);
    _mm_free(outbuf_x[i]);
    _mm_free(outbuf_y[i]);
  }
  _mm_free(inbuf_x);
  _mm_free(inbuf_y);
  _mm_free(outbuf_x);
  _mm_free(outbuf_y);
  
  waveFieldWriter.writeTimestep(globals.endTime, degreesOfFreedomGrid, true);
  
  return step;
}

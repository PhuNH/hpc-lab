#include "Simulator.h"

#include <algorithm>
#include <cmath>
#include <iostream>

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
			  LocalConstants const&  locals,
              Grid<Material>&         materialGrid,
              Grid<DegreesOfFreedom>& degreesOfFreedomGrid,
              WaveFieldWriter&        waveFieldWriter,
              SourceTerm&             sourceterm,
			  MPI_Comm cartcomm)
{
	
  Grid<DegreesOfFreedom> timeIntegratedGrid(locals.elts_size[0], locals.elts_size[1]);
  
  double time;
  int step = 0;
  
  double **inbuf_x, **inbuf_y;
  inbuf_x = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  inbuf_y = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  for (int i = 0; i < 2; i++) {
	inbuf_x[i] = (double*) _mm_malloc(locals.elts_size[0]*NUMBER_OF_DOFS, ALIGNMENT);
    inbuf_y[i] = (double*) _mm_malloc(locals.elts_size[1]*NUMBER_OF_DOFS, ALIGNMENT);
  }
  
  MPI_Datatype dofs;
  MPI_Type_vector(locals.elts_size[0], 1, 1, MPI_DOUBLE, &dofs);
  MPI_Type_commit(&dofs);
	
  MPI_Datatype columns;
  MPI_Type_vector(locals.elts_size[1], 1, locals.elts_size[0], dofs, &columns);
  MPI_Type_commit(&columns);

  
  for (time = 0.0; time < globals.endTime; time += globals.maxTimestep) {
    // should be uncommented when we will have dealt with writing file issue
	//waveFieldWriter.writeTimestep(time, degreesOfFreedomGrid);
  
    
    double timestep = std::min(globals.maxTimestep, globals.endTime - time);
	
    for (int y = 0; y < locals.elts_size[1]; ++y) {
      for (int x = 0; x < locals.elts_size[0]; ++x) {
        double Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        double rotatedAplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        
        Material& material = materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]); // Attention : Shifted coordinates
        DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);
        DegreesOfFreedom& timeIntegrated = timeIntegratedGrid.get(x, y);
        
        computeAder(timestep, globals, material, degreesOfFreedom, timeIntegrated);
        
        computeVolumeIntegral(globals, material, timeIntegrated, degreesOfFreedom);

        computeAplus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1] - 1), Aplus);
        rotateFluxSolver(0., -1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxm0, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1] + 1), Aplus);
        rotateFluxSolver(0., 1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxm1, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[0] - 1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(-1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fym0, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[0] + 1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fym1, rotatedAplus, timeIntegrated, degreesOfFreedom);
      }
    }
	
	// Barrier : All the timeIntegrated variables have to be updated before data exchange
	MPI_Barrier(MPI_COMM_WORLD);
	
	/*
	Exchanging the data (necessary timeIntegrated column/row)
    */
	
	int tag = 13;
    MPI_Request reqs[8];
    MPI_Status stats[8];
	
	MPI_Isend(timeIntegratedGrid.get_row(0), NUMBER_OF_DOFS * locals.elts_size[0], MPI_DOUBLE, locals.adj_list[UP], tag, MPI_COMM_WORLD, &reqs[UP]);
    MPI_Irecv(inbuf_x[0], locals.elts_size[0]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[UP], tag, MPI_COMM_WORLD, &reqs[UP+2]);
	
    MPI_Isend(timeIntegratedGrid.get_row(locals.elts_size[1]-1), locals.elts_size[0]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[DOWN], tag, MPI_COMM_WORLD, &reqs[DOWN]);
    MPI_Irecv(inbuf_x[1], locals.elts_size[0]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[DOWN], tag, MPI_COMM_WORLD, &reqs[DOWN+2]);

	MPI_Isend(timeIntegratedGrid.get_col_first(0), 1, columns, locals.adj_list[LEFT], tag, MPI_COMM_WORLD, &reqs[LEFT]);
    MPI_Irecv(inbuf_y[0], locals.elts_size[1]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[LEFT], tag, MPI_COMM_WORLD, &reqs[LEFT+4]);
    MPI_Isend(timeIntegratedGrid.get_col_first(locals.elts_size[0]-1), 1, columns, locals.adj_list[RIGHT], tag, MPI_COMM_WORLD, &reqs[RIGHT]);
    MPI_Irecv(inbuf_y[1], locals.elts_size[1]*NUMBER_OF_DOFS, MPI_DOUBLE, locals.adj_list[RIGHT], tag, MPI_COMM_WORLD, &reqs[RIGHT+4]);
	
	MPI_Waitall(8, reqs, stats);
	
	/*
	DegreesOfFreedom * timeIntegratedUp = (DegreesOfFreedom *) inbuf_x[0];
	DegreesOfFreedom * timeIntegratedDown = (DegreesOfFreedom *) inbuf_x[1];
	DegreesOfFreedom * timeIntegratedLeft = (DegreesOfFreedom *) inbuf_y[0];
	DegreesOfFreedom * timeIntegratedRight = (DegreesOfFreedom *) inbuf_y[1];

    for (int y = 0; y < locals.elts_size[1]; ++y) {
      for (int x = 0; x < locals.elts_size[0]; ++x) {
        double Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        double rotatedAplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];

        Material& material = materialGrid.get(x, y);
        DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);

        computeAminus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]-1), Aplus);
        rotateFluxSolver(0., -1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxp0, rotatedAplus, timeIntegratedUp[x], degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]+1), Aplus);
        rotateFluxSolver(0., 1., Aplus, rotatedAplus);
        computeFlux(-globals.hx / (globals.hx * globals.hy), GlobalMatrices::Fxp1, rotatedAplus, timeIntegratedDown[x], degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[0]-1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(-1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fyp0, rotatedAplus, timeIntegratedLeft[y], degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[0]+1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(1., 0., Aplus, rotatedAplus);
        computeFlux(-globals.hy / (globals.hx * globals.hy), GlobalMatrices::Fyp1, rotatedAplus, timeIntegratedRight[y], degreesOfFreedom);
      }
    }
    
    if (sourceterm.x >= locals.start_elts[0] && sourceterm.x <= locals.start_elts[0]+locals.elts_size[0] && sourceterm.y >= locals.start_elts[1] && sourceterm.y <= locals.start_elts[1]+locals.elts_size[1]) {
      double areaInv = 1. / (globals.hx*globals.hy);
      DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(sourceterm.x-locals.start_elts[0], sourceterm.y-locals.start_elts[1]);
      double timeIntegral = (*sourceterm.antiderivative)(time + timestep) - (*sourceterm.antiderivative)(time);
      for (unsigned b = 0; b < NUMBER_OF_BASIS_FUNCTIONS; ++b) {
        degreesOfFreedom[sourceterm.quantity * NUMBER_OF_BASIS_FUNCTIONS + b] += areaInv * timeIntegral * sourceterm.phi[b];
      }
    }
	*/
	
    ++step;
    if (step % 100 == 0) {
      std::cout << "At time / timestep: " << time << " / " << step << std::endl;
    }
	
  }
    int test;
	scanf("waiting : %d",&test);
  
    for (int i = 0; i < 2; i++) {
        _mm_free(inbuf_x[i]);
        _mm_free(inbuf_y[i]);
    }
    _mm_free(inbuf_x);
    _mm_free(inbuf_y);
	
	printf("before free \n");
	MPI_Type_free(&columns);
	MPI_Type_free(&dofs);
	printf("after free \n");
  
  // should be uncommented when we will have dealt with writing file issue
  // waveFieldWriter.writeTimestep(globals.endTime, degreesOfFreedomGrid, true);
  
  return step;
}

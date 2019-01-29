#include "Simulator.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "constants.h"
#include "Kernels.h"
#include "Model.h"
#include "GlobalMatrices.h"
#include <omp.h>

double determineTimestep(double hx, double hy, Grid<Material>& materialGrid)
{
  double maxWaveSpeed = 0.0;
  for (int y = 0; y < materialGrid.Y(); ++y) {
    for (int x = 0; x < materialGrid.X(); ++x) {
      maxWaveSpeed = std::max(maxWaveSpeed, materialGrid.get(x, y).wavespeed);
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
  Grid<DegreesOfFreedom> timeIntegratedGrid(locals.elts_size[0]+2, locals.elts_size[1]+2); // +2 for ghost layers
  
  double Tnxy0_1[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
  double Tnxy01[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
  double Tnxy_10[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
  double Tnxy10[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
 
  double TTnxy0_1[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
  double TTnxy01[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
  double TTnxy_10[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation
  double TTnxy10[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES] = {}; // zero initialisation

  computeTnxyTTnxy(Tnxy0_1, Tnxy01, Tnxy_10, Tnxy10, TTnxy0_1, TTnxy01, TTnxy_10, TTnxy10);
  
  double time;
  int step = 0;
  
  int size0InNrDofs = locals.elts_size[0]*NUMBER_OF_DOFS,
      size1InNrDofs = locals.elts_size[1]*NUMBER_OF_DOFS;
  int size0InBytes = locals.elts_size[0]*sizeof(DegreesOfFreedom),
      size1InBytes = locals.elts_size[1]*sizeof(DegreesOfFreedom);
  
  // TODO get rid of inbuf_x and outbuf_x
  double **inbuf_x, **inbuf_y, **outbuf_x, **outbuf_y;
  inbuf_x = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  inbuf_y = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  outbuf_x = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  outbuf_y = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
  
  for (int i = 0; i < 2; i++) {
    inbuf_x[i] = (double*) _mm_malloc(size0InBytes, ALIGNMENT);
    inbuf_y[i] = (double*) _mm_malloc(size1InBytes, ALIGNMENT);
    outbuf_x[i] = (double*) _mm_malloc(size0InBytes, ALIGNMENT);
    outbuf_y[i] = (double*) _mm_malloc(size1InBytes, ALIGNMENT);
  }
  
  //MPI_Win win;
  //double *sharedMem;
  // outbuf_x[0] - outbuf_x[1] - outbuf_y[0] - outbuf_y[1]
  //MPI_Win_allocate(2*(size0InNrDofs + size1InNrDofs) * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &sharedMem, &win);
  MPI_Win wins[4];
  MPI_Win_create(inbuf_x[0], size0InBytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &wins[0]);
  MPI_Win_create(inbuf_x[1], size0InBytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &wins[1]);
  MPI_Win_create(inbuf_y[0], size1InBytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &wins[2]);
  MPI_Win_create(inbuf_y[1], size1InBytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &wins[3]);
  
  for (time = 0.0; time < globals.endTime; time += globals.maxTimestep) {
    waveFieldWriter.writeTimestep(time, degreesOfFreedomGrid, globals, locals);
    
    double timestep = std::min(globals.maxTimestep, globals.endTime - time);
    
    #pragma omp parallel for num_threads(4)
    for (int k = 0; k < locals.elts_size[1]*locals.elts_size[0]; k++) {
        int x,y;
        x = (int) k / locals.elts_size[1];
        y = k % locals.elts_size[1];
        
        double Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        double rotatedAplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        
        Material& material = materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]); // Attention : Shifted coordinates
        DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);
        DegreesOfFreedom& timeIntegrated = timeIntegratedGrid.get(x+1, y+1);
        
        computeAder(timestep, globals, material, degreesOfFreedom, timeIntegrated);
        
        computeVolumeIntegral(globals, material, timeIntegrated, degreesOfFreedom);

        computeAplus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1] - 1), Aplus);
        rotateFluxSolver(Tnxy0_1, TTnxy0_1, Aplus, rotatedAplus);
        computeFlux(globals, globals.hxFxm0, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1] + 1), Aplus);
        rotateFluxSolver(Tnxy01, TTnxy01, Aplus, rotatedAplus);
        computeFlux(globals, globals.hxFxm1, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[0] - 1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(Tnxy_10, TTnxy_10, Aplus, rotatedAplus);
        computeFlux(globals, globals.hyFym0, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        computeAplus(material, materialGrid.get(x + locals.start_elts[0] + 1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(Tnxy10, TTnxy10, Aplus, rotatedAplus);
        computeFlux(globals, globals.hyFym1, rotatedAplus, timeIntegrated, degreesOfFreedom);
        
        if (x == 0) memcpy(&outbuf_y[0][y * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
        if (x == locals.elts_size[0] - 1) memcpy(&outbuf_y[1][y * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
        if (y == 0) memcpy(&outbuf_x[0][x * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
        if (y == locals.elts_size[1] - 1) memcpy(&outbuf_x[1][x * NUMBER_OF_DOFS], timeIntegrated, sizeof(DegreesOfFreedom));
    }

    // Barrier : All the timeIntegrated variables have to be updated before data exchange
    //MPI_Barrier(MPI_COMM_WORLD);
    
    /*
    Exchanging the data (necessary timeIntegrated column/row)
    */
    /*
    int tag = 13;
    MPI_Request reqs[8];
    MPI_Status stats[8];
    // TODO collective operations?
    MPI_Isend(outbuf_x[0], size0InNrDofs, MPI_DOUBLE, locals.adj_list[UP], tag + locals.rank, MPI_COMM_WORLD, &reqs[UP]);
    MPI_Irecv(inbuf_x[0], size0InNrDofs, MPI_DOUBLE, locals.adj_list[UP], tag + locals.rank, MPI_COMM_WORLD, &reqs[UP+4]);
    MPI_Isend(outbuf_x[1], size0InNrDofs, MPI_DOUBLE, locals.adj_list[DOWN], tag + locals.adj_list[DOWN], MPI_COMM_WORLD, &reqs[DOWN]);
    MPI_Irecv(inbuf_x[1], size0InNrDofs, MPI_DOUBLE, locals.adj_list[DOWN], tag + locals.adj_list[DOWN], MPI_COMM_WORLD, &reqs[DOWN+4]);

    MPI_Isend(outbuf_y[0], size1InNrDofs, MPI_DOUBLE, locals.adj_list[LEFT], tag + locals.rank, MPI_COMM_WORLD, &reqs[LEFT]);
    MPI_Irecv(inbuf_y[0], size1InNrDofs, MPI_DOUBLE, locals.adj_list[LEFT], tag + locals.rank, MPI_COMM_WORLD, &reqs[LEFT+4]);
    MPI_Isend(outbuf_y[1], size1InNrDofs, MPI_DOUBLE, locals.adj_list[RIGHT], tag + locals.adj_list[RIGHT], MPI_COMM_WORLD, &reqs[RIGHT]);
    MPI_Irecv(inbuf_y[1], size1InNrDofs, MPI_DOUBLE, locals.adj_list[RIGHT], tag + locals.adj_list[RIGHT], MPI_COMM_WORLD, &reqs[RIGHT+4]);
    
    MPI_Waitall(8, reqs, stats);
    */
    for (int i = 0; i < 4; i++)
      MPI_Win_post(locals.nbGroups[i], 0, wins[i]);
    
    MPI_Win_start(locals.nbGroups[0], 0, wins[1]);
    MPI_Put(outbuf_x[0], size0InNrDofs, MPI_DOUBLE, locals.adj_list[UP], 0, size0InNrDofs, MPI_DOUBLE, wins[1]);
    MPI_Win_complete(wins[1]);
    
    MPI_Win_start(locals.nbGroups[1], 0, wins[0]);
    MPI_Put(outbuf_x[1], size0InNrDofs, MPI_DOUBLE, locals.adj_list[DOWN], 0, size0InNrDofs, MPI_DOUBLE, wins[0]);
    MPI_Win_complete(wins[0]);
    
    MPI_Win_start(locals.nbGroups[2], 0, wins[3]);
    MPI_Put(outbuf_y[0], size1InNrDofs, MPI_DOUBLE, locals.adj_list[LEFT], 0, size1InNrDofs, MPI_DOUBLE, wins[3]);
    MPI_Win_complete(wins[3]);
    
    MPI_Win_start(locals.nbGroups[3], 0, wins[2]);
    MPI_Put(outbuf_y[1], size1InNrDofs, MPI_DOUBLE, locals.adj_list[RIGHT], 0, size1InNrDofs, MPI_DOUBLE, wins[2]);
    MPI_Win_complete(wins[2]);
    
    for (int i = 0; i < 4; i++)
      MPI_Win_wait(wins[i]);
    
    #pragma omp parallel for num_threads(4)
    for (int x = 1; x <= locals.elts_size[0]; ++x) {
      memcpy(timeIntegratedGrid.get(x, 0), &inbuf_x[0][(x-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
      memcpy(timeIntegratedGrid.get(x, locals.elts_size[1]+1), &inbuf_x[1][(x-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
    }
    
    #pragma omp parallel for num_threads(4)
    for (int y = 1; y <= locals.elts_size[1]; ++y) {
      memcpy(timeIntegratedGrid.get(0, y), &inbuf_y[0][(y-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
      memcpy(timeIntegratedGrid.get(locals.elts_size[0]+1, y), &inbuf_y[1][(y-1) * NUMBER_OF_DOFS], sizeof(DegreesOfFreedom));
    }

    #pragma omp parallel for num_threads(4)
    for (int k = 0; k < locals.elts_size[1]*locals.elts_size[0]; k++) {
        int x,y;
        x = (int) k / locals.elts_size[1];
        y = k % locals.elts_size[1];
        
        double Aplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
        double rotatedAplus[NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];

        Material& material = materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]);
        DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(x, y);

        computeAminus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]-1), Aplus);
        rotateFluxSolver(Tnxy0_1, TTnxy0_1, Aplus, rotatedAplus);
        computeFlux(globals, globals.hxFxp0, rotatedAplus, timeIntegratedGrid.get(x+1, y), degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[0], y + locals.start_elts[1]+1), Aplus);
        rotateFluxSolver(Tnxy01, TTnxy01, Aplus, rotatedAplus);
        computeFlux(globals, globals.hxFxp1, rotatedAplus, timeIntegratedGrid.get(x+1, y+2), degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[0]-1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(Tnxy_10, TTnxy_10, Aplus, rotatedAplus);
        computeFlux(globals, globals.hyFyp0, rotatedAplus, timeIntegratedGrid.get(x, y+1), degreesOfFreedom);
        
        computeAminus(material, materialGrid.get(x + locals.start_elts[0]+1, y + locals.start_elts[1]), Aplus);
        rotateFluxSolver(Tnxy10, TTnxy10, Aplus, rotatedAplus);
        computeFlux(globals, globals.hyFyp1, rotatedAplus, timeIntegratedGrid.get(x+2, y+1), degreesOfFreedom);
    }
    
    if (sourceterm.x >= locals.start_elts[0] && sourceterm.x < locals.start_elts[0]+locals.elts_size[0] && sourceterm.y >= locals.start_elts[1] && sourceterm.y < locals.start_elts[1]+locals.elts_size[1]) {
      double areaInv = 1. / (globals.hx*globals.hy);
      DegreesOfFreedom& degreesOfFreedom = degreesOfFreedomGrid.get(sourceterm.x-locals.start_elts[0], sourceterm.y-locals.start_elts[1]);
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
  
  for (int i = 0; i < 4; i++)
    MPI_Win_free(&wins[i]);
  
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
  
  waveFieldWriter.writeTimestep(globals.endTime, degreesOfFreedomGrid, globals, locals, true);
  
  return step;
}

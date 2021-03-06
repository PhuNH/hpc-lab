#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include <mpi.h>

#include "typedefs.h"
#include "Grid.h"
#include "WaveFieldWriter.h"

double determineTimestep(double hx, double hy, Grid<Material>& materialGrid);

int simulate( GlobalConstants const&  globals,
			  LocalConstants const&  locals,
              Grid<Material>&         materialGrid,
              Grid<DegreesOfFreedom>& degreesOfFreedomGrid,
              WaveFieldWriter&        waveFieldWriter,
              SourceTerm&             sourceterm);

#endif // SIMULATOR_H_

#include <cmath>
#include <iostream>

#include "tclap/CmdLine.h"
#include "typedefs.h"
#include "Simulator.h"
#include "Grid.h"
#include "InitialCondition.h"
#include <mpi.h>

void get_size_subgrid(int index_proc_axis, int nb_procs_axis, int grid_size, int * size_current, int * start){
	
	int min_size = ((int) grid_size/nb_procs_axis);
	
	if(grid_size % nb_procs_axis > index_proc_axis){
		*size_current = min_size + 1;
	}
	else{
		*size_current = min_size;
	}
	
	if(index_proc_axis == 0){
		*start = 0;
	}
	else{
		
		if(grid_size % nb_procs_axis > index_proc_axis){
			*start = index_proc_axis + index_proc_axis * min_size;
		}
		else{
			*start = grid_size % nb_procs_axis + index_proc_axis * min_size;
		}
		
	}
	
}

void initScenario0(GlobalConstants& globals, LocalConstants& locals, Grid<Material>& materialGrid, Grid<DegreesOfFreedom>& degreesOfFreedomGrid)
{    
  for (int y = 0; y < globals.Y; ++y) {
    for (int x = 0; x < globals.X; ++x) {
      Material& material = materialGrid.get(x, y);
      material.rho0 = 1.;
      material.K0 = 4.;
    }
  }

  initialCondition(globals, locals, materialGrid, degreesOfFreedomGrid);
}

void initScenario1(GlobalConstants& globals, LocalConstants& locals, Grid<Material>& materialGrid, Grid<DegreesOfFreedom>& degreesOfFreedomGrid)
{
  double checkerWidth = 0.25;

  for (int y = 0; y < globals.Y; ++y) {
    for (int x = 0; x < globals.X; ++x) {
      Material& material = materialGrid.get(x, y);
      int matId = static_cast<int>(x*globals.hx/checkerWidth) % 2 ^ static_cast<int>(y*globals.hy/checkerWidth) % 2;
      if (matId == 0) {
        material.rho0 = 1.;
        material.K0 = 2.;
      } else {
        material.rho0 = 2.;
        material.K0 = 0.5;
      }
    }
  }

  initialCondition(globals, locals, materialGrid, degreesOfFreedomGrid);
}

double sourceFunctionAntiderivative(double time)
{
  return sin(time);
}

void initSourceTerm23(GlobalConstants& globals, LocalConstants& locals, SourceTerm& sourceterm)
{
  sourceterm.quantity = 0; // pressure source
  double xs = 0.5;
  double ys = 0.5;
  sourceterm.x = static_cast<int>(xs / (globals.hx));
  sourceterm.y = static_cast<int>(ys / (globals.hy));
  double xi = (xs - sourceterm.x*globals.hx) / globals.hx;
  double eta = (ys - sourceterm.y*globals.hy) / globals.hy;
  
  initSourcetermPhi(xi, eta, sourceterm);
  
  sourceterm.antiderivative = sourceFunctionAntiderivative;  
}

void initScenario2(GlobalConstants& globals, LocalConstants& locals, Grid<Material>& materialGrid, Grid<DegreesOfFreedom>& degreesOfFreedomGrid, SourceTerm& sourceterm)
{
  for (int y = 0; y < globals.Y; ++y) {
    for (int x = 0; x < globals.X; ++x) {
      Material& material = materialGrid.get(x, y);
      material.rho0 = 1.;
      material.K0 = 2.;
    }
  }
  
  initSourceTerm23(globals, locals, sourceterm);
}

void initScenario3(GlobalConstants& globals, LocalConstants& locals, Grid<Material>& materialGrid, Grid<DegreesOfFreedom>& degreesOfFreedomGrid, SourceTerm& sourceterm)
{
  for (int y = 0; y < globals.Y; ++y) {
    for (int x = 0; x < globals.X; ++x) {
      Material& material = materialGrid.get(x, y);
      int matId;
      double xp = x*globals.hx;
      double yp = y*globals.hy;
      matId = (xp >= 0.25 && xp <= 0.75 && yp >= 0.25 && yp <= 0.75) ? 0 : 1;
      if (matId == 0) {
        material.rho0 = 1.;
        material.K0 = 2.;
      } else {
        material.rho0 = 2.;
        material.K0 = 0.5;
      }
    }
  }
  
  initSourceTerm23(globals, locals, sourceterm);
}

int main(int argc, char** argv)
{
  
  int scenario;
  double wfwInterval;
  std::string wfwBasename;
  GlobalConstants globals;
  LocalConstants locals;
  
  try {
    TCLAP::CmdLine cmd("ADER-DG for linear acoustics.", ' ', "0.1");
    TCLAP::ValueArg<int> scenarioArg("s", "scenario", "Scenario. 0=Convergence test. 1=Checkerboard.", true, 0, "int");
    TCLAP::ValueArg<int> XArg("x", "x-number-of-cells", "Number of cells in x direction.", true, 0, "int");
    TCLAP::ValueArg<int> YArg("y", "y-number-of-cells", "Number of cells in y direction.", true, 0, "int");
    TCLAP::ValueArg<std::string> basenameArg("o", "output", "Basename of wavefield writer output. Leave empty for no output.", false, "", "string");
    TCLAP::ValueArg<double> intervalArg("i", "interval", "Time interval of wavefield writer.", false, 0.1, "double");
    TCLAP::ValueArg<double> timeArg("t", "end-time", "Final simulation time.", false, 0.5, "double");
    TCLAP::ValueArg<int> horizontalArg("u", "hprocs", "Number of nodes - Horizontal axis", true, 0, "int");
    TCLAP::ValueArg<int> verticalArg("v", "vprocs", "Number of nodes - Vertical axis", true, 0, "int");
    cmd.add(scenarioArg);
    cmd.add(XArg);
    cmd.add(YArg);
    cmd.add(basenameArg);
    cmd.add(intervalArg);
    cmd.add(timeArg);
   cmd.add(horizontalArg);
    cmd.add(verticalArg);
    
    cmd.parse(argc, argv);
    
    scenario = scenarioArg.getValue();
    globals.X = XArg.getValue();
    globals.Y = YArg.getValue();
    wfwBasename = basenameArg.getValue();
    wfwInterval = intervalArg.getValue();
    globals.endTime = timeArg.getValue();
	
	
	globals.dims_proc[0] = horizontalArg.getValue();
	globals.dims_proc[1] = verticalArg.getValue();
    
    if (scenario < 0 || scenario > 3) {
      std::cerr << "Unknown scenario." << std::endl;
      return -1;
    }
    if (globals.X < 0 || globals.Y < 0) {
      std::cerr << "X or Y smaller than 0. Does not make sense." << std::endl;
      return -1;
    }
  } catch (TCLAP::ArgException &e) {
    std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
    return -1;
  }
  
  globals.hx = 1. / globals.X;
  globals.hy = 1. / globals.Y;
  
  // Initialize MPI functions
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &locals.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &globals.nb_procs);
  
  // Initialize cartesian grid
  int periods[2] = {0, 0}, reorder = 0;
  MPI_Comm cartcomm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, globals.dims_proc, periods, reorder, &cartcomm);
  MPI_Cart_coords(cartcomm, locals.rank, 2, locals.coords_proc);
  MPI_Cart_shift(cartcomm, 0, 1, &locals.adj_list[UP], &locals.adj_list[DOWN]);
  MPI_Cart_shift(cartcomm, 1, 1, &locals.adj_list[LEFT], &locals.adj_list[RIGHT]);
  
  // Initializing locals variable
  get_size_subgrid(locals.coords_proc[0], globals.dims_proc[0], globals.X, &locals.elts_size[0], &locals.start_elts[0]);
  get_size_subgrid(locals.coords_proc[1], globals.dims_proc[1], globals.Y, &locals.elts_size[1], &locals.start_elts[1]);
  
  printf("Rank : %d -- iproc = %d -- jproc = %d -- (xstart = %d, ystart = %d) -- (xsize = %d, ysize = %d) \n",locals.rank,locals.coords_proc [0],locals.coords_proc[1],locals.start_elts[0],locals.start_elts[1],locals.elts_size[0],locals.elts_size[1]);
  
  Grid<DegreesOfFreedom> degreesOfFreedomGrid(locals.elts_size[0], locals.elts_size[1]); // Change with MPI structure
  Grid<Material> materialGrid(globals.X, globals.Y); // Each node stores all : could be optimized - to see later
  SourceTerm sourceterm;
  
  switch (scenario) {
    case 0:
      initScenario0(globals, locals, materialGrid, degreesOfFreedomGrid);
      break;
    case 1:
      initScenario1(globals, locals, materialGrid, degreesOfFreedomGrid);
      break;
    case 2:
      initScenario2(globals, locals, materialGrid, degreesOfFreedomGrid, sourceterm);
      break;
    case 3:
      initScenario3(globals, locals, materialGrid, degreesOfFreedomGrid, sourceterm);
      break;
    default:
      break;
  }
  
  globals.maxTimestep = determineTimestep(globals.hx, globals.hy, materialGrid);
  
  // we will see how to deal with that after
  WaveFieldWriter waveFieldWriter(wfwBasename, globals, locals, wfwInterval, static_cast<int>(ceil( sqrt(NUMBER_OF_BASIS_FUNCTIONS) )));

  int steps = simulate(globals, locals, materialGrid, degreesOfFreedomGrid, waveFieldWriter, sourceterm, cartcomm);
  /*
  if (scenario == 0) {
    double l2error[NUMBER_OF_QUANTITIES];
    L2error(globals.endTime, globals, materialGrid, degreesOfFreedomGrid, l2error);
    std::cout << "L2 error analysis" << std::endl << "=================" << std::endl;
    std::cout << "Pressue (p):    " << l2error[0] << std::endl;
    std::cout << "X-Velocity (u): " << l2error[1] << std::endl;
    std::cout << "Y-Velocity (v): " << l2error[2] << std::endl;
  }
  
  std::cout << "Total number of timesteps: " << steps << std::endl;
  */
  MPI_Finalize();

 
  return 0;
}

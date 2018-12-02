/******************************************************************************
* Copyright (C) 2011 Technische Universitaet Muenchen                         *
* This file is part of the training material of the master's course           *
* Scientific Computing                                                        *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include <x86intrin.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <mpi.h>

#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

/// store number of grid points in one dimension
std::size_t grid_points_1d = 0;

/// store begin timestep
struct timeval begin;
/// store end timestep
struct timeval end;

int rank, size;
size_t h_procs, v_procs;
int x_min, x_max, y_min, y_max, nbrs[4], i_min, i_max, j_min, j_max;

/**
 * initialize and start timer
 */
void timer_start() {
	gettimeofday(&begin,(struct timezone *)0);
}

/**
 * stop timer and return measured time
 *
 * @return measured time
 */
double timer_stop() {
	gettimeofday(&end,(struct timezone *)0);
	double seconds, useconds;
	double ret, tmp;

	if (end.tv_usec >= begin.tv_usec)
	{
		seconds = (double)end.tv_sec - (double)begin.tv_sec;
		useconds = (double)end.tv_usec - (double)begin.tv_usec;
	}
	else
	{
		seconds = (double)end.tv_sec - (double)begin.tv_sec;
		seconds -= 1;					// Correction
		useconds = (double)end.tv_usec - (double)begin.tv_usec;
		useconds += 1000000;			// Correction
	}

	// get time in seconds
	tmp = (double)useconds;
	ret = (double)seconds;
	tmp /= 1000000;
	ret += tmp;

	return ret;
}

/**
 * stores a given grid into a file
 * 
 * @param grid the grid that should be stored
 * @param filename the filename
 */
void store_grid(double* grid, std::string filename) {
	std::fstream filestr;
	filestr.open (filename.c_str(), std::fstream::out);
	
	// calculate mesh width 
	double mesh_width = 1.0/((double)(grid_points_1d-1));

	// store grid incl. boundary points
	for (int i = 0; i < grid_points_1d; i++)
	{
		for (int j = 0; j < grid_points_1d; j++)
		{
			filestr << mesh_width*i << " " << mesh_width*j << " " << grid[(i*grid_points_1d)+j] << std::endl;
		}
		
		filestr << std::endl;
	}

	filestr.close();
}

/**
 * calculate the grid's initial values for given grid points
 *
 * @param x the x-coordinate of a given grid point
 * @param y the y-coordinate of a given grid point
 *
 * @return the initial value at position (x,y)
 */
double eval_init_func(double x, double y) {
	return (x*x)*(y*y);
}

/**
 * initializes a given grid: inner points are set to zero
 * boundary points are initialized by calling eval_init_func
 *
 * @param grid the grid to be initialized
 */
void init_grid(double* grid) {
	// set all points to zero
	for (int i = 0; i < grid_points_1d*grid_points_1d; i++)
	{
		grid[i] = 0.0;
	}

	double mesh_width = 1.0/((double)(grid_points_1d-1));
	
	for (int i = 0; i < grid_points_1d; i++)
	{
		// x-boundaries
		grid[i] = eval_init_func(0.0, ((double)i)*mesh_width);
		grid[i + ((grid_points_1d)*(grid_points_1d-1))] = eval_init_func(1.0, ((double)i)*mesh_width);
		// y-boundaries
		grid[i*grid_points_1d] = eval_init_func(((double)i)*mesh_width, 0.0);
		grid[(i*grid_points_1d) + (grid_points_1d-1)] = eval_init_func(((double)i)*mesh_width, 1.0);
	}
}

/**
 * initializes the right hand side, we want to keep it simple and
 * solve the Laplace equation instead of Poisson (-> b=0)
 *
 * @param b the right hand side
 */
void init_b(double* b) {
	// set all points to zero
	for (int i = 0; i < grid_points_1d*grid_points_1d; i++)
	{
		b[i] = 0.0;
	}
}

/**
 * copies data from one grid to another
 *
 * @param dest destination grid
 * @param src source grid
 */
void g_copy(double* dest, double* src) {
	for (int i = y_min; i < y_max; i++)
        for (int j = x_min; j < x_max; j++)
            dest[i*grid_points_1d+j] = src[i*grid_points_1d+j];
}

/**
 * calculates the dot product of the two grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param grid1 first grid
 * @param grid2 second grid
 */
double g_dot_product(double* grid1, double* grid2) {
	double tmp = 0.0;

	for (int i = i_min; i < i_max; i++)
		for (int j = j_min; j < j_max; j++)
			tmp += grid1[i*grid_points_1d+j] * grid2[i*grid_points_1d+j];
	
    double sum = 0.0;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sum;
}

/**
 * scales a grid by a given scalar (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param grid grid to be scaled
 * @param scalar scalar which is used to scale to grid
 */
void g_scale(double* grid, double scalar) {
	for (int i = i_min; i < i_max; i++)
		for (int j = j_min; j < j_max; j++)
			grid[i*grid_points_1d+j] *= scalar;
}

/**
 * implements BLAS's Xaxpy operation for grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param dest destination grid
 * @param src source grid
 * @param scalar scalar to scale to source grid
 */
void g_scale_add(double* dest, double* src, double scalar) {
	for (int i = i_min; i < i_max; i++)
		for (int j = j_min; j < j_max; j++)
			dest[i*grid_points_1d+j] += scalar * src[i*grid_points_1d+j];
}

/**
 * implements the the 5-point finite differences stencil (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 * 
 * @param grid grid for which the stencil should be evaluated
 * @param result grid where the stencil's evaluation should be stored
 */
void g_product_operator(double* grid, double* result) {
	size_t  x_size = x_max-x_min,
            y_size = y_max-y_min;
    double **inbuf_x, **inbuf_y;
    inbuf_x = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
    inbuf_y = (double**) _mm_malloc(2*sizeof(double*), ALIGNMENT);
    for (int i = 0; i < 2; i++) {
        inbuf_x[i] = (double*) _mm_malloc(x_size*sizeof(double), ALIGNMENT);
        inbuf_y[i] = (double*) _mm_malloc(y_size*sizeof(double), ALIGNMENT);
    }
    
    int tag = 13;
    MPI_Request reqs[8];
    MPI_Status stats[8];
    
    MPI_Isend(&grid[y_min*grid_points_1d+x_min], x_size, MPI_DOUBLE, nbrs[UP], tag, MPI_COMM_WORLD, &reqs[UP]);
    MPI_Irecv(inbuf_x[0], x_size, MPI_DOUBLE, nbrs[UP], tag, MPI_COMM_WORLD, &reqs[UP+4]);
    MPI_Isend(&grid[(y_max-1)*grid_points_1d+x_min], x_size, MPI_DOUBLE, nbrs[DOWN], tag, MPI_COMM_WORLD, &reqs[DOWN]);
    MPI_Irecv(inbuf_x[1], x_size, MPI_DOUBLE, nbrs[DOWN], tag, MPI_COMM_WORLD, &reqs[DOWN+4]);
    
    MPI_Datatype vectortype;
    MPI_Type_vector(y_size, 1, grid_points_1d, MPI_DOUBLE, &vectortype);
    MPI_Type_commit(&vectortype);
    MPI_Isend(&grid[y_min*grid_points_1d+x_min], 1, vectortype, nbrs[LEFT], tag, MPI_COMM_WORLD, &reqs[LEFT]);
    MPI_Irecv(inbuf_y[0], y_size, MPI_DOUBLE, nbrs[LEFT], tag, MPI_COMM_WORLD, &reqs[LEFT+4]);
    MPI_Isend(&grid[y_min*grid_points_1d+x_max-1], 1, vectortype, nbrs[RIGHT], tag, MPI_COMM_WORLD, &reqs[RIGHT]);
    MPI_Irecv(inbuf_y[1], y_size, MPI_DOUBLE, nbrs[RIGHT], tag, MPI_COMM_WORLD, &reqs[RIGHT+4]);
    
    MPI_Waitall(8, reqs, stats);
    MPI_Type_free(&vectortype);
    
    if (nbrs[UP] >= 0) {
        memcpy(&grid[(y_min-1)*grid_points_1d+x_min], inbuf_x[0], x_size*sizeof(double));
    }
    if (nbrs[DOWN] >= 0) {
        memcpy(&grid[y_max*grid_points_1d+x_min], inbuf_x[1], x_size*sizeof(double));
    }
    if (nbrs[LEFT] >= 0) {
        for (int k = 0; k < y_size; k++)
            grid[(y_min+k)*grid_points_1d+x_min-1] = inbuf_y[0][k];
    }
    if (nbrs[RIGHT] >= 0) {
        for (int k = 0; k < y_size; k++)
            grid[(y_min+k)*grid_points_1d+x_max] = inbuf_y[1][k];
    }
    
    for (int i = 0; i < 2; i++) {
        _mm_free(inbuf_x[i]);
        _mm_free(inbuf_y[i]);
    }
    _mm_free(inbuf_x);
    _mm_free(inbuf_y);

	double mesh_width = 1.0/((double)(grid_points_1d-1));
    
    for (int i = i_min; i < i_max; i++)
	{
		for (int j = j_min; j < j_max; j++)
		{
			result[(i*grid_points_1d)+j] =  (
							(4.0*grid[(i*grid_points_1d)+j]) 
							- grid[((i+1)*grid_points_1d)+j]
							- grid[((i-1)*grid_points_1d)+j]
							- grid[(i*grid_points_1d)+j+1]
							- grid[(i*grid_points_1d)+j-1]
							) * (mesh_width*mesh_width);
		}
	}
}

/**
 * The CG Solver (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * For details please see :
 * http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
 *
 * @param grid the grid containing the initial condition
 * @param b the right hand side
 * @param cg_max_iterations max. number of CG iterations 
 * @param cg_eps the CG's epsilon
 */
std::size_t solve(double* grid, double* b, std::size_t cg_max_iterations, double cg_eps) {
    if (rank == 0) std::cout << "Starting Conjugated Gradients" << std::endl;

	double eps_squared = cg_eps*cg_eps;
	std::size_t needed_iters = 0;

	// define temporal vectors
	double* q = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
	double* r = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
	double* d = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
	double* b_save = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
			
	g_copy(q, grid);
	g_copy(r, grid);
	g_copy(d, grid);
	g_copy(b_save, b);
	
	double delta_0 = 0.0;
	double delta_old = 0.0;
	double delta_new = 0.0;
	double beta = 0.0;
	double a = 0.0;
	double residuum = 0.0;
	
	g_product_operator(grid, d);
	g_scale_add(b, d, -1.0);
	g_copy(r, b);
	g_copy(d, r);

	// calculate starting norm
	delta_new = g_dot_product(r, r);
	delta_0 = delta_new*eps_squared;
	residuum = (delta_0/eps_squared);
	
    if (rank == 0) {
        std::cout << "Starting norm of residuum: " << (delta_0/eps_squared) << std::endl;
        std::cout << "Target norm:               " << (delta_0) << std::endl;
    }

	while ((needed_iters < cg_max_iterations) && (delta_new > delta_0)) {
		// q = A*d
		g_product_operator(d, q);

		// a = d_new / d.q
		a = delta_new/g_dot_product(d, q);
		
		// x = x + a*d
		g_scale_add(grid, d, a);
		
		if ((needed_iters % 50) == 0)
		{
			g_copy(b, b_save);
			g_product_operator(grid, q);
			g_scale_add(b, q, -1.0);
			g_copy(r, b);
		}
		else
		{
			// r = r - a*q
			g_scale_add(r, q, -a);
		}
		
		// calculate new deltas and determine beta
		delta_old = delta_new;
		delta_new = g_dot_product(r, r);
		beta = delta_new/delta_old;

		// adjust d
		g_scale(d, beta);
		g_scale_add(d, r, 1.0);
		
		residuum = delta_new;
		needed_iters++;
		if (rank == 0) std::cout << "(iter: " << needed_iters << ")delta: " << delta_new << std::endl;
	}

	if (rank == 0) {
        std::cout << "Number of iterations: " << needed_iters << " (max. " << cg_max_iterations << ")" << std::endl;
        std::cout << "Final norm of residuum: " << delta_new << std::endl;
    }
	
	_mm_free(d);
	_mm_free(q);
	_mm_free(r);
	_mm_free(b_save);
    
    
}

/**
 * main application
 *
 * @param argc number of cli arguments
 * @param argv values of cli arguments
 */
int main(int argc, char* argv[]) {
	/*printf("%d\n", argc);
    MPI_Init(&argc, &argv);
	printf("%d\n", argc);*/
    
    char name[1024];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(name, 1024);
    
    std::cout << "Hello, this is (rank << " << rank << " of " << size << " on " << name << ")" << std::endl;
	
	// check if all parameters are specified
    if (argc < 4) {
        if (rank == 0) {
            std::cout << std::endl;
            std::cout << "meshwidth" << std::endl;
            std::cout << "cg_max_iterations" << std::endl;
            std::cout << "cg_eps" << std::endl;
            std::cout << std::endl;
            std::cout << "example:" << std::endl;
            std::cout << "./app 0.125 100 0.0001" << std::endl;
            std::cout << std::endl;
        }
		return -1;
	}
	
	// read cli arguments
	double mesh_width = atof(argv[1]);
	size_t cg_max_iterations = atoi(argv[2]);
	double cg_eps = atof(argv[3]);
    
    h_procs = argc < 5 ? (size < 4 ? 1 : 2) : atoi(argv[4]);
    v_procs = argc < 6 ? size/h_procs : atoi(argv[5]);
    
    if (h_procs * v_procs != size) {
        if (rank == 0)
            std::cout << "There is some issue with number of processors. Terminating..." << std::endl;
        return -1;
    }
    if (grid_points_1d % h_procs == 1) h_procs--;
    if (grid_points_1d % v_procs == 1) v_procs--;

	// calculate grid points per dimension
	grid_points_1d = (std::size_t)(1.0/mesh_width)+1;
	
	int dims[2] = {h_procs, v_procs}, periods[2] = {0, 0}, reorder = 0;
    int coords[2];
    MPI_Comm cartcomm;
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    
    int last_x_len = grid_points_1d % v_procs, last_y_len = grid_points_1d % h_procs;
    int normal_x_len = (grid_points_1d - last_x_len) / (v_procs - 1),
        normal_y_len = (grid_points_1d - last_y_len) / (h_procs - 1);
    x_min = normal_x_len * coords[1];
    x_max = coords[1] < v_procs - 1 ? x_min + normal_x_len : grid_points_1d;
    y_min = normal_y_len * coords[0];
    y_max = coords[0] < h_procs - 1 ? y_min + normal_y_len : grid_points_1d;
    
    i_min = nbrs[UP] >= 0 ? y_min : y_min+1;
    i_max = nbrs[DOWN] >= 0 ? y_max : y_max-1;
    j_min = nbrs[LEFT] >= 0 ? x_min : x_min+1;
    j_max = nbrs[RIGHT] >= 0 ? x_max : x_max-1;
    
    // initialize the gird and rights hand side
	double* grid = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
	double* b = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
	init_grid(grid);
	if (rank == 0) store_grid(grid, "initial_condition.gnuplot");
	init_b(b);
	if (rank == 0) store_grid(b, "b.gnuplot");
	
	// solve Poisson equation using CG method
	timer_start();
	solve(grid, b, cg_max_iterations, cg_eps);
	double time = timer_stop();
    
    std::stringstream ss;
    ss << rank;
    std::string store_file = "solution" + ss.str() + ".gnuplot";
	store_grid(grid, store_file);
	
	if (rank == 0) std::cout << std::endl << "Needed time: " << time << " s" << std::endl << std::endl;
	
	_mm_free(grid);
	_mm_free(b);
    
    MPI_Finalize();

	return 0;
}


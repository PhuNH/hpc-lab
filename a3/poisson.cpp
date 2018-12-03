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

void printGrid(std::string name, double *grid, int x_1, int x_2, int y_1, int y_2) {
    std::cout << name << ":" << std::endl;
    for (int i = y_1; i < y_2; i++) {
        for (int j = x_1; j < x_2; j++)
            std::cout << " " << grid[i*grid_points_1d+j];
        std::cout << std::endl;
    }
}

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
            //filestr << "\t" << grid[(i*grid_points_1d)+j];
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
        
//     printGrid(std::string("grid1"), grid1, 0, grid_points_1d, 0, grid_points_1d);
//     printGrid(std::string("grid2"), grid2, 0, grid_points_1d, 0, grid_points_1d);
//     std::cout << "tmp " << tmp << std::endl;
    
    double sum = 0.0;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//     std::cout << "sum " << sum << std::endl;
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
//     printGrid(std::string("recv up"), inbuf_x[0], 0, x_size, 0, 1);
//     printGrid(std::string("recv down"), inbuf_x[1], 0, x_size, 0, 1);
//     printGrid(std::string("recv left"), inbuf_y[0], 0, y_size, 0, 1);
//     printGrid(std::string("recv right"), inbuf_y[1], 0, y_size, 0, 1);
    
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
//     printGrid(std::string("grid"), grid, 0, grid_points_1d, 0, grid_points_1d);
    
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
//     printGrid(std::string("result"), result, 0, grid_points_1d, 0, grid_points_1d);
    MPI_Barrier(MPI_COMM_WORLD);
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
std::size_t solve(double* grid, double* b, std::size_t cg_max_iterations, double cg_eps, MPI_Comm cartcomm) {
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
	
//     std::cout << "product operator grid with d" << std::endl;
	g_product_operator(grid, d);
	g_scale_add(b, d, -1.0);
	g_copy(r, b);
	g_copy(d, r);

	// calculate starting norm
//     std::cout << "dot product r with r" << std::endl;
	delta_new = g_dot_product(r, r);
	delta_0 = delta_new*eps_squared;
	residuum = (delta_0/eps_squared);
	
    if (rank == 0) {
        std::cout << "Starting norm of residuum: " << (delta_0/eps_squared) << std::endl;
        std::cout << "Target norm:               " << (delta_0) << std::endl;
    }

	while ((needed_iters < cg_max_iterations) && (delta_new > delta_0)) {
		// q = A*d
//         std::cout << "product operator d with q" << std::endl;
		g_product_operator(d, q);

		// a = d_new / d.q
//         std::cout << "dot product d with q" << std::endl;
		a = delta_new/g_dot_product(d, q);
		
		// x = x + a*d
		g_scale_add(grid, d, a);
		
		if ((needed_iters % 50) == 0)
		{
			g_copy(b, b_save);
//             std::cout << "product operator grid with q" << std::endl;
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
//         std::cout << "dot product r with r" << std::endl;
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
    
    // collect
    /*double* collect = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
    
    MPI_Datatype patch, sizedpatch;
    int sizes[2] = {grid_points_1d, grid_points_1d},
        subsizes[2] = {y_max-y_min, x_max-x_min},
        starts[2] = {y_min, x_min};
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &patch);
    MPI_Type_create_resized(patch, 0, sizeof(double), &sizedpatch);
    MPI_Type_commit(&sizedpatch);
    
    MPI_Allgather(grid, 1, sizedpatch, collect, 1, sizedpatch, MPI_COMM_WORLD);
    //if (rank == 0) MPI_Gather(MPI_IN_PLACE, 1, patch, grid, 1, patch, 0, MPI_COMM_WORLD);
    //else MPI_Gather(grid, 1, patch, grid, 1, patch, 0, MPI_COMM_WORLD);
    printGrid(std::string("grid"), collect, 0, grid_points_1d, 0, grid_points_1d);
    
    MPI_Type_free(&sizedpatch);
    _mm_free(collect);*/
    int tag = 17;
    if (rank != 0) {
        MPI_Request req;
        MPI_Status stat;
        MPI_Datatype vectortype;
        MPI_Type_vector(y_max-y_min, x_max-x_min, grid_points_1d, MPI_DOUBLE, &vectortype);
        MPI_Type_commit(&vectortype);
        MPI_Isend(&grid[y_min*grid_points_1d+x_min], 1, vectortype, 0, tag, MPI_COMM_WORLD, &req);
        MPI_Waitall(1, &req, &stat);
        MPI_Type_free(&vectortype);
    } else {
        MPI_Request reqs[size-1];
        MPI_Status stats[size-1];
        double **inbuf;
        inbuf = (double**) _mm_malloc((size-1)*sizeof(double*), ALIGNMENT);
        int maxbufsize = (y_max-y_min)*(x_max-x_min);
        for (int i = 1; i < size; i++) {
            inbuf[i-1] = (double*) _mm_malloc(maxbufsize*sizeof(double), ALIGNMENT);
            MPI_Irecv(inbuf[i-1], maxbufsize, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &reqs[i-1]);
        }
        MPI_Waitall(size-1, reqs, stats);
        
        for (int i = 1; i < size; i++) {
            int coords[2];
            MPI_Cart_coords(cartcomm, i, 2, coords);
            
            int x_first, y_first;
            int x_remainder = grid_points_1d % v_procs, y_remainder = grid_points_1d % h_procs;
            int x_len = grid_points_1d / v_procs, y_len = grid_points_1d / h_procs;
            if (coords[0] < y_remainder) {
                y_len++;
                y_first = y_len * coords[0];
            } else
                y_first = y_len * coords[0] + y_remainder;
            if (coords[1] < x_remainder) {
                x_len++;
                x_first = x_len * coords[1];
            } else
                x_first = x_len * coords[1] + x_remainder;
            
            for (int row = 0; row < y_len; row++) {
                memcpy(&grid[(y_first+row)*grid_points_1d+x_first], inbuf[i-1]+row*x_len, x_len*sizeof(double));
            }
            _mm_free(inbuf[i-1]);
        }
        _mm_free(inbuf);
    }
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
    
//     char name[1024];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
//     gethostname(name, 1024);
    
//     std::stringstream ss;
//     ss << rank;
//     std::string out_file = "out" + ss.str() + ".txt";
//     std::ofstream out(out_file.c_str());
//     std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
//     std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out!
//     
//     std::cout << "Hello, this is (rank " << rank << " of " << size << " on " << name << ")" << std::endl;
	
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
    //if (grid_points_1d % h_procs == 1) h_procs--;
    //if (grid_points_1d % v_procs == 1) v_procs--;

	// calculate grid points per dimension
	grid_points_1d = (std::size_t)(1.0/mesh_width)+1;
	
	int dims[2] = {h_procs, v_procs}, periods[2] = {0, 0}, reorder = 0;
    int coords[2];
    MPI_Comm cartcomm;
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    
    int x_remainder = grid_points_1d % v_procs, y_remainder = grid_points_1d % h_procs;
    int x_len = grid_points_1d / v_procs, y_len = grid_points_1d / h_procs;
//     std::cout << "coords[0] " << coords[0] << "; coords[1] " << coords[1] << "; x_len " << x_len << "; y_len " << y_len << std::endl;
    if (coords[0] < y_remainder) {
        y_len++;
        y_min = y_len * coords[0];
    } else
        y_min = y_len * coords[0] + y_remainder;
    if (coords[1] < x_remainder) {
        x_len++;
        x_min = x_len * coords[1];
    } else
        x_min = x_len * coords[1] + x_remainder;
//     std::cout << "x_remainder " << x_remainder << "; y_remainder " << y_remainder << "; x_len " << x_len << "; y_len " << y_len << std::endl;
    x_max = x_min + x_len;
    y_max = y_min + y_len;
//     std::cout << "x_min " << x_min << "; x_max " << x_max << "; y_min " << y_min << "; y_max " << y_max << std::endl;
    
    i_min = nbrs[UP] >= 0 ? y_min : y_min+1;
    i_max = nbrs[DOWN] >= 0 ? y_max : y_max-1;
    j_min = nbrs[LEFT] >= 0 ? x_min : x_min+1;
    j_max = nbrs[RIGHT] >= 0 ? x_max : x_max-1;
//     std::cout << "i_min " << i_min << "; i_max " << i_max << "; j_min " << j_min << "; j_max " << j_max << std::endl;
    
    // initialize the gird and rights hand side
	double* grid = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
	double* b = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), ALIGNMENT);
	init_grid(grid);
// 	if (rank == 0) store_grid(grid, "initial_condition.gnuplot");
	init_b(b);
// 	if (rank == 0) store_grid(b, "b.gnuplot");
	
	// solve Poisson equation using CG method
	timer_start();
	solve(grid, b, cg_max_iterations, cg_eps, cartcomm);
	double time = timer_stop();
    
//     std::string store_file = "solution" + ss.str() + ".gnuplot";
// 	store_grid(grid, store_file);
	
	if (rank == 0) {
//         store_grid(grid, "solution.gnuplot");
        std::cout << std::endl << "Needed time: " << time << " s" << std::endl << std::endl;
    }
	
	_mm_free(grid);
	_mm_free(b);
    
//     std::cout.rdbuf(coutbuf);
    
    MPI_Finalize();

	return 0;
}


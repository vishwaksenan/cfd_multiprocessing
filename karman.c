#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <mpi.h>
#include <omp.h>
#include "alloc.h"
#include "boundary.h"
#include "datadef.h"
#include "init.h"
#include "simulation.h"
#include "mpi_process.h"


void write_bin(float **u, float **v, float **p, char **flag,
     int imax, int jmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char *file);

static void print_usage(void);
static void print_version(void);
static void print_help(void);

static char *progname;

int proc = 0;                       /* Rank of the current process */
int nprocs = 0;                /* Number of processes in communicator */

int *ileft, *iright;           /* Array bounds for each processor */

#define PACKAGE "karman"
#define VERSION "1.0"

/* Command line options */
static struct option long_opts[] = {
    { "del-t",   1, NULL, 'd' },
    { "help",    0, NULL, 'h' },
    { "imax",    1, NULL, 'x' },
    { "infile",  1, NULL, 'i' },
    { "jmax",    1, NULL, 'y' },
    { "outfile", 1, NULL, 'o' },
    { "t-end",   1, NULL, 't' },
    { "verbose", 1, NULL, 'v' },
    { "version", 1, NULL, 'V' },
    { 0,         0, 0,    0   } 
};
#define GETOPTS "d:hi:o:t:v:Vx:y:"

int main(int argc, char *argv[])
{
    int verbose = 2;          /* Verbosity level */
    float xlength = 22.0;     /* Width of simulated domain */
    float ylength = 4.1;      /* Height of simulated domain */
    int imax = 1000;           /* Number of cells horizontally */
    int jmax = 120;           /* Number of cells vertically */

    char *infile;             /* Input raw initial conditions */
    char *outfile;            /* Output raw simulation results */

    float t_end = 2.1;        /* Simulation runtime */
    float del_t = 0.003;      /* Duration of each timestep */
    float tau = 0.5;          /* Safety factor for timestep control */

    int itermax = 100;        /* Maximum number of iterations in SOR */
    float eps = 0.001;        /* Stopping error threshold for SOR */
    float omega = 1.7;        /* Relaxation parameter for SOR */
    float gamma = 0.9;        /* Upwind differencing factor in PDE
                                discretisation */

    float Re = 150.0;         /* Reynolds number */
    float ui = 1.0;           /* Initial X velocity */
    float vi = 0.0;           /* Initial Y velocity */

    float t, delx, dely;
    int  i, j, itersor = 0, ifluid = 0, ibound = 0;
    float res;
    float **u, **v, **p, **rhs, **f, **g;
    char  **flag;
    int init_case, iters = 0;
    int show_help = 0, show_usage = 0, show_version = 0;

    progname = argv[0];
    infile = strdup("karman.bin");
    outfile = strdup("karman.bin");

    int optc;
    while ((optc = getopt_long(argc, argv, GETOPTS, long_opts, NULL)) != -1) {
        switch (optc) {
            case 'h':
                show_help = 1;
                break;
            case 'V':
                show_version = 1;
                break;
            case 'v':
                verbose = atoi(optarg);
                break;
            case 'x':
                imax = atoi(optarg);
                break;
            case 'y':
                jmax = atoi(optarg);
                break;
            case 'i':
                free(infile);
                infile = strdup(optarg);
                break;
            case 'o':
                free(outfile);
                outfile = strdup(optarg);
                break;
            case 'd':
                del_t = atof(optarg);
                break;
            case 't':
                t_end = atof(optarg);
                break;
            default:
                show_usage = 1;
        }
    }
    if (show_usage || optind < argc) {
        print_usage();
        return 1;
    }
    
    if (show_version) {
        print_version();
        if (!show_help) {
            return 0;
        }
    }
    
    if (show_help) {
        print_help();
        return 0;
    }

    delx = xlength/imax;
    dely = ylength/jmax;

    /* Allocate arrays */
    printf("Before allocation\n");
    u    = alloc_floatmatrix(imax+2, jmax+2);
    v    = alloc_floatmatrix(imax+2, jmax+2);
    f    = alloc_floatmatrix(imax+2, jmax+2);
    g    = alloc_floatmatrix(imax+2, jmax+2);
    p    = alloc_floatmatrix(imax+2, jmax+2);
    rhs  = alloc_floatmatrix(imax+2, jmax+2); 
    flag = alloc_charmatrix(imax+2, jmax+2);                    

    if (!u || !v || !f || !g || !p || !rhs || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        return 1;
    }

    /* Read in initial values from a file if it exists */
    init_case = read_bin(u, v, p, flag, imax, jmax, xlength, ylength, infile);
        
    if (init_case > 0) {
        /* Error while reading file */
        return 1;
    }

    if (init_case < 0) {
        /* Set initial values if file doesn't exist */
        #pragma omp parallel for
        for (i=0;i<=imax+1;i++) {
            for (j=0;j<=jmax+1;j++) {
                u[i][j] = ui;
                v[i][j] = vi;
                p[i][j] = 0.0;
            }
        }
        init_flag(flag, imax, jmax, delx, dely, &ibound);
        apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
    }

    int world_size, world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int *details = get_sub_grid_details(imax, jmax, world_rank, world_size);

    int x_local_size = details[0];
    int y_local_size = details[1] - 2;
    int x_min_local = details[2];
    int x_max_local = details[3];
    int y_min_local = details[4] + 1;
    int y_max_local = details[5] - 1;
    int top_rank = world_rank - 1;
    int bot_rank = world_rank + 1;

    // Creating the sub array
    float **sub_u    = alloc_floatmatrix(x_local_size+2, y_local_size + 2);
    float **sub_v     = alloc_floatmatrix(x_local_size+2, y_local_size + 2);
    float **sub_f    = alloc_floatmatrix(x_local_size+2, y_local_size + 2);
    float **sub_g    = alloc_floatmatrix(x_local_size+2, y_local_size + 2);
    float **sub_p    = alloc_floatmatrix(x_local_size+2, y_local_size + 2);
    float **sub_rhs  = alloc_floatmatrix(x_local_size+2, y_local_size + 2); 
    char ** sub_flag = alloc_charmatrix(x_local_size+2, y_local_size + 2);
    // update the sub array
    update_subarray(sub_u, u, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1);
    update_subarray(sub_v, v, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1);
    update_subarray(sub_f, f, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1);
    update_subarray(sub_g, g, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1);
    update_subarray(sub_p, p, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1);
    update_subarray(sub_rhs, rhs, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1);
    update_char_subarray(sub_flag, flag, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1);

    // Updating halos for the sub_array
    update_halos_for_subarray(sub_u, u, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1, top_rank, bot_rank, world_size);
    update_halos_for_subarray(sub_v, v, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1, top_rank, bot_rank, world_size);
    update_halos_for_subarray(sub_f, f, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1, top_rank, bot_rank, world_size);
    update_halos_for_subarray(sub_g, g, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1, top_rank, bot_rank, world_size);
    update_halos_for_subarray(sub_p, p, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1, top_rank, bot_rank, world_size);
    update_halos_for_subarray(sub_rhs, rhs, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1, top_rank, bot_rank, world_size);
    update_halos_for_char_subarray(sub_flag, flag, x_min_local, x_max_local, y_min_local - 1, y_max_local + 1, top_rank, bot_rank, world_size);

    //Creating size for sending and receiving data
    float *sub_u_top_send_data, *sub_v_top_send_data, *sub_f_top_send_data, *sub_g_top_send_data, *sub_p_top_send_data, *sub_rhs_top_send_data;
    float *sub_u_bot_send_data, *sub_v_bot_send_data, *sub_f_bot_send_data, *sub_g_bot_send_data, *sub_p_bot_send_data, *sub_rhs_bot_send_data;

    float *sub_u_top_recv_data, *sub_v_top_recv_data, *sub_f_top_recv_data, *sub_g_top_recv_data, *sub_p_top_recv_data, *sub_rhs_top_recv_data;
    float *sub_u_bot_recv_data, *sub_v_bot_recv_data, *sub_f_bot_recv_data, *sub_g_bot_recv_data, *sub_p_bot_recv_data, *sub_rhs_bot_recv_data;

    sub_u_top_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));   
    sub_v_top_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_f_top_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_g_top_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_p_top_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_rhs_top_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));

    sub_u_bot_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_v_bot_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_f_bot_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_g_bot_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_p_bot_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    sub_rhs_bot_recv_data = (float *)malloc((y_local_size + 2) * sizeof(float));
    MPI_Status stat;
    
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        //do the process on the sub_array
        set_timestep_interval(&del_t, x_local_size, y_local_size, delx, dely, sub_u, sub_v, Re, tau);

        ifluid = (x_local_size * y_local_size) - ibound;

        compute_tentative_velocity(sub_u, sub_v, sub_f, sub_g, sub_flag, x_local_size, y_local_size,
            del_t, delx, dely, gamma, Re);

        compute_rhs(sub_f, sub_g, sub_rhs, sub_flag, x_local_size, y_local_size, del_t, delx, dely);
        if (ifluid > 0) {
            itersor = poisson(sub_p, sub_rhs, sub_flag, x_local_size, y_local_size, delx, dely,
                        eps, itermax, omega, &res, ifluid);
        } else {
            itersor = 0;
        }

        if (proc == 0 && verbose > 1) {
            printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                iters, t+del_t, del_t, itersor, res, ibound);
        }

        update_velocity(sub_u, sub_v, sub_f, sub_g, sub_p, sub_flag, x_local_size, y_local_size, del_t, delx, dely);

        apply_boundary_conditions(sub_u, sub_v, sub_flag, x_local_size, y_local_size, ui, vi);

        // update halos of the sub_array for sending the top array
        sub_u_top_send_data = (float *)sub_u[1];
        sub_v_top_send_data = (float *)sub_v[1];
        sub_f_top_send_data = (float *)sub_f[1];
        sub_g_top_send_data = (float *)sub_g[1];
        sub_p_top_send_data = (float *)sub_p[1];
        sub_rhs_top_send_data = (float *)sub_rhs[1];

        // update halos of the sub_array for sending the bot array
        sub_u_bot_send_data = (float *)sub_u[x_local_size];
        sub_v_bot_send_data = (float *)sub_v[x_local_size];
        sub_f_bot_send_data = (float *)sub_f[x_local_size];
        sub_g_bot_send_data = (float *)sub_g[x_local_size];
        sub_p_bot_send_data = (float *)sub_p[x_local_size];
        sub_rhs_bot_send_data = (float *)sub_rhs[x_local_size];

        // if(world_rank == 0){
        //     printf("-------------------------------------------------------------\n");
        //     printf("Message from Rank %d\n", world_rank);
        //     printf("Sending bot buffer data to botton_buffer to world 2\n");
        //     for(int i = 0;i<cols + 2; i++){
        //         printf("%d ", bot_send_data[i]);
        //     }
        //     printf("\n");
        // }

        if(top_rank == -1 && bot_rank==1){
            MPI_Sendrecv(sub_u_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 12345, sub_u_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 12345, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_v_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 25, sub_v_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 25, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_f_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 63, sub_f_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 63, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_g_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 69, sub_g_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 69, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_p_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 8, sub_p_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 8, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_rhs_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 71, sub_rhs_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 71, MPI_COMM_WORLD, &stat);
        }

        else if(bot_rank == world_size && top_rank == world_size - 2){
            MPI_Sendrecv(sub_u_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 12345, sub_u_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 12345, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_v_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 25, sub_v_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 25, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_f_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 63, sub_f_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 63, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_g_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 69, sub_g_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 69, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_p_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 8, sub_p_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 8, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_rhs_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 71, sub_rhs_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 71, MPI_COMM_WORLD, &stat);
        }
        else{
            MPI_Sendrecv(sub_u_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 12345, sub_u_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 12345, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_u_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 12345, sub_u_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 12345, MPI_COMM_WORLD, &stat);

            MPI_Sendrecv(sub_v_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 25, sub_v_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 25, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_v_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 25, sub_v_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 25, MPI_COMM_WORLD, &stat);

            MPI_Sendrecv(sub_f_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 63, sub_f_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 63, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_f_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 63, sub_f_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 63, MPI_COMM_WORLD, &stat);

            MPI_Sendrecv(sub_g_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 69, sub_g_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 69, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_g_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 69, sub_g_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 69, MPI_COMM_WORLD, &stat);

            MPI_Sendrecv(sub_p_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 8, sub_p_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 8, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_p_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 8, sub_p_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 8, MPI_COMM_WORLD, &stat);

            MPI_Sendrecv(sub_rhs_top_send_data, (y_local_size + 2), MPI_INT, top_rank, 71, sub_rhs_bot_recv_data, (y_local_size + 2), MPI_INT, bot_rank, 71, MPI_COMM_WORLD, &stat);
            MPI_Sendrecv(sub_rhs_bot_send_data, (y_local_size + 2), MPI_INT, bot_rank, 71, sub_rhs_top_recv_data, (y_local_size + 2), MPI_INT, top_rank, 71, MPI_COMM_WORLD, &stat);
        }

        // write the top data in the 0th row
        // write the bot data in the (size + 1) row
        #pragma omp parallel for
        for(int i = 0; i<y_local_size + 2;i++){
            sub_u[0][i] = sub_u_top_recv_data[i];
            sub_u[x_local_size + 1][i] = sub_u_bot_recv_data[i];

            sub_v[0][i] = sub_v_top_recv_data[i];
            sub_v[x_local_size + 1][i] = sub_v_bot_recv_data[i];

            sub_f[0][i] = sub_f_top_recv_data[i];
            sub_f[x_local_size + 1][i] = sub_f_bot_recv_data[i];

            sub_g[0][i] = sub_g_top_recv_data[i];
            sub_g[x_local_size + 1][i] = sub_g_bot_recv_data[i];

            sub_p[0][i] = sub_p_top_recv_data[i];
            sub_p[x_local_size + 1][i] = sub_p_bot_recv_data[i];

            sub_rhs[0][i] = sub_rhs_top_recv_data[i];
            sub_rhs[x_local_size + 1][i] = sub_rhs_bot_recv_data[i];
        }
    } /* End of main loop */
    MPI_Barrier(MPI_COMM_WORLD);

    float **u_global = alloc_floatmatrix(imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    float **v_global = alloc_floatmatrix(imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    float **p_global = alloc_floatmatrix(imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);

    set_float_zero(u_global, imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    set_float_zero(v_global, imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    set_float_zero(p_global, imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);

    copy_to_global_array(sub_u, u_global, x_min_local, x_max_local, y_min_local, y_max_local);
    MPI_Barrier(MPI_COMM_WORLD);
    copy_to_global_array(sub_v, v_global, x_min_local, x_max_local, y_min_local, y_max_local);
    MPI_Barrier(MPI_COMM_WORLD);
    copy_to_global_array(sub_p, p_global, x_min_local, x_max_local, y_min_local, y_max_local);
    MPI_Barrier(MPI_COMM_WORLD);
    float *u_send_vect = vectorize(u_global, imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    float *v_send_vect = vectorize(v_global, imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    float *p_send_vect = vectorize(p_global, imax + 2, jmax + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    float *u_recv_vect = (float *)malloc(((imax + 2) * (jmax + 2)) * sizeof(float));
    MPI_Barrier(MPI_COMM_WORLD);
    float *v_recv_vect = (float *)malloc(((imax + 2) * (jmax + 2)) * sizeof(float));
    MPI_Barrier(MPI_COMM_WORLD);
    float *p_recv_vect = (float *)malloc(((imax + 2) * (jmax + 2)) * sizeof(float));
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Reduce(u_send_vect, u_recv_vect, (imax + 2)*(jmax + 2), MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(v_send_vect, v_recv_vect, (imax + 2)*(jmax + 2), MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(p_send_vect, p_recv_vect, (imax + 2)*(jmax + 2), MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    


    if(world_rank == 0){
        float **u_devectorize = devectorize(u_recv_vect, imax + 2, jmax + 2);
        float **v_devectorize = devectorize(v_recv_vect, imax + 2, jmax + 2);
        float **p_devectorize = devectorize(p_recv_vect, imax + 2, jmax + 2);
        
    
        if (outfile != NULL && strcmp(outfile, "") != 0 && proc == 0) {
            write_bin(u_devectorize, v_devectorize, p_devectorize, flag, imax, jmax, xlength, ylength, outfile);
        }
    }
    

    free_matrix(u);
    free_matrix(v);
    free_matrix(f);
    free_matrix(g);
    free_matrix(p);
    free_matrix(rhs);
    free_matrix(flag);
    
    MPI_Finalize();
    return 0;
}

/* Save the simulation state to a file */
void write_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i;
    FILE *fp;

    fp = fopen(file, "wb"); 

    if (fp == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        return;
    }

    fwrite(&imax, sizeof(int), 1, fp);
    fwrite(&jmax, sizeof(int), 1, fp);
    fwrite(&xlength, sizeof(float), 1, fp);
    fwrite(&ylength, sizeof(float), 1, fp);

    for (i=0;i<imax+2;i++) {
        fwrite(u[i], sizeof(float), jmax+2, fp);
        fwrite(v[i], sizeof(float), jmax+2, fp);
        fwrite(p[i], sizeof(float), jmax+2, fp);
        fwrite(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
}

/* Read the simulation state from a file */
int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i,j;
    FILE *fp;

    if (file == NULL) return -1;

    if ((fp = fopen(file, "rb")) == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        fprintf(stderr, "Generating default state instead.\n");
        return -1;
    }

    fread(&i, sizeof(int), 1, fp);
    fread(&j, sizeof(int), 1, fp);
    float xl, yl;
    fread(&xl, sizeof(float), 1, fp);
    fread(&yl, sizeof(float), 1, fp);

    if (i!=imax || j!=jmax) {
        fprintf(stderr, "Warning: imax/jmax have wrong values in %s\n", file);
        fprintf(stderr, "%s's imax = %d, jmax = %d\n", file, i, j);
        fprintf(stderr, "Program's imax = %d, jmax = %d\n", imax, jmax);
        return 1;
    }
    if (xl!=xlength || yl!=ylength) {
        fprintf(stderr, "Warning: xlength/ylength have wrong values in %s\n", file);
        fprintf(stderr, "%s's xlength = %g,  ylength = %g\n", file, xl, yl);
        fprintf(stderr, "Program's xlength = %g, ylength = %g\n", xlength,
            ylength);
        return 1;
    }

    for (i=0; i<imax+2; i++) {
        fread(u[i], sizeof(float), jmax+2, fp);
        fread(v[i], sizeof(float), jmax+2, fp);
        fread(p[i], sizeof(float), jmax+2, fp);
        fread(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
    return 0;
}

static void print_usage(void)
{
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
}

static void print_version(void)
{
    fprintf(stderr, "%s %s\n", PACKAGE, VERSION);
}

static void print_help(void)
{
    fprintf(stderr, "%s. A simple computational fluid dynamics tutorial.\n\n",
        PACKAGE);
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", progname);
    fprintf(stderr, "  -h, --help            Print a summary of the options\n");
    fprintf(stderr, "  -V, --version         Print the version number\n");
    fprintf(stderr, "  -v, --verbose=LEVEL   Set the verbosity level. 0 is silent\n");
    fprintf(stderr, "  -x, --imax=IMAX       Set the number of interior cells in the X direction\n");
    fprintf(stderr, "  -y, --jmax=JMAX       Set the number of interior cells in the Y direction\n");
    fprintf(stderr, "  -t, --t-end=TEND      Set the simulation end time\n");
    fprintf(stderr, "  -d, --del-t=DELT      Set the simulation timestep size\n");
    fprintf(stderr, "  -i, --infile=FILE     Read the initial simulation state from this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
    fprintf(stderr, "  -o, --outfile=FILE    Write the final simulation state to this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
}

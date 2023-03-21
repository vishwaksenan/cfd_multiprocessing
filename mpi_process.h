#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "alloc.h"


void zero_grid(float **grid, int rows, int cols);
void update_subarray(float** subarray, float** source_grid, int x_min, int x_max, int y_min, int y_max);
void update_halos_for_subarray(float** subarray, float** source_grid, int x_min, int x_max, int y_min, int y_max, int top_rank, int bot_rank, int world_size);
float** create_local_true_size(float** sub_array, int rows, int cols, int x_min, int x_max, int y_min, int y_max, int size);
void update_original(float **src, float **dest, int x_min, int x_max, int y_min, int y_max);
float* vectorize(float **grid, int rows, int cols);
float** devectorize(float *vect, int rows, int cols);
int* get_sub_grid_details(int rows, int cols, int current_rank, int world_size);
void set_float_zero(float **arr, int x_size, int y_size);
void copy_to_global_array(float **src, float **dest, int x_min, int x_max, int y_min, int y_max);

void copy_to_global_array(float **src, float **dest, int x_min, int x_max, int y_min, int y_max){
    int row_iter = 0;
    #pragma omp parallel for
    for(int i=x_min;i<=x_max;i++){
        for(int j=y_min-1; j<=y_max+1; j++){
            dest[i][j] = src[row_iter][j];
        }
        row_iter++;
    }
}

void set_float_zero(float **arr, int x_size, int y_size){
    #pragma omp parallel for
    for(int i=0;i<x_size;i++){
        for(int j=0;j<y_size;j++){
            arr[i][j] = 0.0;
        }
    }
}

void zero_grid(float **grid, int rows, int cols){
    #pragma omp parallel for
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            grid[i][j] = 0.0;
        }
    }
}

void update_subarray(float** subarray, float** source_grid, int x_min, int x_max, int y_min, int y_max){
    int row_iter = 1;
    #pragma omp parallel for
    for(int i=x_min;i<=x_max;i++){
        int col_iter = 0;
        for(int j=y_min-1; j<=y_max+1; j++){
            subarray[row_iter][col_iter] = source_grid[i][j];
            col_iter++;
        }
        row_iter++;
    }
}

void update_char_subarray(char** subarray, char** source_grid, int x_min, int x_max, int y_min, int y_max){
    int row_iter = 1;
    #pragma omp parallel for
    for(int i=x_min;i<=x_max;i++){
        int col_iter = 0;
        for(int j=y_min-1; j<=y_max+1; j++){
            subarray[row_iter][col_iter] = source_grid[i][j];
            col_iter++;
        }
        row_iter++;
    }
}

void update_halos_for_subarray(float** subarray, float** source_grid, int x_min, int x_max, int y_min, int y_max, int top_rank, int bot_rank, int world_size){
    if(top_rank!=-1){
        #pragma omp parallel for
        for(int i=0;i<(y_max - y_min +1);i++){
            subarray[0][i] = source_grid[x_min - 1][i];
        }
    }

    if(bot_rank != world_size){
        #pragma omp parallel for
        for(int i=0;i<(y_max - y_min + 1);i++){
            subarray[(x_max - x_min + 2)][i] = source_grid[x_min - 1][i];
        }
    }
}

void update_halos_for_char_subarray(char** subarray, char** source_grid, int x_min, int x_max, int y_min, int y_max, int top_rank, int bot_rank, int world_size){
    if(top_rank!=-1){
        for(int i=0;i<(y_max - y_min + 1);i++){
            subarray[0][i] = source_grid[x_min - 1][i];
        }
    }

    if(bot_rank != world_size){
        for(int i=0;i<(y_max - y_min + 1);i++){
            subarray[(x_max - x_min + 2)][i] = source_grid[x_min - 1][i];
        }
    }
}

float** create_local_true_size(float** sub_array, int rows, int cols, int x_min, int x_max, int y_min, int y_max, int size){
    float** local_zero_grid = alloc_floatmatrix(cols, size);
    zero_grid(local_zero_grid, rows, cols);
    int row_iter = 1;
    for(int i=x_min;i<=x_max;i++){
        int col_iter = 0;
        for(int j=y_min; j<=y_max; j++){
            local_zero_grid[i][j] = sub_array[row_iter][col_iter];
            col_iter++;
        }
        row_iter++;
    }
    return local_zero_grid;
}

void update_original(float **src, float **dest, int x_min, int x_max, int y_min, int y_max){
    for(int i=x_min;i<=x_max;i++){
        for(int j=y_min;j<=y_max;j++){
            dest[i][j] = src[i][j];
        }
    }
}

float* vectorize(float **grid, int rows, int cols){
    float *vec = (float *)malloc((rows * cols) * sizeof(float));
    int k = 0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            vec[k] = grid[i][j];
            k++;
        }
    }
    return vec;
}

float** devectorize(float *vect, int rows, int cols){
    float **grid_2d = alloc_floatmatrix(rows, cols);
    int k = 0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            grid_2d[i][j] = vect[k];
            k++;
        }
    }
    return grid_2d;
}

int* get_sub_grid_details(int rows, int cols, int current_rank, int world_size){
    int x_size, y_size, x_min, x_max, y_min, y_max;
    int* details = (int *)malloc(6 * sizeof(int));
    int divisor = rows / world_size;
    int rem = rows % world_size;
    if(current_rank == world_size - 1){
        x_size = divisor + rem;
        x_max = rows;
        x_min = rows - x_size + 1;
    }
    else{
        x_size = divisor;
        x_min = current_rank * x_size + 1;
        x_max = (current_rank + 1) * (x_size);  
    }
    y_size = cols + 2;
    y_min = 0;
    y_max = cols + 1;
    details[0] = x_size;
    details[1] = y_size;
    details[2] = x_min;
    details[3] = x_max;
    details[4] = y_min;
    details[5] = y_max;
    return details;
}



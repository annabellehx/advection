#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

void get_dimensions(int N, int rank, int size, int *start_row, int *end_row, int *start_col, int *end_col)
{
    int grid_dim_x = (int)sqrt(size);
    int grid_dim_y = size / grid_dim_x;
    int block_rows = N / grid_dim_x;
    int block_cols = N / grid_dim_y;
    int row_index = rank / grid_dim_y;
    int col_index = rank % grid_dim_y;

    *start_row = row_index * block_rows;
    *start_col = col_index * block_cols;
    *end_row = (row_index < grid_dim_x - 1) ? *start_row + block_rows - 1 : N - 1;
    *end_col = (col_index < grid_dim_y - 1) ? *start_col + block_cols - 1 : N - 1;
}

double **initialize_matrix(int local_rows, int local_cols, int NTHREADS)
{
    double **matrix = malloc((local_rows + 2) * sizeof(double *));

#ifdef USE_OMP
#pragma omp parallel for num_threads(NTHREADS)
#endif
    for (int i = 0; i < local_rows + 2; ++i)
        matrix[i] = malloc((local_cols + 2) * sizeof(double));

    matrix[0][0] = 0;
    matrix[0][local_cols + 1] = 0;
    matrix[local_rows + 1][0] = 0;
    matrix[local_rows + 1][local_cols + 1] = 0;

    return matrix;
}

void update_ghost_cells(double **matrix, int rank, int size, int local_rows, int local_cols, int NTHREADS,
                        double *send_up, double *send_down, double *send_left, double *send_right,
                        double *recv_up, double *recv_down, double *recv_left, double *recv_right)
{
    int grid_dim_x = (int)sqrt(size);
    int grid_dim_y = size / grid_dim_x;

    int up_neighbor = (rank - grid_dim_y + size) % size;
    int down_neighbor = (rank + grid_dim_y) % size;
    int left_neighbor = (rank % grid_dim_y > 0) ? rank - 1 : rank + grid_dim_y - 1;
    int right_neighbor = (rank % grid_dim_y < grid_dim_y - 1) ? rank + 1 : rank - grid_dim_y + 1;

#ifdef USE_OMP
#pragma omp parallel num_threads(NTHREADS)
    {
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int i = 0; i < local_cols; ++i)
        {
            send_up[i] = matrix[1][i + 1];
            send_down[i] = matrix[local_rows][i + 1];
        }

#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int i = 0; i < local_rows; ++i)
        {
            send_left[i] = matrix[i + 1][1];
            send_right[i] = matrix[i + 1][local_cols];
        }
#ifdef USE_OMP
    }
#endif

    MPI_Request send_up_request, send_down_request, send_left_request, send_right_request;
    MPI_Request recv_up_request, recv_down_request, recv_left_request, recv_right_request;

    MPI_Isend(send_up, local_cols, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD, &send_up_request);
    MPI_Isend(send_down, local_cols, MPI_DOUBLE, down_neighbor, 1, MPI_COMM_WORLD, &send_down_request);
    MPI_Isend(send_left, local_rows, MPI_DOUBLE, left_neighbor, 2, MPI_COMM_WORLD, &send_left_request);
    MPI_Isend(send_right, local_rows, MPI_DOUBLE, right_neighbor, 3, MPI_COMM_WORLD, &send_right_request);

    MPI_Irecv(recv_down, local_cols, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, &recv_down_request);
    MPI_Irecv(recv_up, local_cols, MPI_DOUBLE, up_neighbor, 1, MPI_COMM_WORLD, &recv_up_request);
    MPI_Irecv(recv_right, local_rows, MPI_DOUBLE, right_neighbor, 2, MPI_COMM_WORLD, &recv_right_request);
    MPI_Irecv(recv_left, local_rows, MPI_DOUBLE, left_neighbor, 3, MPI_COMM_WORLD, &recv_left_request);

    MPI_Wait(&send_up_request, MPI_STATUS_IGNORE);
    MPI_Wait(&send_down_request, MPI_STATUS_IGNORE);
    MPI_Wait(&send_left_request, MPI_STATUS_IGNORE);
    MPI_Wait(&send_right_request, MPI_STATUS_IGNORE);

    MPI_Wait(&recv_up_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_down_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_left_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_right_request, MPI_STATUS_IGNORE);

#ifdef USE_OMP
#pragma omp parallel num_threads(NTHREADS)
    {
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int i = 0; i < local_rows; ++i)
        {
            matrix[i + 1][0] = recv_left[i];
            matrix[i + 1][local_cols + 1] = recv_right[i];
        }

#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int i = 0; i < local_cols; ++i)
        {
            matrix[0][i + 1] = recv_up[i];
            matrix[local_rows + 1][i + 1] = recv_down[i];
        }
#ifdef USE_OMP
    }
#endif
}

void lax_method(double **C, double **C_next, double dt, double dx, double u, double v, int i, int j)
{
    C_next[i][j] = (C[i - 1][j] + C[i + 1][j] + C[i][j - 1] + C[i][j + 1]) / 4.0;
    C_next[i][j] -= dt / (2 * dx) * (u * (C[i + 1][j] - C[i - 1][j]) + v * (C[i][j + 1] - C[i][j - 1]));
}

void free_matrix(double **matrix, int local_rows, int NTHREADS)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(NTHREADS)
#endif
    for (int i = 0; i < local_rows + 2; ++i)
        free(matrix[i]);

    free(matrix);
}

void print_matrix(FILE *file, double **local_matrix, int rank, int size, int N, int local_rows, int local_cols)
{
    int local_size = local_rows * local_cols;
    double *local_data = malloc(local_size * sizeof(double));
    double *global_data = NULL;

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < local_cols; ++j)
            local_data[i * local_cols + j] = local_matrix[i + 1][j + 1];

    if (rank == 0)
        global_data = malloc(N * N * sizeof(double));

    MPI_Gather(local_data, local_size, MPI_DOUBLE, global_data, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                int local_rank = (i / (N / (int)sqrt(size))) * (int)sqrt(size) + (j / (N / (int)sqrt(size)));
                int local_i = i % (N / (int)sqrt(size));
                int local_j = j % (N / (int)sqrt(size));
                fprintf(file, "%.2lf ", global_data[local_rank * local_size + local_i * local_cols + local_j]);
            }

    free(local_data);

    if (rank == 0)
    {
        fprintf(file, "\n");
        free(global_data);
    }
}

int advection(int N, double L, double T, int rank, int size, int NTHREADS)
{
    double dx = L / (N - 1);
    double dt = 0.000125;
    int NT = T / dt;

    int start_row, end_row, start_col, end_col;
    get_dimensions(N, rank, size, &start_row, &end_row, &start_col, &end_col);

    int local_rows = end_row - start_row + 1;
    int local_cols = end_col - start_col + 1;
    double *u = malloc(local_cols * sizeof(double));
    double *v = malloc(local_rows * sizeof(double));

    double *send_up = malloc(local_cols * sizeof(double));
    double *send_down = malloc(local_cols * sizeof(double));
    double *recv_up = malloc(local_cols * sizeof(double));
    double *recv_down = malloc(local_cols * sizeof(double));

    double *send_left = malloc(local_rows * sizeof(double));
    double *send_right = malloc(local_rows * sizeof(double));
    double *recv_left = malloc(local_rows * sizeof(double));
    double *recv_right = malloc(local_rows * sizeof(double));

    double **C_curr = initialize_matrix(local_rows, local_cols, NTHREADS);
    double **C_next = initialize_matrix(local_rows, local_cols, NTHREADS);

    // FILE *file = fopen("matrix.txt", "w");

#ifdef USE_OMP
#pragma omp parallel for num_threads(NTHREADS)
#endif
    for (int i = 0; i < local_rows; ++i)
    {
        double x = -L / 2 + (start_row + i) * dx;
        v[i] = -sqrt(2) * x;

        for (int j = 0; j < local_cols; ++j)
        {
            double y = -L / 2 + (start_col + j) * dx;
            u[j] = sqrt(2) * y;
            C_curr[i + 1][j + 1] = (int)(fabs(x) <= L / 2 && fabs(y) <= 0.1);
        }
    }

    update_ghost_cells(C_curr, rank, size, local_rows, local_cols, NTHREADS, send_up, send_down, send_left, send_right, recv_up, recv_down, recv_left, recv_right);

    // print_matrix(file, C_curr, rank, size, N, local_rows, local_cols);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int n = 0; n < NT; ++n)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(NTHREADS)
#endif
        for (int i = 0; i < local_rows; ++i)
            for (int j = 0; j < local_cols; ++j)
                lax_method(C_curr, C_next, dt, dx, u[j], v[i], i + 1, j + 1);

        update_ghost_cells(C_next, rank, size, local_rows, local_cols, NTHREADS, send_up, send_down, send_left, send_right, recv_up, recv_down, recv_left, recv_right);

        double **temp = C_curr;
        C_curr = C_next;
        C_next = temp;

        // if (n == (NT - 1) / 2 || n == NT - 1)
        // print_matrix(file, C_curr, rank, size, N, local_rows, local_cols);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double stop = MPI_Wtime();
    // fclose(file);

    free_matrix(C_curr, local_rows, NTHREADS);
    free_matrix(C_next, local_rows, NTHREADS);
    free(u);
    free(v);

    free(send_up);
    free(send_down);
    free(recv_up);
    free(recv_down);
    free(send_left);
    free(send_right);
    free(recv_left);
    free(recv_right);

    if (rank == 0)
    {
        double total_time = stop - start;
        printf("Total Time : %lf (sec)\n", total_time);
        printf("Grind Rate : %lf (cells/sec)\n\n", (double)N * N * NT / total_time);
    }

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <N> <L> <T> <NTHREADS> \n", argv[0]);
        return EXIT_FAILURE;
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);
    double L = atof(argv[2]);
    double T = atof(argv[3]);
    int NTHREADS = atoi(argv[4]);

    if (rank == 0)
    {
        printf("N = %d   L = %lf   T = %lf \n", N, L, T);
        printf("Approximate Amount of Memory Required :\t%lu bytes \n", 2 * size * (N / (int)sqrt(size) + 2) * (N / (int)sqrt(size) + 2) * sizeof(double));
        printf("Number of Nodes Used :\t\t\t%d nodes \n", size);
        printf("Number of Cores for Parallelizing :\t%d cores \n\n", NTHREADS);
    }

    advection(N, L, T, rank, size, NTHREADS);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

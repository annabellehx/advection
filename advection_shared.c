#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

double **initialize_matrix(int N, int NTHREADS)
{
    double **matrix = malloc((N + 2) * sizeof(double *));

#pragma omp parallel for num_threads(NTHREADS) default(none) shared(N, matrix)
    for (int i = 0; i < N + 2; ++i)
        matrix[i] = malloc((N + 2) * sizeof(double));

    matrix[0][0] = 0;
    matrix[0][N + 1] = 0;
    matrix[N + 1][0] = 0;
    matrix[N + 1][N + 1] = 0;

    return matrix;
}

void update_ghost_cells(double **matrix, int N, int NTHREADS)
{
#pragma omp parallel for num_threads(NTHREADS) default(none) shared(N, matrix)
    for (int i = 0; i < N; ++i)
    {
        matrix[0][i + 1] = matrix[N][i + 1];
        matrix[i + 1][0] = matrix[i + 1][N];
        matrix[N + 1][i + 1] = matrix[1][i + 1];
        matrix[i + 1][N + 1] = matrix[i + 1][1];
    }
}

void lax_method(double **C, double **C_next, double dt, double dx, double u, double v, int i, int j)
{
    C_next[i][j] = (C[i - 1][j] + C[i + 1][j] + C[i][j - 1] + C[i][j + 1]) / 4.0;
    C_next[i][j] -= dt / (2 * dx) * (u * (C[i + 1][j] - C[i - 1][j]) + v * (C[i][j + 1] - C[i][j - 1]));
}

void free_matrix(double **matrix, int N, int NTHREADS)
{
#pragma omp parallel for num_threads(NTHREADS) default(none) shared(N, matrix)
    for (int i = 0; i < N + 2; ++i)
        free(matrix[i]);

    free(matrix);
}

void print_matrix(FILE *file, double **matrix, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            fprintf(file, "%.2lf ", matrix[i + 1][j + 1]);

    fprintf(file, "\n");
}

int advection(int N, double L, double T, int NTHREADS)
{
    double dx = L / (N - 1);
    double dt = 0.000125;
    int NT = T / dt;

    double *u = malloc(N * sizeof(double));
    double *v = malloc(N * sizeof(double));
    double **C_curr = initialize_matrix(N, NTHREADS);
    double **C_next = initialize_matrix(N, NTHREADS);

    // FILE *file = fopen("matrix.txt", "w");

#pragma omp parallel for num_threads(NTHREADS) default(none) shared(C_curr, N, L, dx, u, v)
    for (int i = 0; i < N; ++i)
    {
        double x = -L / 2 + i * dx;
        v[i] = -sqrt(2) * x;

        for (int j = 0; j < N; ++j)
        {
            double y = -L / 2 + j * dx;
            u[j] = sqrt(2) * y;
            C_curr[i + 1][j + 1] = (int)(fabs(x) <= L / 2 && fabs(y) <= 0.1);
        }
    }

    update_ghost_cells(C_curr, N, NTHREADS);
    // print_matrix(file, C_curr, N);

    double start = omp_get_wtime();

    for (int n = 0; n < NT; ++n)
    {
#pragma omp parallel for num_threads(NTHREADS) default(none) shared(C_curr, C_next, N, dx, dt, u, v)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                lax_method(C_curr, C_next, dt, dx, u[j], v[i], i + 1, j + 1);

        update_ghost_cells(C_next, N, NTHREADS);
        double **temp = C_curr;
        C_curr = C_next;
        C_next = temp;

        // if (n == (NT - 1) / 2 || n == NT - 1)
        // print_matrix(file, C_curr, N);
    }

    double stop = omp_get_wtime();
    // fclose(file);

    free_matrix(C_curr, N, NTHREADS);
    free_matrix(C_next, N, NTHREADS);
    free(u);
    free(v);

    double total_time = stop - start;
    printf("Total Time : %lf (sec)\n", total_time);
    printf("Grind Rate : %lf (cells/sec)\n\n", (double)N * N * NT / total_time);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <N> <L> <T> <NTHREADS> \n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    double L = atof(argv[2]);
    double T = atof(argv[3]);
    int NTHREADS = atoi(argv[4]);

    printf("SHARED MEMORY PARALLEL LAX \n\n");
    printf("N = %d   L = %lf   T = %lf \n", N, L, T);
    printf("Approximate Amount of Memory Required :\t%lu bytes \n", 2 * (N + 2) * (N + 2) * sizeof(double));
    printf("Number of Cores for Parallelizing :\t%d cores \n\n", NTHREADS);

    advection(N, L, T, NTHREADS);
    return EXIT_SUCCESS;
}

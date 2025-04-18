#define OUT_FILE "result.dat"
#define TAU 0.1

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

bool is_boundary(int row, int n) {
    int i = row / n;
    int j = row % n;
    return i == 0 || i == n - 1 || j == 0 || j == n - 1;
}

double get_a(int row, int col, int n) {
    if (is_boundary(row, n)) {
        return row == col ? 1.0 : 0.0;
    }
    if (row == col) return -4.0;
    if (row + 1 == col && (row % n) + 1 < n) return 1.0;
    if (row - 1 == col && (row % n) - 1 >= 0) return 1.0;
    if (row + n == col && (row / n) + 1 < n) return 1.0;
    if (row - n == col && (row / n) - 1 >= 0) return 1.0;
    return 0.0;
}

double norm(double* x, int grid_size) {
    double result = 0.0;
    #pragma acc parallel loop reduction(+:result)
    for (int i = 0; i < grid_size; i++) {
        result += x[i] * x[i];
    }
    return sqrt(result);
}

void mul_mv_sub(double* res, double* A, double* x, double* b, int grid_size, int n) {
    #pragma acc parallel loop
    for (int i = 0; i < grid_size; i++) {
        if (is_boundary(i, n)) {
            res[i] = 0.0;
        } else {
            res[i] = -b[i];
            for (int j = 0; j < grid_size; j++) {
                res[i] += A[i * grid_size + j] * x[j];
            }
        }
    }
}

void next(double* x, double* delta, int grid_size, int n) {
    #pragma acc parallel loop
    for (int i = 0; i < grid_size; i++) {
        if (!is_boundary(i, n)) {
            x[i] -= TAU * delta[i];
        }
    }
}

void init_matrix(double* A, int grid_size, int n) {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            A[i * grid_size + j] = get_a(i, j, n);
        }
    }
}

void init_b(double* b, int grid_size, int n) {
    #pragma acc parallel loop
    for (int i = 0; i < grid_size; i++) {
        b[i] = 0.0;
    }

    double UL = 10.0;
    double UR = 20.0;
    double LR = 30.0;
    double LL = 20.0;
    #pragma acc parallel loop
    for (int j = 0; j < n; j++) {
        double t = (double)j / (n - 1);
        b[j] = UL + t * (UR - UL);
        b[(n - 1) * n + j] = LL + t * (LR - LL);
    }

    #pragma acc parallel loop
    for (int i = 0; i < n; i++) {
        double t = (double)i / (n - 1);
        b[i * n] = UL + t * (LL - UL);
        b[i * n + (n - 1)] = UR + t * (LR - UR);
    }
}

void solve_simple_iter(double* A, double* x, double* b, int grid_size, int n, int max_iters, double acc) {
    double* Axmb = (double*)malloc(grid_size * sizeof(double));
    if (!Axmb) {
        fprintf(stderr, "Failed to allocate memory for Axmb\n");
        exit(1);
    }
    double norm_b = norm(b, grid_size), norm_Axmb;
    int iter = 0;

    do {
        mul_mv_sub(Axmb, A, x, b, grid_size, n);
        norm_Axmb = norm(Axmb, grid_size);
        next(x, Axmb, grid_size, n);
        if (++iter % 100 == 0) {
            printf("%d: %.8lf >= %.8lf\r", iter, norm_Axmb / norm_b, acc);
            fflush(stdout);
        }
    } while (norm_Axmb / norm_b >= acc && iter < max_iters);

    printf("\33[2K\rDone in %d iterations with error %.8lf\n", iter, norm_Axmb / norm_b);
    free(Axmb);
}

void print_grid(double* x, int grid_n) {
    for (int i = 0; i < grid_n; i++) {
        for (int j = 0; j < grid_n; j++) {
            printf("%.2f ", x[i * grid_n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help message")
        ("acc", po::value<double>()->default_value(1e-6), "Set accuracy")
        ("size", po::value<int>()->default_value(128), "Set grid size")
        ("it", po::value<int>()->default_value(1000000), "Set number of iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    double acc = vm["acc"].as<double>();
    int n = vm["size"].as<int>();
    int max_iters = vm["it"].as<int>();
    int grid_size = n * n;

    double* A = (double*)malloc(grid_size * grid_size * sizeof(double));
    double* b = (double*)malloc(grid_size * sizeof(double));
    double* x = (double*)malloc(grid_size * sizeof(double));
    if (!A || !b || !x) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    #pragma acc data create(A[0:grid_size*grid_size], b[0:grid_size], x[0:grid_size]) copyin(grid_size, n)
    {
        init_matrix(A, grid_size, n);
        init_b(b, grid_size, n);
        #pragma acc kernels
        for (int i = 0; i < grid_size; i++) x[i] = 0.0;
        auto t1 = std::chrono::high_resolution_clock::now();
        solve_simple_iter(A, x, b, grid_size, n, max_iters, acc);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        printf("Time taken: %lld ms\n", duration);
    }

    if (n == 10 || n == 13) {
        printf("\nGrid (%dx%d):\n", n, n);
        print_grid(x, n);
    }

    FILE* f = fopen(OUT_FILE, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open output file\n");
        free(A); free(b); free(x);
        return 1;
    }
    fwrite(x, sizeof(double), grid_size, f);
    fclose(f);
    free(A);
    free(b);
    free(x);
    return 0;
}
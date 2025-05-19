#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    int dimension;
    double precision;
    int iter_limit;
    po::options_description config("Configuration");
    config.add_options()
        ("help", "show help")
        ("size", po::value<int>(&dimension)->default_value(256), "grid size (N x N)")
        ("accuracy", po::value<double>(&precision)->default_value(1e-6), "precision goal")
        ("max_iterations", po::value<int>(&iter_limit)->default_value(1000000), "max iterations");
    po::variables_map params;
    po::store(po::parse_command_line(argc, argv, config), params);
    po::notify(params);
    if (params.count("help")) {
        std::cout << config << "\n";
        return 1;
    }
    std::cout << "Launching computation...\n";
    double* matrix = (double*)malloc(dimension * dimension * sizeof(double));
    double* matrix_new = (double*)malloc(dimension * dimension * sizeof(double));
    double* diff = (double*)malloc(dimension * dimension * sizeof(double));
    nvtxRangePushA("setup");
    for (size_t i = 0; i < dimension * dimension; i++) {
        matrix[i] = 0.0;
        matrix_new[i] = 0.0;
        diff[i] = 0.0;
    }
    matrix[0] = 10.0;
    matrix[dimension - 1] = 20.0;
    matrix[dimension * (dimension - 1)] = 30.0;
    matrix[dimension * dimension - 1] = 20.0;
    double ul = matrix[0], ur = matrix[dimension - 1];
    double ll = matrix[dimension * (dimension - 1)], lr = matrix[dimension * dimension - 1];
    for (int i = 1; i < dimension - 1; i++) {
        matrix[i] = ul + (ur - ul) * i / (dimension - 1.0);
        matrix[dimension * (dimension - 1) + i] = ll + (lr - ll) * i / (dimension - 1.0);
        matrix[dimension * i] = ul + (ll - ul) * i / (dimension - 1.0);
        matrix[dimension * i + dimension - 1] = ur + (lr - ur) * i / (dimension - 1.0);
    }
    #pragma acc enter data create(diff[:dimension*dimension]) copyin(matrix[:dimension*dimension], matrix_new[:dimension*dimension])
    nvtxRangePop();
    cublasHandle_t handle;
    cublasCreate(&handle);
    double error = precision + 1.0;
    int iter_count = 0;
    auto start = std::chrono::steady_clock::now();
    nvtxRangePushA("main_loop");
    while (error > precision && iter_count < iter_limit) {
        bool compute_error = (iter_count % 1000 == 0);
        if (compute_error) {
            #pragma acc parallel loop present(matrix, matrix_new, diff)
            for (int i = 1; i < dimension - 1; i++) {
                for (int j = 1; j < dimension - 1; j++) {
                    size_t idx = i * dimension + j;
                    matrix_new[idx] = 0.25 * (
                        matrix[(i + 1) * dimension + j] +
                        matrix[(i - 1) * dimension + j] +
                        matrix[i * dimension + j - 1] +
                        matrix[i * dimension + j + 1]
                    );
                    diff[idx] = fabs(matrix_new[idx] - matrix[idx]);
                }
            }
            int max_idx;
            cublasIdamax(handle, dimension * dimension, diff, 1, &max_idx);
            #pragma acc update self(diff[max_idx-1:1])
            error = diff[max_idx - 1];
        } else {
            #pragma acc parallel loop present(matrix, matrix_new)
            for (int i = 1; i < dimension - 1; i++) {
                for (int j = 1; j < dimension - 1; j++) {
                    size_t idx = i * dimension + j;
                    matrix_new[idx] = 0.25 * (
                        matrix[(i + 1) * dimension + j] +
                        matrix[(i - 1) * dimension + j] +
                        matrix[i * dimension + j - 1] +
                        matrix[i * dimension + j + 1]
                    );
                }
            }
        }
        #pragma acc parallel loop present(matrix, matrix_new)
        for (int i = 1; i < dimension - 1; i++) {
            for (int j = 1; j < dimension - 1; j++) {
                matrix[i * dimension + j] = matrix_new[i * dimension + j];
            }
        }
        iter_count++;
    }
    nvtxRangePop();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds\n";
    std::cout << "Iterations performed: " << iter_count << "\n";
    std::cout << "Achieved error: " << error << "\n";
    #pragma acc exit data delete(matrix[:dimension*dimension], matrix_new[:dimension*dimension], diff[:dimension*dimension])
    cublasDestroy(handle);
    free(matrix);
    free(matrix_new);
    free(diff);
    return 0;
}
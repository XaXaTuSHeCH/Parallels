#include <boost/program_options.hpp>
#include <iostream>
#include <chrono>
#include <cublas_v2.h>
#include <memory>
#include <nvtx3/nvToolsExt.h>

#define INDEX(row, col, width) ((row) * (width) + (col))

void setup_grid(double* grid, double* grid_new, int rows, int cols) {
    std::fill_n(grid, rows * cols, 0.0);
    std::fill_n(grid_new, rows * cols, 0.0);

    const double corner_top_left = 10.0;
    const double corner_top_right = 20.0;
    const double corner_bottom_right = 30.0;
    const double corner_bottom_left = 20.0;

    // Top boundary
    for (int j = 0; j < cols; ++j) {
        double t = static_cast<double>(j) / (cols - 1);
        grid[INDEX(0, j, cols)] = grid_new[INDEX(0, j, cols)] = 
            corner_top_left * (1.0 - t) + corner_top_right * t;
    }

    // Bottom boundary
    for (int j = 0; j < cols; ++j) {
        double t = static_cast<double>(j) / (cols - 1);
        grid[INDEX(rows - 1, j, cols)] = grid_new[INDEX(rows - 1, j, cols)] = 
            corner_bottom_left * (1.0 - t) + corner_bottom_right * t;
    }

    // Left boundary
    for (int i = 0; i < rows; ++i) {
        double t = static_cast<double>(i) / (rows - 1);
        grid[INDEX(i, 0, cols)] = grid_new[INDEX(i, 0, cols)] = 
            corner_top_left * (1.0 - t) + corner_bottom_left * t;
    }

    // Right boundary
    for (int i = 0; i < rows; ++i) {
        double t = static_cast<double>(i) / (rows - 1);
        grid[INDEX(i, cols - 1, cols)] = grid_new[INDEX(i, cols - 1, cols)] = 
            corner_top_right * (1.0 - t) + corner_bottom_right * t;
    }
}

void cleanup(double* grid, double* grid_new) {
    free(grid);
    free(grid_new);
}

namespace opt = boost::program_options;

int main(int argc, char* argv[]) {
    int grid_rows, grid_cols, max_iterations;
    double convergence_threshold;

    opt::options_description options("Command-line options");
    options.add_options()
        ("help,h", "display this help message")
        ("grid-size,s", opt::value<int>(&grid_rows)->default_value(512), "grid size (s x s)")
        ("tolerance,t", opt::value<double>(&convergence_threshold)->default_value(1.0e-6), "convergence tolerance")
        ("max-iterations,m", opt::value<int>(&max_iterations)->default_value(1000000), "maximum iterations");

    opt::variables_map config;
    opt::store(opt::parse_command_line(argc, argv, options), config);
    opt::notify(config);

    if (config.count("help")) {
        std::cout << options << std::endl;
        return 1;
    }

    grid_cols = grid_rows;

    double* grid = static_cast<double*>(malloc(sizeof(double) * grid_rows * grid_cols));
    double* grid_new = static_cast<double*>(malloc(sizeof(double) * grid_rows * grid_cols));

    nvtxRangePush("setup");
    setup_grid(grid, grid_new, grid_rows, grid_cols);
    nvtxRangePop();

    std::cout << "Running Jacobi solver on " << grid_rows << " x " << grid_cols << " grid\n";

    auto start_time = std::chrono::steady_clock::now();
    int iteration_count = 0;
    double max_diff = 1.0;

    auto cublas_cleanup = [](cublasHandle_t* handle) {
        if (handle && *handle) {
            cublasDestroy(*handle);
            delete handle;
        }
    };

    std::unique_ptr<cublasHandle_t, decltype(cublas_cleanup)> cublas_handle(
        new cublasHandle_t, cublas_cleanup);
    cublasCreate(cublas_handle.get());

    double* device_diff_array;
    cudaMalloc(&device_diff_array, sizeof(double) * grid_rows * grid_cols);

    int max_diff_index;
    nvtxRangePush("solver_loop");
    #pragma acc data copy(grid[0:grid_rows*grid_cols], grid_new[0:grid_rows*grid_cols]) \
                     create(device_diff_array[0:grid_rows*grid_cols])
    {
        while (max_diff > convergence_threshold && iteration_count < max_iterations) {
            #pragma acc parallel loop collapse(2) present(grid, grid_new)
            for (int i = 1; i < grid_rows - 1; ++i) {
                for (int j = 1; j < grid_cols - 1; ++j) {
                    grid_new[INDEX(i, j, grid_cols)] = 0.25 * (
                        grid[INDEX(i, j + 1, grid_cols)] + 
                        grid[INDEX(i, j - 1, grid_cols)] + 
                        grid[INDEX(i + 1, j, grid_cols)] + 
                        grid[INDEX(i - 1, j, grid_cols)]
                    );
                }
            }

            if (iteration_count % 1000 == 0) {
                #pragma acc parallel loop collapse(2) present(grid, grid_new, device_diff_array)
                for (int i = 1; i < grid_rows - 1; ++i) {
                    for (int j = 1; j < grid_cols - 1; ++j) {
                        device_diff_array[INDEX(i, j, grid_cols)] = 
                            std::abs(grid_new[INDEX(i, j, grid_cols)] - grid[INDEX(i, j, grid_cols)]);
                    }
                }

                #pragma acc host_data use_device(device_diff_array)
                {
                    cublasStatus_t status = cublasIdamax(
                        *cublas_handle, grid_rows * grid_cols, device_diff_array, 1, &max_diff_index);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "Error in cublasIdamax\n";
                    }

                    if (max_diff_index > 0 && max_diff_index <= grid_rows * grid_cols) {
                        max_diff_index -= 1;
                        cudaMemcpy(&max_diff, &device_diff_array[max_diff_index], 
                                  sizeof(double), cudaMemcpyDeviceToHost);
                    } else {
                        std::cerr << "Invalid index from cublasIdamax: " << max_diff_index + 1 << "\n";
                        max_diff = 0.0;
                    }
                }
            }

            std::swap(grid, grid_new);

            if (iteration_count % 10000 == 0) {
                std::cout << "Iteration " << iteration_count << ", max difference: " << max_diff << "\n";
            }

            ++iteration_count;
        }
    }
    nvtxRangePop();

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

    std::cout << "Total iterations: " << iteration_count << "\n";
    std::cout << "Final max difference: " << max_diff << "\n";
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";

    cudaFree(device_diff_array);
    cleanup(grid, grid_new);

    return 0;
}
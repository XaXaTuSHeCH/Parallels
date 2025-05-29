#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

__global__ void computeGrid(double* grid, double* grid_new, int grid_size) {
    __shared__ double s_grid[34][34];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int s_col = threadIdx.x + 1;
    int s_row = threadIdx.y + 1;

    if (row < grid_size && col < grid_size) {
        s_grid[s_row][s_col] = grid[row * grid_size + col];

        if (threadIdx.x == 0 && col > 0)
            s_grid[s_row][s_col - 1] = grid[row * grid_size + col - 1];
        if (threadIdx.x == blockDim.x - 1 && col < grid_size - 1)
            s_grid[s_row][s_col + 1] = grid[row * grid_size + col + 1];
        if (threadIdx.y == 0 && row > 0)
            s_grid[s_row - 1][s_col] = grid[(row - 1) * grid_size + col];
        if (threadIdx.y == blockDim.y - 1 && row < grid_size - 1)
            s_grid[s_row + 1][s_col] = grid[(row + 1) * grid_size + col];
    }

    __syncthreads();

    if (row > 0 && row < grid_size - 1 && col > 0 && col < grid_size - 1) {
        size_t idx = row * grid_size + col;
        grid_new[idx] = 0.25 * (
            s_grid[s_row + 1][s_col] +
            s_grid[s_row - 1][s_col] +
            s_grid[s_row][s_col - 1] +
            s_grid[s_row][s_col + 1]
        );
    }
}

__global__ void updateGrid(double* grid, double* grid_new, int grid_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < grid_size - 1 && col > 0 && col < grid_size - 1) {
        size_t idx = row * grid_size + col;
        grid[idx] = grid_new[idx];
    }
}

__global__ void computeError(double* grid, double* grid_new, int grid_size, double* errors) {
    typedef cub::BlockReduce<double, 32> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    double local_error = 0.0;

    if (row > 0 && row < grid_size - 1 && col > 0 && col < grid_size - 1) {
        size_t idx = row * grid_size + col;
        grid_new[idx] = 0.25 * (
            grid[(row + 1) * grid_size + col] +
            grid[(row - 1) * grid_size + col] +
            grid[row * grid_size + col - 1] +
            grid[row * grid_size + col + 1]
        );
        local_error = fabs(grid_new[idx] - grid[idx]);
    }

    double block_max = BlockReduce(temp_storage).Reduce(local_error, cub::Max());
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        errors[blockIdx.y * gridDim.x + blockIdx.x] = block_max;
    }
}

int main(int argc, char* argv[]) {
    int grid_size, max_iter;
    double target_accuracy;

    po::options_description opts("Program Options");
    opts.add_options()
        ("help", "display help")
        ("size", po::value<int>(&grid_size)->default_value(128), "grid dimension (N x N)")
        ("accuracy", po::value<double>(&target_accuracy)->default_value(1e-6), "error threshold")
        ("max_iterations", po::value<int>(&max_iter)->default_value(1000000), "iteration limit");

    po::variables_map vars;
    po::store(po::parse_command_line(argc, argv, opts), vars);
    po::notify(vars);
    if (vars.count("help")) {
        std::cout << opts << "\n";
        return 1;
    }

    std::cout << "Starting CUDA simulation with grid size " << grid_size << "x" << grid_size << "\n";

    double* grid = (double*)malloc(grid_size * grid_size * sizeof(double));
    double* grid_new = (double*)malloc(grid_size * grid_size * sizeof(double));

    nvtxRangePushA("initialize");
    for (size_t idx = 0; idx < grid_size * grid_size; ++idx) {
        grid[idx] = 0.0;
        grid_new[idx] = 0.0;
    }

    grid[0] = 10.0;
    grid[grid_size - 1] = 20.0;
    grid[grid_size * (grid_size - 1)] = 30.0;
    grid[grid_size * grid_size - 1] = 20.0;

    double tl = grid[0], tr = grid[grid_size - 1];
    double bl = grid[grid_size * (grid_size - 1)], br = grid[grid_size * grid_size - 1];
    for (int k = 1; k < grid_size - 1; ++k) {
        grid[k] = tl + (tr - tl) * k / (grid_size - 1.0);
        grid[grid_size * (grid_size - 1) + k] = bl + (br - bl) * k / (grid_size - 1.0);
        grid[grid_size * k] = tl + (bl - tl) * k / (grid_size - 1.0);
        grid[grid_size * k + grid_size - 1] = tr + (br - tr) * k / (grid_size - 1.0);
    }
    nvtxRangePop();

    double *d_grid, *d_grid_new, *d_errors;
    cudaMalloc(&d_grid, grid_size * grid_size * sizeof(double));
    cudaMalloc(&d_grid_new, grid_size * grid_size * sizeof(double));

    dim3 blockDim(32, 32);
    dim3 gridDim((grid_size + 31) / 32, (grid_size + 31) / 32);
    int error_array_size = gridDim.x * gridDim.y;

    cudaMalloc(&d_errors, error_array_size * sizeof(double));
    cudaMemcpy(d_grid, grid, grid_size * grid_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_new, grid_new, grid_size * grid_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t computeNode, updateNode;
    cudaKernelNodeParams computeParams = {}, updateParams = {};

    computeParams.func = (void*)computeGrid;
    computeParams.gridDim = gridDim;
    computeParams.blockDim = blockDim;
    computeParams.sharedMemBytes = 0;
    void* computeArgs[] = {&d_grid, &d_grid_new, &grid_size};
    computeParams.kernelParams = computeArgs;

    updateParams.func = (void*)updateGrid;
    updateParams.gridDim = gridDim;
    updateParams.blockDim = blockDim;
    updateParams.sharedMemBytes = 0;
    void* updateArgs[] = {&d_grid, &d_grid_new, &grid_size};
    updateParams.kernelParams = updateArgs;

    cudaGraphAddKernelNode(&computeNode, graph, nullptr, 0, &computeParams);
    cudaGraphAddKernelNode(&updateNode, graph, &computeNode, 1, &updateParams);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    double max_error = target_accuracy + 1.0;
    int current_iter = 0;
    auto start_time = std::chrono::steady_clock::now();

    nvtxRangePushA("computation");
    while (max_error > target_accuracy && current_iter < max_iter) {
        if (current_iter % 1000 == 0) {
            cudaMemsetAsync(d_errors, 0, error_array_size * sizeof(double), stream);
            computeError<<<gridDim, blockDim, 0, stream>>>(d_grid, d_grid_new, grid_size, d_errors);

            double* h_errors = (double*)malloc(error_array_size * sizeof(double));
            cudaMemcpy(h_errors, d_errors, error_array_size * sizeof(double), cudaMemcpyDeviceToHost);
            max_error = 0.0;
            for (int i = 0; i < error_array_size; ++i) {
                max_error = std::fmax(max_error, h_errors[i]);
            }
            free(h_errors);
        } else {
            cudaGraphLaunch(graphExec, stream);
        }
        current_iter++;
    }

    cudaStreamSynchronize(stream);
    nvtxRangePop();
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Execution time: " << duration.count() << " seconds\n";
    std::cout << "Total iterations: " << current_iter << "\n";
    std::cout << "Final error: " << max_error << "\n";

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_grid);
    cudaFree(d_grid_new);
    cudaFree(d_errors);
    free(grid);
    free(grid_new);

    return 0;
}

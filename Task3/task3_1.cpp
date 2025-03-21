#include <iostream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    srand(42);
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int sizes[] = {20000, 40000};
    for (int s : sizes) {
        cout << s << "x" << s << " matrix:" << endl;
        double** matrix = new double*[s];
        for (int i = 0; i < s; ++i) matrix[i] = new double[s];
        double* vec = new double[s];
        double* result = new double[s];
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) matrix[i][j] = (double)rand() / RAND_MAX;
            vec[i] = (double)rand() / RAND_MAX;
            result[i] = 0.0;
        }
        for (int th : threads) {
            double start = omp_get_wtime();
            vector<thread> mythreads;
            int chunk = s / th;
            int mod = s % th;
            for (int i = 0; i < th; ++i) {
                int start_idx = i * chunk;
                int end_idx = start_idx + chunk;
                if (i == th - 1) end_idx += mod;
                mythreads.emplace_back([&, start_idx, end_idx]() {
                    for (int i = start_idx; i < end_idx; ++i) {
                        result[i] = 0.0;
                        for (int j = 0; j < s; ++j) result[i] += matrix[i][j] * vec[j];
                    }
                });
            }
            for (auto& t : mythreads) t.join();
            double end = omp_get_wtime();
            cout << th << "threads, " << (end - start) << " sec." << endl;
        }
        for (int i = 0; i < s; ++i) delete[] matrix[i];
        delete[] matrix;
        delete[] vec;
        delete[] result;
    }
    return 0;
}

#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

int main(){
    srand(42);
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int sizes[] = {20000, 40000};
    for (int s : sizes){
        int rows = s;
        int cols = s;
        cout << rows << ' ' << cols << endl;
        double** matrix = new double*[rows];
        for (int i = 0; i < rows; i++){
            matrix[i] = new double[cols];
        }
        double* vector = new double[cols];
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                matrix[i][j] = (double)rand() / RAND_MAX;
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < cols; i++){
            vector[i] = (double)rand() / RAND_MAX;
        }
        for (int t : threads){
            omp_set_num_threads(t);
            double* result = new double[rows]();
            double start = omp_get_wtime();
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; i++){
                for (int j = 0; j < cols; j++){
                    result[i] += matrix[i][j] * vector[j];
                }
            }
            double end = omp_get_wtime();
            cout << t << " threads, " << (end - start) << " sec." << endl;
            delete[] result;
        }
        for (int i = 0; i < rows; i++){
            delete[] matrix[i];
        }
        delete[] matrix;
        delete[] vector;
    }
    return 0;
}

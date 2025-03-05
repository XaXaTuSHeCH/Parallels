#include <iostream>
#include <omp.h>

using namespace std;

const int N = 5000;
const double eps = 1e-5;
const double tau = 0.01;

void separate(double** A, const double* b, double* x) {
    auto* Ax = new double[N];
    auto* x_new = new double[N];
    double err = 1.0;
    while (err >= eps) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            Ax[i] = 0.0;
            for (int j = 0; j < N; j++) {
                Ax[i] += A[i][j] * x[j];
            }
        }
        double x_new_m = 0.0;
        #pragma omp parallel for reduction(+:x_new_m)
        for (int i = 0; i < N; i++) {
            x_new[i] = Ax[i] - b[i];
            x[i] -= x_new[i] * tau;
            x_new_m += x_new[i] * x_new[i];
        }
        err = x_new_m / N;
    }
    delete[] Ax;
    delete[] x_new;
}

void general(double** A, const double* b, double* x) {
    auto* Ax = new double[N];
    auto* x_new = new double[N];
    double err = 1.0;
    while (err >= eps) {
        double x_new_m = 0.0;
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < N; i++) {
                Ax[i] = 0.0;
                for (int j = 0; j < N; j++) {
                    Ax[i] += A[i][j] * x[j];
                }
            }
            #pragma omp for reduction(+:x_new_m)
            for (int i = 0; i < N; i++) {
                x_new[i] = Ax[i] - b[i];
                x[i] -= x_new[i] * tau;
                x_new_m += x_new[i] * x_new[i];
            }
        }
        err = x_new_m / N;
    }
    delete[] Ax;
    delete[] x_new;
}

void optimize(double** A, const double* b, double* x, const string& schedule_type, const int iter) {
    auto* Ax = new double[N];
    auto* x_new = new double[N];
    double err = 1.0;
    while (err >= eps) {
        double x_new_m = 0.0;
        #pragma omp parallel
        {
            if (schedule_type == "static") {
                #pragma omp for schedule(static, iter)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
                #pragma omp for reduction(+:x_new_m) schedule(static, iter)
                for (int i = 0; i < N; i++) {
                    x_new[i] = Ax[i] - b[i];
                    x[i] -= x_new[i] * tau;
                    x_new_m += x_new[i] * x_new[i];
                }
            }
            else if (schedule_type == "dynamic") {
                #pragma omp for schedule(dynamic, iter)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
                #pragma omp for reduction(+:x_new_m) schedule(dynamic, iter)
                for (int i = 0; i < N; i++) {
                    x_new[i] = Ax[i] - b[i];
                    x[i] -= x_new[i] * tau;
                    x_new_m += x_new[i] * x_new[i];
                }
            }
            else if (schedule_type == "guided") {
                #pragma omp for schedule(guided, iter)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
                #pragma omp for reduction(+:x_new_m) schedule(guided, iter)
                for (int i = 0; i < N; i++) {
                    x_new[i] = Ax[i] - b[i];
                    x[i] -= x_new[i] * tau;
                    x_new_m += x_new[i] * x_new[i];
                }
            }
            else if (schedule_type == "auto") {
                #pragma omp for schedule(auto)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
                #pragma omp for reduction(+:x_new_m) schedule(auto)
                for (int i = 0; i < N; i++) {
                    x_new[i] = Ax[i] - b[i];
                    x[i] -= x_new[i] * tau;
                    x_new_m += x_new[i] * x_new[i];
                }
            }
        }
        err = x_new_m / N;
    }
    delete[] Ax;
    delete[] x_new;
}


int main() {
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int iters[] = {1,10,100};
    string schedules[] = {"static", "dynamic", "guided"};
    auto** A = new double*[N];
    for (int i = 0; i < N; i++) {
        A[i] = new double[N];
    }
    auto* b = new double[N];
    auto* x = new double[N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = N + 1;
        x[i] = 0.0;
    }
    cout << "Separate testing..." << endl;
    for (const auto t : threads) {
        auto* x_copy = new double[N];
        for (int i = 0; i < N; i++) x_copy[i] = x[i];
        omp_set_num_threads(t);
        const double start = omp_get_wtime();
        separate(A, b, x_copy);
        const double end = omp_get_wtime();
        cout << (end - start) << " sec." << endl;
        delete[] x_copy;
    }
    cout << "General testing..." << endl;
    for (const auto t : threads) {
        auto* x_copy = new double[N];
        for (int i = 0; i < N; i++) x_copy[i] = x[i];
        omp_set_num_threads(t);
        const double start = omp_get_wtime();
        general(A, b, x_copy);
        const double end = omp_get_wtime();
        cout << (end - start) << " sec." << endl;
        delete[] x_copy;
    }
    cout << "Schedule testing on 8 threads..." << endl;
    omp_set_num_threads(8);
    for (const auto& type : schedules) {
        for (const auto iter : iters) {
            auto* x_copy = new double[N];
            for (int i = 0; i < N; i++) x_copy[i] = x[i];
            const double start = omp_get_wtime();
            optimize(A, b, x_copy, type, iter);
            const double end = omp_get_wtime();
            cout << type << ", " << iter << " iters, " << (end - start) << " sec." << endl;
            delete[] x_copy;
        }
    }
    auto* x_copy = new double[N];
    for (int i = 0; i < N; i++) x_copy[i] = x[i];
    const double start = omp_get_wtime();
    optimize(A, b, x_copy, "auto", 1);
    const double end = omp_get_wtime();
    cout << "auto, " << (end - start) << " sec." << endl;
    delete[] x_copy;
    for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete[] A;
    delete[] b;
    delete[] x;
    return 0;
}

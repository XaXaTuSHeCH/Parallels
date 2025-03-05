#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

double integrate_omp(double a, double b, int nsteps, int num_threads) {
    double step = (b-a)/nsteps;
    double sum = 0.0;
    #pragma omp parallel num_threads(num_threads)
    {
        double local = 0.0;
        #pragma omp for
        for (int i = 0; i < nsteps; i++) {
            double x = a+(i+0.5)*step;
            local += sin(x);
        }
        #pragma omp atomic
            sum += local;
    }
    return sum * step;
}

int main() {
    double a = 0.0, b = M_PI;
    int nsteps = 40000000;
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    for (int t : threads) {
        double start = omp_get_wtime();
        double result = integrate_omp(a, b, nsteps, t);
        double end = omp_get_wtime();
        cout << t << " threads, " << (end - start) << " sec, " << result << endl;
    }
    return 0;
}

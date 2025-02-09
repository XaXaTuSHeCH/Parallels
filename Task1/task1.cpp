#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef TYPE
    #define TYPE double
#endif

using namespace std;

constexpr size_t N = 10000000;

int main() {
    TYPE* values = new TYPE[N];
    for (size_t i = 0; i < N; ++i) {
        values[i] = sin((static_cast<double>(i) / N) * 2 * M_PI);
    }
    TYPE sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += values[i];
    }
    delete[] values;
    cout << sum << endl;
    return 0;
}
#include <iostream>
#include <cmath>

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

constexpr size_t N = 10000000;

int main() {
    DATA_TYPE* values = new DATA_TYPE[N];
    for (size_t i = 0; i < N; ++i) {
        values[i] = std::sin((static_cast<double>(i) / N) * 2 * M_PI);
    }
    DATA_TYPE sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += values[i];
    }
    delete[] values;
    std::cout << sum << std::endl;
    return 0;
}
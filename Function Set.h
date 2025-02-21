#include <cmath>
#include <vector>
#include <iostream>

// Sphere Function
// f(x) = sum(x[i]^2)
// Global Minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
// Smooth: Yes
inline double SphereFunction(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

// Rosenbrock Function
// f(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
// Global Minimum: f(x*) = 0 at x* = (1, 1, ..., 1)
// Smooth: Yes
inline double RosenbrockFunction(const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double term1 = 100.0 * std::pow(x[i + 1] - x[i] * x[i], 2);
        double term2 = std::pow(1.0 - x[i], 2);
        sum += term1 + term2;
    }
    return sum;
}

// Rastrigin Function
// f(x) = sum(x[i]^2 - 10 * cos(2 * pi * x[i]) + 10)
// Global Minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
// Smooth: Yes
inline double RastriginFunction(const std::vector<double>& x) {
    const double A = 10.0;
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi - A * std::cos(2.0 * M_PI * xi) + A;
    }
    return sum;
}

// Schwefel Function
// f(x) = 418.9829 * n - sum(x[i] * sin(sqrt(abs(x[i]))))
// Global Minimum: f(x*) = 0 at x* = (420.9687, 420.9687, ..., 420.9687)
// Smooth: Yes
inline double SchwefelFunction(const std::vector<double>& x) {
    const double A = 418.9829;
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * std::sin(std::sqrt(std::abs(xi)));
    }
    return A * x.size() - sum;
}

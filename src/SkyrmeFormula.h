#pragma once
#include <functional>
#include <vector>

struct SkyrmeInput {
    double alpha0 = 0.0;
    double beta0  = 0.0;
    double gamma  = 0.0;
    const std::vector<double>& pair_density;
    SkyrmeInput(double a, double b, double g, const std::vector<double>& e)
        : alpha0(a), beta0(b), gamma(g), pair_density(e) {}
};

using SkyrmeFormula = std::function<double(const SkyrmeInput&)>;

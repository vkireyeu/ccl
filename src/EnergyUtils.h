#pragma once

#include "libccl.h"
class CClusterizer;
class CCParticle;

#include "SkyrmeFormula.h"
#include <array>


// Debug stuff for tests
struct DebugInfo {
    std::vector<double> b_values;
    std::vector<double> r2_values;
    std::vector<double> rho_values;
    std::vector<std::array<double, 6>> boosted_states;
};

double computeClusterKineticEnergy(const std::vector<CCParticle*>& group);
double computeClusterPotentialEnergy(const std::vector<CCParticle*>& group, CClusterizer* parent, DebugInfo* debug);
double computeBindingEnergy(const std::vector<CCParticle*>& group, CClusterizer* parent);

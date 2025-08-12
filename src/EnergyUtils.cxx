#include "EnergyUtils.h"

// ---------------------------------------------------------------------
// Calculate kinetic energy of the cluster (group of baryons):
// E_kin is full here, not divided by cluster mass number A
// ---------------------------------------------------------------------
double computeClusterKineticEnergy(const std::vector<CCParticle*>& group) {
    double totalE = 0, totalPx = 0, totalPy = 0, totalPz = 0;

    // Loop over all particles of this cluster
    for (const auto* p : group) {
        totalE  += p -> getE();     // Get the current particle energy
        totalPx += p -> getPx();    // Px
        totalPy += p -> getPy();    // Py
        totalPz += p -> getPz();    // and Pz
    }

    // Cluster velocities
    double bx = totalPx / totalE;
    double by = totalPy / totalE;
    double bz = totalPz / totalE;

    double E_kin = 0;
    // Kinetic energy of the cluster = sum of the kinetic energies of 
    //   its constituent particles in the COM of the cluster
    for (const auto* p : group) {
        CCParticle tmp = *p;
        tmp.LorentzBoost(bx, by, bz);
        E_kin += tmp.getE() - tmp.getMass();
    }
    return E_kin;
}




// ---------------------------------------------------------------------
// Calculate Coulomb component for the potential energy:
//   only for protons!
// ---------------------------------------------------------------------
double ComputeCoulombPart(const std::vector<CCParticle>& group) {
    const double ec2   = CClusterizer::getEc2();
    const double al    = CClusterizer::getAl();

    double V_coul = 0.0;
    for (size_t i = 0; i + 1 < group.size(); ++i) {
        for (size_t j = i + 1; j < group.size(); ++j) {
            double dx = group[i].getX() - group[j].getX();
            double dy = group[i].getY() - group[j].getY();
            double dz = group[i].getZ() - group[j].getZ();
            double r = std::sqrt(dx*dx + dy*dy + dz*dz + 1e-6);
            double x = std::sqrt(1.0 / al) * r;
            V_coul += ec2 / r * (1.0 - std::erf(x));
        }
    }
    return V_coul;
}




// ---------------------------------------------------------------------
// Asymmetry energy component:
// ---------------------------------------------------------------------
double ComputeEasyPart(const std::vector<CCParticle>& group) {
    const double al  = CClusterizer::getAl();
    const double sal = CClusterizer::getSal();
    const size_t N = group.size();

    // Symmetry-energy parameters (built-in for a moment)
    const double E_0 = 0.0233;    // Asymmetry energy constant 0.0233 GeV = 23.3 MeV
    const double rho_0 = 0.16;    // Saturation density scale (relative units)
    const double gamma = 0.67;    // Density exponent (soft-to-stiff EOS)
    const double eps = 1e-15;     // Small value to avoid division by zero

    // Build per-particle local "density"
    std::vector<double> rho_loc(N, 0.0);
    for (size_t i = 0; i + 1 < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double dx = group[i].getX() - group[j].getX();
            double dy = group[i].getY() - group[j].getY();
            double dz = group[i].getZ() - group[j].getZ();
            double r2 = dx*dx + dy*dy + dz*dz;
            double overlap = std::exp(-r2 / al) * (sal / al); // Scale overlap by sal/al
            rho_loc[i] += overlap;
            rho_loc[j] += overlap;
        }
    }

    // Standard self-term
    for (size_t i = 0; i < N; ++i) {
        rho_loc[i] += 1.0;
    }

    // Build global density sums
    double rho_B = 0.0; // Total baryon density
    double rho_p = 0.0; // Proton density
    double rho_n = 0.0; // Neutron density

    for (size_t i = 0; i < N; ++i) {
        rho_B += rho_loc[i];
        if      (group[i].getType() == ParticleType::Proton)  rho_p += rho_loc[i];
        else if (group[i].getType() == ParticleType::Neutron) rho_n += rho_loc[i];
    }
    if (rho_B <= eps) return 0.0;

    // \delta = (\rho_{n} - \rho_{p}) / \rho_{B}
    // Usually \rho_{B} = \rho_{n} + \rho_{p}, but hyperons also can enter \rho_{B}
    double delta = (rho_n - rho_p) / rho_B;

    // < \rho_{B} > / \rho_{0}  =  (\rho_{B} / N) / \rho_{0}
    double rho_ratio = std::max(rho_B / (rho_0 * N), eps);

    // E_asy = E0 * delta^2 * (<\rho_{B}>/\rho_{0})^gamma  * N
    return E_0 * delta * delta * std::pow(rho_ratio, gamma) * N;
}




// ---------------------------------------------------------------------
// Skyrme potential part -- usually must not be turned off (but can be)
// ---------------------------------------------------------------------
double ComputeSkyrmePart(const std::vector<CCParticle>& group, CClusterizer* parent, DebugInfo* debug = nullptr) {
    const double al     = CClusterizer::getAl();
    const double gamma  = CClusterizer::getGamma();
    const double alpha0 = CClusterizer::getAlpha0();
    const double beta0  = CClusterizer::getBeta0();
    const double hypot  = CClusterizer::getHyPot();

    const size_t N = group.size();

    std::vector<double> pd(N, 0.0);
    // For all boosted baryons (protons, neutrons, lambdas and sigmas) pairs
    for (size_t i = 0; i + 1 < N; ++i) {
        // Weight = CClusterizer::getHyPot() for hyperons, 1 for other particles
        double w_i = (group[i].getType() == ParticleType::LambdaSigma) ? hypot : 1.0;
        for (size_t j = i + 1; j < N; ++j) {
            // Weight = CClusterizer::getHyPot() for hyperons, 1 for other particles
            double w_j = (group[j].getType() == ParticleType::LambdaSigma) ? hypot : 1.0;
            double weight = w_i * w_j; // Common pair weight

            // Calculate distance between group in the cluster COM frame
            double dx = group[i].getX() - group[j].getX();
            double dy = group[i].getY() - group[j].getY();
            double dz = group[i].getZ() - group[j].getZ();
            double r2 = dx*dx + dy*dy + dz*dz;
            // rho = density
            double rho     = std::exp(-r2 / al);
            double contrib = rho * weight;
            pd[i] += contrib;
            pd[j] += contrib;

            if (debug) {
                debug -> r2_values.push_back(r2);
                debug -> rho_values.push_back(rho);
            }
        }
    }

    SkyrmeInput input(alpha0, beta0, gamma, pd);
    return parent->getSkyrmeFormula()(input);
}




// ---------------------------------------------------------------------
// Calculate potential energy of the cluster (group of baryons):
// E_pot is full here, not divided by cluster mass number A
// ---------------------------------------------------------------------
double computeClusterPotentialEnergy(const std::vector<CCParticle*>& group, CClusterizer* parent, DebugInfo* debug = nullptr) {
    // Current cluster energy and momentum will be written to these ariables
    double totalE = 0.0, totalPx = 0.0, totalPy = 0.0, totalPz = 0.0;

    // Loop over all particles of this cluster
    for (const auto* p : group) {
        totalE  += p -> getE();     // Get the current particle energy
        totalPx += p -> getPx();    // Px
        totalPy += p -> getPy();    // Py
        totalPz += p -> getPz();    // and Pz
    }

    // Cluster velocities
    double bx = totalPx / totalE;
    double by = totalPy / totalE;
    double bz = totalPz / totalE;

    if (debug) {
        debug -> b_values.push_back(bx);
        debug -> b_values.push_back(by);
        debug -> b_values.push_back(bz);
    }

    // Boost particles to the cluster COM frame
    std::vector<CCParticle> baryons, protons;
    // Loop over all particles of this cluster
    for (const auto* orig : group) {
        CCParticle p = *orig;         // Take particle
        p.LorentzBoost(bx, by, bz);   // and boost it to the cluster velocity
        baryons.push_back(p);         // Copy boosted particle to teh separate array
        if (p.getType() == ParticleType::Proton)       // If particle is proton:
            protons.push_back(p);     //   copy it to additional protons-only array

        if (debug) {
            debug->boosted_states.push_back({
                p.getPx(), p.getPy(), p.getPz(),
                p.getX(),  p.getY(),  p.getZ()
            });
        }
    }

    // Coulomb part
    double V_coul = 0.0;
    if(CClusterizer::getComputeCoul()){
        V_coul = ComputeCoulombPart(protons);
    }

    // Skyrme part
    double V_skyrme = 0.0;
    if(CClusterizer::getComputeSkyrme()){
        V_skyrme = ComputeSkyrmePart(baryons, parent, debug);
    }

    double E_asy = 0.0;
    if(CClusterizer::getComputeEasy()){
        E_asy = ComputeEasyPart(baryons);
    }

    return V_skyrme + V_coul + E_asy; // This is E_pot -- potential energy
}




// ---------------------------------------------------------------------
// Calculate binding energy  E_bind/A of the cluster (group of baryons):
// E_bind here IS divided by cluster mass number A!
// ---------------------------------------------------------------------
double computeBindingEnergy(const std::vector<CCParticle*>& group, CClusterizer* parent) {
    if (parent -> getComputeEbind() && !parent -> hasSkyrmeFormula()) {
        throw std::runtime_error(
            "CClusterizer: computeEbind=true but no Skyrme formula set. "
            "Call setSkyrmeFormula() first."
        );
    }

    const size_t A = group.size();
    if (A < 2) return 0.0;

    // E_kin/A
    double Ekin = computeClusterKineticEnergy(group)   / (double) A;
    // E_pot/A
    double Epot = computeClusterPotentialEnergy(group, parent) / (double) A;
    // E_bind = E_kin/A + E_pot/A
    return (Ekin + Epot);
}

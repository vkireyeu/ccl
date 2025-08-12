#include "libccl.h"

// Default values (MUST be overriden by setters)
double CClusterizer::Ec2   =  0.00144;    // GeV * fm
double CClusterizer::Al    =  3.;         // fm^2
double CClusterizer::Sal   =  5.;         // fm^2
double CClusterizer::Alpha = -0.15;       // GeV 
double CClusterizer::Beta  =  0.05;       // GeV 
double CClusterizer::Gamma =  1.5;
double CClusterizer::HyPot =  1.;

// Skyrme part for computeClusterPotentialEnergy
bool    CClusterizer::ComputeSkyrme = true;

// Coulomb part for computeClusterPotentialEnergy
bool    CClusterizer::ComputeCoulomb = false;

// Asymmetry energy part for computeClusterPotentialEnergy
bool    CClusterizer::ComputeEasy = false;


CClusterizer::CClusterizer(){
}

CClusterizer::~CClusterizer(){
    delete skyrmeFormula;
}


// ---------------------------------------------------------------------
// Merge group of baryons to single cluster: works for MST and Coalescence
// ---------------------------------------------------------------------
CCParticle CClusterizer::mergeToCluster(const std::vector<CCParticle*>& group) const{
    if (group.empty()) throw std::runtime_error("mergeToCluster: empty group");

    CCParticle cluster;
    double total_mass = 0;
    double sum_x  = 0, sum_y  = 0, sum_z  = 0;
    double sum_px = 0, sum_py = 0, sum_pz = 0;
    double total_e = 0.;
    short  total_charge = 0;
    double latest_time  = 0.;

    short  A = 0;
    short  N = 0;
    short  L = 0;
    short  Z = 0;

    for (const auto& p : group) {
        const double m = p -> getMass();
        total_mass   += m;
        total_e      += p -> getE();
        sum_px       += p -> getPx();
        sum_py       += p -> getPy();
        sum_pz       += p -> getPz();
        sum_x        += m * p -> getX();
        sum_y        += m * p -> getY();
        sum_z        += m * p -> getZ();
        total_charge += p -> getCharge();
        if (p -> getTime() > latest_time) latest_time = p -> getTime();
        if      (p -> getType() == ParticleType::Neutron)     ++N;
        else if (p -> getType() == ParticleType::Proton)      ++Z;
        else if (p -> getType() == ParticleType::LambdaSigma) ++L;
    }

    A = N + Z + L;
    cluster.setPID((long)(10*10E7 + L*10E6 + Z*10E3 + A*10));
    cluster.setCharge(total_charge);
    cluster.setMomentum(sum_px, sum_py, sum_pz, total_mass);

    // Cluster velocity
    double bx = sum_px / total_e;
    double by = sum_py / total_e;
    double bz = sum_pz / total_e;

    // Boost constituent particles to the cluster COM
    std::vector<CCParticle> boosted;
    double latest_COM_time  = 0.;
    boosted.reserve(group.size());
    for (const auto* p : group) {
        CCParticle tmp = *p;
        tmp.LorentzBoost(bx, by, bz);
        boosted.push_back(tmp);
        if (tmp.getTime() > latest_COM_time) latest_COM_time = tmp.getTime();
    }

    // Synchronize the time for particles -- extrapolate coordinates to the last_time
    // and calculate the cluster position using these new coordinates
    double cx = 0, cy = 0, cz = 0;
    for (auto& p : boosted) {
        double dt = latest_COM_time - p.getTime();
        if (dt < 1E-6) dt = 0;
        double E = p.getE();
        double m = p.getMass();
        cx += m * (p.getX() + dt * p.getPx() / E);
        cy += m * (p.getY() + dt * p.getPy() / E);
        cz += m * (p.getZ() + dt * p.getPz() / E);
    }
    cx /= total_mass;
    cy /= total_mass;
    cz /= total_mass;

    // Boost cluster coordinates back
    CCParticle dummy;
    dummy.setPosition(cx, cy, cz, latest_COM_time);
    dummy.LorentzBoost(-bx, -by, -bz);

    // Use these coordinates
    cluster.setPosition(dummy.getX(), dummy.getY(), dummy.getZ(), dummy.getTime());


    // Calculate and set binding energy
    if(ComputeEbind){
        const CClusterizer* self = this;
        cluster.setBindingEnergy(computeBindingEnergy(group, const_cast<CClusterizer*>(self)));
    }

    // Fill Childs with original baryons
    for (auto* p : group) {
        p -> setClusterID(group.front() -> getClusterID());
        cluster.addChild(p);
    }

    cluster.setClusterID(group.front() -> getClusterID());
    return cluster;
}

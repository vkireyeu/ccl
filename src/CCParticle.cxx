#include "CCParticle.h"

CCParticle::CCParticle(){

}

CCParticle::~CCParticle(){
}

ParticleType CCParticle::classifyPID(long pid) {
    static const std::unordered_map<long, ParticleType> pidToType = {
        {2112,       ParticleType::Neutron},
        {2212,       ParticleType::Proton},
        {3122,       ParticleType::LambdaSigma},
        {3212,       ParticleType::LambdaSigma},
        {1000010020, ParticleType::Deuteron},
        {1000010030, ParticleType::Triton},
        {1000020030, ParticleType::Helium3},
        {1000020040, ParticleType::Alpha},
        {1010010030, ParticleType::H3L},
        {1010010040, ParticleType::H4L},
    };

    auto it = pidToType.find(pid);
    if (it != pidToType.end()) {
        return it->second;
    }

    if (pid >= 1000000000) {
        return ParticleType::MSTNucleus;
    }

    return ParticleType::Unknown;
}



void CCParticle::LorentzBoost(double bx, double by, double bz) {
    // Calculate boost parameters
    const double bs    = bx*bx + by*by + bz*bz;
    const double bs2   = 1. - bs;
    const double gamma = 1.0 / sqrt(bs2);

    // Get current four-momentum and position
    const double e  = getE();
    const double px = getPx();
    const double py = getPy();
    const double pz = getPz();
    const double x  = getX();
    const double y  = getY();
    const double z  = getZ();
    const double t  = getTime();

    // Boost momentum and energy
    double bpfactor  = (gamma - 1.0) / bs*(bx*px + by*py + bz*pz);
    const double newPx = px + bpfactor*bx - gamma*bx*e;
    const double newPy = py + bpfactor*by - gamma*by*e;
    const double newPz = pz + bpfactor*bz - gamma*bz*e;

    // Boost position and time
    double bdotx = bx*x + by*y + bz*z;
    double btfactor  = (gamma - 1.0) / bs*bdotx;
    const double newX = x + btfactor*bx - gamma*bx*t;
    const double newY = y + btfactor*by - gamma*by*t;
    const double newZ = z + btfactor*bz - gamma*bz*t;
    const double newT = gamma * (t - bdotx);


    // Update all components
    setPx(newPx);
    setPy(newPy);
    setPz(newPz);
    setX(newX);
    setY(newY);
    setZ(newZ);
    setTime(newT);
}

#include "libccl.h"

// ---------------------------------------------------------------------
// Distance and momentum check between 2 particles for coalescence
// ---------------------------------------------------------------------
bool CClusterizer::shouldCoalesce(CCParticle& p1, CCParticle& p2) const {
    // Create temporary copies for Lorentz transformations
    CCParticle tmp1 = p1;
    CCParticle tmp2 = p2;
    
    // Calculate pair quantities
    const double Tpair = p1.getE() + p2.getE();
    const double bx = (p1.getPx() + p2.getPx()) / Tpair;
    const double by = (p1.getPy() + p2.getPy()) / Tpair;
    const double bz = (p1.getPz() + p2.getPz()) / Tpair;

    // Boost particles to pair COM frame
    tmp1.LorentzBoost(bx, by, bz);
    tmp2.LorentzBoost(bx, by, bz);

    // Time synchronization
    double dt = fabs(tmp1.getTime() - tmp2.getTime());
    if(tmp1.getTime() < tmp2.getTime()) {
        const double factor = dt / tmp1.getE();
        tmp1.setX(tmp1.getX() + factor * tmp1.getPx());
        tmp1.setY(tmp1.getY() + factor * tmp1.getPy());
        tmp1.setZ(tmp1.getZ() + factor * tmp1.getPz());
        tmp1.setTime(tmp2.getTime());
    }
    else if(tmp2.getTime() < tmp1.getTime()) {
        const double factor = dt / tmp2.getE();
        tmp2.setX(tmp2.getX() + factor * tmp2.getPx());
        tmp2.setY(tmp2.getY() + factor * tmp2.getPy());
        tmp2.setZ(tmp2.getZ() + factor * tmp2.getPz());
        tmp2.setTime(tmp1.getTime());
    }

    // Calculate distance in synchronized COM frame
    const double dx  = tmp1.getX() - tmp2.getX();
    const double dy  = tmp1.getY() - tmp2.getY();
    const double dz  = tmp1.getZ() - tmp2.getZ();
    const double dr2 = dx*dx + dy*dy + dz*dz;

    // Radius check
    if(dr2 > Radius*Radius) return false;

    // Optional momentum check
    if(CheckMomentum) {
        const double dpx = tmp1.getPx() - tmp2.getPx();
        const double dpy = tmp1.getPy() - tmp2.getPy();
        const double dpz = tmp1.getPz() - tmp2.getPz();
        const double dp2 = dpx*dpx + dpy*dpy + dpz*dpz;
        if (dp2 > Momentum*Momentum) {
            return false;
        }
    }
    return true;
}



// ---------------------------------------------------------------------
// "Box"-type coalescence for light nuclei
// ---------------------------------------------------------------------
void CClusterizer::makeCoalescence(std::vector<CCParticle>& particles,
                                   ParticleType clusterType,
                                   double radius,
                                   double maxMomentum, bool checkMomentum,
                                   float probability) {
    if (particles.empty()) return;

    this -> setRadius(radius);
    this -> setMomentumWithCheck(maxMomentum, checkMomentum);

    const int BIND_MODE = (cls_UseEbind ? 1 : 0) + (cls_UseIsoEbind ? 2 : 0);
    if (BIND_MODE > 2) {
        throw std::logic_error("Check cls_UseEbind and cls_UseIsoEbind");
    }
    
    short frag_number = -1;
    std::vector<CCParticle*> protons, neutrons, hyperons;
    std::vector<CCParticle>  new_particles;
    std::vector<CCParticle>  clusters;

    for (auto& p : particles) {
        if (static_cast<int>(p.getType()) > 2) new_particles.push_back(p);
        if      (p.getType() == ParticleType::Neutron)     neutrons.push_back(&p);
        else if (p.getType() == ParticleType::Proton)      protons.push_back(&p);
        else if (p.getType() == ParticleType::LambdaSigma) hyperons.push_back(&p);
    }

    // Helper routine which handles now final coalescence nuclei selection criteria
    auto acceptNuclei = [&](const std::vector<CCParticle*>& group) -> bool {
        if (group.empty()) return false;

        // Spin-isospin factor only
        if (BIND_MODE == 0){
            return U01()(get_rng()) <= probability;
        }

        // Binding energy only. In this case procedure can not be called 'coalescence'
        else if (BIND_MODE == 1){
            return computeBindingEnergy(group, this) <= 0.0;
        }

        // Both spin-isospin factor AND binding energy
        else if (BIND_MODE == 2){
            return (U01()(get_rng()) <= probability 
                 && computeBindingEnergy(group, this) <= 0.0);
        }

        return false;
    };

    switch (clusterType) {
        // Deuteron (1p + 1n)
        case ParticleType::Deuteron:
            for (auto* p : protons) {
                if (p -> getClusterID() != -1) continue;
                for (auto* n : neutrons) {
                    if (n -> getClusterID() != -1) continue;
                    if (!shouldCoalesce(*p, *n)) continue;

                    if (!acceptNuclei({p, n})) continue;

                    frag_number++;
                    p -> setClusterID(frag_number);
                    n -> setClusterID(frag_number);
                    new_particles.push_back(mergeToCluster({p, n}));
                    goto next_deuteron;
                }
                next_deuteron:;
            }
            break;

        // Triton (1p + 2n)
        case ParticleType::Triton:
            for (auto* p : protons) {
                if (p -> getClusterID() != -1) continue;
                for (size_t i = 0; i < neutrons.size(); ++i) {
                    if (neutrons[i] -> getClusterID() != -1) continue;
                    if (!shouldCoalesce(*neutrons[i], *p)) continue;
                    CCParticle pair1 = mergeToCluster({neutrons[i], p});

                    for (size_t j = 0; j < neutrons.size(); ++j) {
                        if(i == j) continue;
                        if (neutrons[j] -> getClusterID() != -1) continue;
                        if (!shouldCoalesce(pair1, *neutrons[j])) continue;

                        if (!acceptNuclei({p, neutrons[i], neutrons[j]})) continue;

                        frag_number++;
                        p           -> setClusterID(frag_number);
                        neutrons[i] -> setClusterID(frag_number);
                        neutrons[j] -> setClusterID(frag_number);
                        new_particles.push_back(mergeToCluster({p, neutrons[i], neutrons[j]}));
                        goto next_triton;
                    }
                }
                next_triton:;
            }
            break;

        // Helium-3 (2p + 1n)
        case ParticleType::Helium3:
            for (auto* n : neutrons) {
                if (n -> getInCluster()) continue;
                for (size_t i = 0; i < protons.size(); ++i) {
                    if (protons[i] -> getClusterID() != -1) continue;
                    if (!shouldCoalesce(*protons[i], *n)) continue;
                    CCParticle pair1 = mergeToCluster({protons[i], n});

                    for (size_t j = 0; j < protons.size(); ++j) {
                        if(i == j) continue;
                        if (protons[j] -> getClusterID() != -1) continue;
                        if (!shouldCoalesce(pair1, *protons[j])) continue;
                        
                        if (!acceptNuclei({protons[i], protons[j], n})) continue;

                        frag_number++;
                        protons[i]  -> setClusterID(frag_number);
                        protons[j]  -> setClusterID(frag_number);
                        n           -> setClusterID(frag_number);
                        new_particles.push_back(mergeToCluster({protons[i], protons[j], n}));

                        goto next_he3;
                    }
                }
                next_he3:;
            }
            break;

        // He4 (2p + 2n)
        case ParticleType::Alpha:
            for (size_t i = 0; i < protons.size(); ++i) {
                if (protons[i] -> getClusterID() != -1) continue;

                for (size_t j = 0; j < neutrons.size(); ++j) {
                    if (neutrons[j] -> getClusterID() != -1) continue;
                    if (!shouldCoalesce(*protons[i], *neutrons[j])) continue;
                    CCParticle pair1 = mergeToCluster({protons[i], neutrons[j]});

                    for (size_t k = 0; k < protons.size(); ++k) {
                        if(k == i) continue;
                        if (protons[k] -> getClusterID() != -1) continue;
                        if (!shouldCoalesce(pair1, *protons[k])) continue;
                        CCParticle pair2 = mergeToCluster({&pair1, protons[k]});

                        for (size_t l = 0; l < neutrons.size(); ++l) {
                            if(l == j) continue;
                            if (neutrons[l] -> getClusterID() != -1) continue;
                            if (!shouldCoalesce(pair2, *neutrons[l])) continue;

                            if (!acceptNuclei({protons[i], neutrons[j],
                                               protons[k], neutrons[l]})) continue;

                            frag_number++;
                            protons[i]  -> setClusterID(frag_number);
                            protons[k]  -> setClusterID(frag_number);
                            neutrons[j] -> setClusterID(frag_number);
                            neutrons[l] -> setClusterID(frag_number);
                            new_particles.push_back(mergeToCluster({protons[i], neutrons[j],
                                                                    protons[k], neutrons[l]}));
                            goto next_he4;
                        }
                    }
                }
                next_he4:;
            }
            break;

        // H3L (1p + 1n + 1l)
        case ParticleType::H3L:
            for (auto* y : hyperons) {
                if (y -> getClusterID() != -1) continue;
                for (auto* n : neutrons) {
                    if (n -> getClusterID() != -1) continue;
                    if (!shouldCoalesce(*y, *n)) continue;
                    CCParticle pair1 = mergeToCluster({y, n});
                    for (auto* p : protons) {
                        if (p -> getClusterID() != -1) continue;
                        if (!shouldCoalesce(pair1, *p)) continue;

                        if (!acceptNuclei({p, n, y})) continue;

                        frag_number++;
                        p -> setClusterID(frag_number);
                        y -> setClusterID(frag_number);
                        n -> setClusterID(frag_number);
                        new_particles.push_back(mergeToCluster({p, n, y}));
                            
                        goto next_h3l;
                    }
                }
                next_h3l:;
            }
            break;

        // H4L (1p + 2n + 1l)
        case ParticleType::H4L:
            for (auto* y : hyperons) {
                if (y -> getClusterID() != -1) continue;

                for (size_t j = 0; j < neutrons.size(); ++j) {
                    if (neutrons[j] -> getClusterID() != -1) continue;
                    if (!shouldCoalesce(*y, *neutrons[j])) continue;
                    CCParticle pair1 = mergeToCluster({y, neutrons[j]});

                    for (size_t k = 0; k < neutrons.size(); ++k) {
                        if(k == j) continue;
                        if (neutrons[k] -> getClusterID() != -1) continue;
                        if (!shouldCoalesce(pair1, *neutrons[k])) continue;
                        CCParticle pair2 = mergeToCluster({&pair1, neutrons[k]});

                        for (auto* p : protons) {
                            if (p -> getClusterID() != -1) continue;
                            if (!shouldCoalesce(pair2, *p)) continue;

                            if (!acceptNuclei({p, neutrons[j], neutrons[k], y})) continue;

                            frag_number++;
                            p           -> setClusterID(frag_number);
                            y           -> setClusterID(frag_number);
                            neutrons[j] -> setClusterID(frag_number);
                            neutrons[k] -> setClusterID(frag_number);
                            new_particles.push_back(mergeToCluster({p, neutrons[j], neutrons[k], y}));

                            goto next_h4l;
                        }
                    }
                }
                next_h4l:;
            }
            break;


        default:
            throw std::invalid_argument("Unknown clusterType");
    }


    for (auto& p : particles) {
        if (p.getClusterID() == -1)
            new_particles.push_back(p);
    }

    particles = std::move(new_particles);
}

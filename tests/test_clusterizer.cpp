#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include "libccl.h"

#define VERBOSE true

bool equal_ignore_order(const std::vector<int>& a, const std::vector<int>& b) {
    return a.size() == b.size() &&
           std::is_permutation(a.begin(), a.end(), b.begin());
}



// ---------------------------------------------------------------------
// Binding energy calculation test
// ---------------------------------------------------------------------
void run_ebind_tests() {
    std::ifstream infile("event.dat");
    if (!infile.is_open()) {
        std::cerr << "Failed to open event.dat\n";
        return;
    }

    CClusterizer *cls = new CClusterizer();
    cls -> setRadius(4.0);
    cls -> setCheckMomentum(0);
    cls -> setAlphaBetaGamma(-0.125333, 0.071, 2.0); // QMD params for this test
    cls -> setComputeEbind(1);
    cls -> setComputeCoul(1);

    cls -> setSkyrmeFormula([](const SkyrmeInput& in) {
        double A = 0.0, B = 0.0, C = 0.0;
        for (double v : in.pair_density) {
            A += v;
            B += std::pow(v, in.gamma);
            C += v * std::log(1 + v);
        }
        return in.alpha0 * A + in.beta0 * B + 0.01 * C;
    });

    std::vector<CCParticle> particles;
    std::vector<int> cluster1 = {66686, 66852};
    std::vector<int> cluster2 = {66703, 66835, 66864};
    std::vector<int> cluster3 = {66685, 66688, 66850};
    std::vector<int> needed_particles;
    needed_particles.insert(needed_particles.end(), cluster1.begin(), cluster1.end());
    needed_particles.insert(needed_particles.end(), cluster2.begin(), cluster2.end());
    needed_particles.insert(needed_particles.end(), cluster3.begin(), cluster3.end());

    std::string line;
    std::getline(infile, line); // Header line 1

    while (std::getline(infile, line)) {
        CCParticle p;
        double px, py, pz, x, y, z, m;
        int nr, uid, pdg;

        std::istringstream iss(line);
        if (!(iss >> nr >> pdg >> px >> py >> pz >> x >> y >> z >> m >> uid)) {
            continue;
        }

        if(std::find(needed_particles.begin(), needed_particles.end(), uid) == needed_particles.end())
           continue;

        p.setIDS(-1, pdg, nr);
        p.setPosition(x, y, z, 0);
        p.setMomentum(px, py, pz, m);
        p.setUniqueID(uid);
        particles.push_back(p);
        nr++;
    }
    std::cout << "Loaded " << particles.size() << " particles\n";
    assert(particles.size() == 8 && "Expected 8 particles loaded");

    std::vector<CCParticle> particles_after_mst;
    particles_after_mst = cls -> makeMST(particles);

    std::cout << "Found " << particles_after_mst.size() << " clusters\n";
    assert(particles_after_mst.size() == 3 && "Expected 3 clusters found");


    for(size_t i = 0; i < particles_after_mst.size(); ++i){
        assert(particles_after_mst[i].getPID() > 99999 && "Expected PDG code larger than 99999");
        CCParticle cluster = particles_after_mst[i];


        auto childs     = cluster.getChilds();
        auto childs_ptrs = cluster.getChildsPtrs();
        std::vector<int> child_uids;
        for (auto& c : childs) {
            child_uids.push_back(c.getUniqueID());
        }

        assert(equal_ignore_order(child_uids, cluster1) ||
               equal_ignore_order(child_uids, cluster2) ||
               equal_ignore_order(child_uids, cluster3) );

        if (VERBOSE) {
            std::cout << "====================================================\n";
            std::cout << "Cluster: "<< i << ", PDG: "<< cluster.getPID()<<"\n";
            std::cout << "====================================================\n";
            std::cout << "Childs:\n";
        
            for (auto& c : childs) {
                std::cout << "Selected particle UniqueID: " << c.getUniqueID() << "\n";
            }

            DebugInfo dbg;
            double energy = computeClusterPotentialEnergy(childs_ptrs, cls, &dbg);
            std::cout << "Cluster bx: " << dbg.b_values[0]  << " by: " << dbg.b_values[1]  << " bz: " << dbg.b_values[2]  << "\n";
            for (size_t i = 0; i < dbg.boosted_states.size(); ++i) {
                auto& arr = dbg.boosted_states[i];
                std::cout << "Particle " << i << " boosted state:\n";
                std::cout << "  Px=" << arr[0] << " Py=" << arr[1] << " Pz=" << arr[2]
                          << " X="  << arr[3] << " Y="  << arr[4] << " Z="  << arr[5] << "\n";
            }
            std::cout << "r2 values:\n";
            for (auto v : dbg.r2_values) std::cout << "  " << v << "\n";
            std::cout << "rho values:\n";
            for (auto v : dbg.rho_values) std::cout << "  " << v << "\n";
            std::cout << "E_pot: "    << energy/(double)childs.size()       << "\n";
        }
    }
}




// ---------------------------------------------------------------------
// MST test
// ---------------------------------------------------------------------
void run_mst_tests() {
    CClusterizer cls;
    cls.setRadius(4.0);
    cls.setCheckMomentum(0);
    cls.setComputeEbind(0);

    cls.setSkyrmeFormula([](const SkyrmeInput& in) {
        double A = 0.0, B = 0.0, C = 0.0;
        for (double v : in.pair_density) {
            A += v;
            B += std::pow(v, in.gamma);
            C += v * std::log(1 + v);
        }
        return in.alpha0 * A + in.beta0 * B + 0.01 * C;
    });

    std::ifstream infile("event.dat");
    if (!infile.is_open()) {
        std::cerr << "Failed to open event.dat\n";
        return;
    }
    std::vector<CCParticle> particles;
    std::string line;
    std::getline(infile, line); // Header line 1

    while (std::getline(infile, line)) {
        CCParticle p;
        double px, py, pz, x, y, z, m;
        int nr, uid, pdg;

        std::istringstream iss(line);
        if (!(iss >> nr >> pdg >> px >> py >> pz >> x >> y >> z >> m >> uid)) {
            continue;
        }

        p.setIDS(-1, pdg, nr);
        p.setPosition(x, y, z, 0);
        p.setMomentum(px, py, pz, m);
        p.setUniqueID(uid);
        particles.push_back(p);
    }
    std::cout << "Loaded " << particles.size() << " particles\n";
    assert(particles.size() == 203 && "Expected 203 particles loaded");

    std::vector<CCParticle> particles_after_mst;
    particles_after_mst = cls.makeMST(particles);

    int ncls = 0;
    for(size_t i = 0; i < particles_after_mst.size(); ++i){
        if (particles_after_mst[i].getPID() > 9999) ncls++;
    }
    std::cout << "Found " << ncls << " clusters\n";
    assert(ncls == 10 && "Expected 10 particles loaded");


    for(size_t i = 0; i < particles_after_mst.size(); ++i){
        if (particles_after_mst[i].getPID() < 99999) continue;
        CCParticle cluster = particles_after_mst[i];

        auto childs      = cluster.getChilds();
        auto childs_ptrs = cluster.getChildsPtrs();

        if (VERBOSE) {
            std::cout << "====================================================\n";
            std::cout << "Cluster: "<< i << ", PDG: "<< cluster.getPID()<<"\n";
            std::cout << "Child count: " << childs.size() << "\n";

            DebugInfo dbg;
            double energy = computeClusterPotentialEnergy(childs_ptrs, &cls, &dbg);
            std::cout << "E_pot: "    << energy/(double)childs.size()       << "\n";
            std::cout << "Minimal proximity pairs:\n";
        }
        std::vector<int> connected;
        connected.push_back(0);
        std::set<int> used = {0};
        while (connected.size() < childs.size()) {
            bool added = false;

            for (int a : connected) {
                for (size_t b = 0; b < childs.size(); ++b) {
                    if (used.count(b)) continue;

                    if (cls.proximityCheckMST(*childs_ptrs[a], *childs_ptrs[b])) {
                        // Compute distance in COM frame
                        CCParticle tmp1 = *childs_ptrs[a];
                        CCParticle tmp2 = *childs_ptrs[b];

                        double Tpair =  tmp1.getE()  + tmp2.getE();
                        double bx    = (tmp1.getPx() + tmp2.getPx()) / Tpair;
                        double by    = (tmp1.getPy() + tmp2.getPy()) / Tpair;
                        double bz    = (tmp1.getPz() + tmp2.getPz()) / Tpair;

                        tmp1.LorentzBoost(bx, by, bz);
                        tmp2.LorentzBoost(bx, by, bz);

                        double dx = tmp1.getX() - tmp2.getX();
                        double dy = tmp1.getY() - tmp2.getY();
                        double dz = tmp1.getZ() - tmp2.getZ();
                        double dr = std::sqrt(dx * dx + dy * dy + dz * dz);

                        if (VERBOSE) {
                            std::cout << "Pair (" << tmp1.getUniqueID() << ", "
                                      << tmp2.getUniqueID() << ")"
                                      << " distance = " << dr << "\n";
                        }
                        assert(dr < 4.0 && "Expected distance dr < 4.0");
                        connected.push_back(b);
                        used.insert(b);
                        added = true;
                        break;
                    }
                }
                if (added) break;
            }

            if (!added) {
                std::cerr << "Warning: Could not connect all particles!\n";
                break;
            }
        }
        if (VERBOSE) std::cout << "====================================================\n";
    }
}





// ---------------------------------------------------------------------
// Simulated Annealing test
// Due to probabilistic nature it is mostly verbose checks
// Mode 1: MST clusters are passed to the makeSA() precedure one by one
// ---------------------------------------------------------------------
void run_sa_tests_1() {
    CClusterizer cls;
    cls.setRadius(4.0);
    cls.setCheckMomentum(0);
    cls.setComputeEbind(0);

    cls.setSkyrmeFormula([](const SkyrmeInput& in) {
        double A = 0.0, B = 0.0, C = 0.0;
        for (double v : in.pair_density) {
            A += v;
            B += std::pow(v, in.gamma);
            C += v * std::log(1 + v);
        }
        return in.alpha0 * A + in.beta0 * B + 0.01 * C;
    });

    std::ifstream infile("event.dat");
    if (!infile.is_open()) {
        std::cerr << "Failed to open event.dat\n";
        return;
    }
    std::vector<CCParticle> particles;
    std::string line;
    std::getline(infile, line); // Header line 1

    while (std::getline(infile, line)) {
        CCParticle p;
        double px, py, pz, x, y, z, m;
        int nr, uid, pdg;

        std::istringstream iss(line);
        if (!(iss >> nr >> pdg >> px >> py >> pz >> x >> y >> z >> m >> uid)) {
            continue;
        }

        p.setIDS(-1, pdg, nr);
        p.setPosition(x, y, z, 0);
        p.setMomentum(px, py, pz, m);
        p.setUniqueID(uid);
        particles.push_back(p);
    }
    std::cout << "Loaded " << particles.size() << " particles\n";

    std::vector<CCParticle> particles_after_mst;
    particles_after_mst = cls.makeMST(particles);

    for(size_t i = 0; i < particles_after_mst.size(); ++i){
        if (particles_after_mst[i].getPID() < 99999) continue;
        CCParticle cluster = particles_after_mst[i];

        auto childs      = cluster.getChilds();
        auto childs_ptrs = cluster.getChildsPtrs();

        auto particles_after_sa = cls.makeSA(childs);
        if (VERBOSE) {
            std::cout << "====================================================\n";
            std::cout << "Cluster: "<< i << ", PDG: "<< cluster.getPID();
            std::cout << ", Ebind: " << computeBindingEnergy(childs_ptrs, &cls) * childs.size()<<"\n";
            std::cout << "====================================================\n";
            std::cout << "Child count: " << childs.size() << "\n";
            for (auto& p : particles_after_sa) {
                auto sa_childs      = p.getChilds();
                auto sa_childs_ptrs = p.getChildsPtrs();
                if(sa_childs.size() > 1){
                    std::cout << "SA bound: "<<  p.getPID()<< " Ebind: "
                    <<  computeBindingEnergy(sa_childs_ptrs, &cls) * sa_childs.size()<<"\n";
                }
                else{
                    std::cout << "SA unbound: "<<  p.getPID()<<"\n";
                }
            }
        }
    }
}




// ---------------------------------------------------------------------
// Simulated Annealing test
// Due to probabilistic nature it is mostly verbose checks
// Mode 2: all baryons (free and bound) are passed to makeSA() at once
// ---------------------------------------------------------------------
void run_sa_tests_2() {
    CClusterizer cls;
    cls.setRadius(4.0);
    cls.setCheckMomentum(0);
    cls.setComputeEbind(0);
    cls.setSASteps(1000);
    cls.setSAPnew(0.25);

    cls.setSkyrmeFormula([](const SkyrmeInput& in) {
        double A = 0.0, B = 0.0, C = 0.0;
        for (double v : in.pair_density) {
            A += v;
            B += std::pow(v, in.gamma);
            C += v * std::log(1 + v);
        }
        return in.alpha0 * A + in.beta0 * B + 0.01 * C;
    });

    std::ifstream infile("event.dat");
    if (!infile.is_open()) {
        std::cerr << "Failed to open event.dat\n";
        return;
    }
    std::vector<CCParticle> particles;
    std::string line;
    std::getline(infile, line); // Header line 1

    while (std::getline(infile, line)) {
        CCParticle p;
        double px, py, pz, x, y, z, m;
        int nr, uid, pdg;

        std::istringstream iss(line);
        if (!(iss >> nr >> pdg >> px >> py >> pz >> x >> y >> z >> m >> uid)) {
            continue;
        }

        p.setIDS(-1, pdg, nr);
        p.setPosition(x, y, z, 0);
        p.setMomentum(px, py, pz, m);
        p.setUniqueID(uid);
        particles.push_back(p);
    }
    std::cout << "Loaded " << particles.size() << " particles\n";

    std::vector<CCParticle> particles_after_mst;
    particles_after_mst = cls.makeMST(particles);

    std::vector<CCParticle> pool;
    for (auto& p : particles_after_mst) {
        if (p.hasChilds()) {
            auto childs_ptrs = p.getChilds();
            for (auto& c : childs_ptrs) {
                pool.push_back(c);         // bound baryons
            }
        }
        else {
            pool.push_back(p);             // free baryons
        }
    }

    auto particles_after_sa = cls.makeSA(pool);
        if (VERBOSE) {
        for (auto& p : particles_after_sa) {
            auto sa_childs      = p.getChilds();
            auto sa_childs_ptrs = p.getChildsPtrs();
            if(sa_childs.size() > 1){
                std::cout << "SA bound: "<<  p.getPID()<< " Ebind: "
                <<  computeBindingEnergy(sa_childs_ptrs, &cls) * sa_childs.size()<<"\n";
            }
            else{
                std::cout << "SA unbound: "<<  p.getPID()<<"\n";
            }
        }
    }
}



// ---------------------------------------------------------------------
// Main routine
// ---------------------------------------------------------------------
int main() {
    std::cout << "=======\n";
    std::cout << "Energy tests:\n";
    run_ebind_tests();

    std::cout << "=======\n";
    std::cout << "MST tests\n";
    run_mst_tests();

    std::cout << "=======\n";
    std::cout << "SA tests, mode 1\n";
    run_sa_tests_1();

    std::cout << "=======\n";
    std::cout << "SA tests, mode 2\n";
    run_sa_tests_2();

    return 0;
}


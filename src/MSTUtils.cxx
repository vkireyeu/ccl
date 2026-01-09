#include "libccl.h"
#include <algorithm>
#include <map>


// ---------------------------------------------------------------------
// Distance check: return true if particles are in some radius; false - if not
// Needed for MST
// ---------------------------------------------------------------------
bool CClusterizer::proximityCheckMST(const CCParticle& p1, const CCParticle& p2) const {
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
    if(TrueTimeMST){
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
    }

    // Calculate distance in synchronized COM frame
    const double dx  = tmp1.getX() - tmp2.getX();
    const double dy  = tmp1.getY() - tmp2.getY();
    const double dz  = tmp1.getZ() - tmp2.getZ();
    const double dr2 = dx*dx + dy*dy + dz*dz;

    // Radius check
    if(dr2 > Radius*Radius) return false;

    return true;
}


// ---------------------------------------------------------------------
// Momentum check: return true if particles passes the momentum check
// Optional for MST
// ---------------------------------------------------------------------
bool CClusterizer::passesMomentumCheck(const std::vector<CCParticle*>& group) const {
    CCParticle clust = mergeToCluster(group);
    double bx = clust.getPx() / clust.getE();
    double by = clust.getPy() / clust.getE();
    double bz = clust.getPz() / clust.getE();

    for (const auto* p : group) {
        CCParticle boosted = *p;
        boosted.LorentzBoost(bx, by, bz);
        if (boosted.getP() >= Momentum) return false;
    }
    return true;
}




// ---------------------------------------------------------------------
// DSU-based (union-find) MST implementation
// ---------------------------------------------------------------------
std::vector<CCParticle> CClusterizer::makeMST(const std::vector<CCParticle>& input){
    std::vector<CCParticle> particles = input;
    if (particles.empty()) return {};

    // Particles vector shuffle
    std::shuffle(particles.begin(), particles.end(), std::default_random_engine{});
    const size_t N = particles.size();

    // Union-Find structures
    std::vector<int> parent(N);
    std::iota(parent.begin(), parent.end(), 0); // values from 0 to N: parent[i] = i


    // Helper routine: returns the root of particle x
    auto find = [&](int x) {
        while (x != parent[x]) x = parent[x] = parent[parent[x]];
        return x;
    };

    // Helper routine: connects two clusters if they are not already connected
    auto unite = [&](int a, int b) {
        int rootA = find(a);        // Find the root of the cluster A containing particle a
        int rootB = find(b);        // Find the root of the cluster B containing particle b
        if (rootA == rootB) return; // Do nothing is the roots of the cluster are already the same

        // On-line momentum checks
        if (CheckMomentum) {
            std::vector<CCParticle*> group;

            // Collect all particles from both clusters
            for (size_t i = 0; i < particles.size(); ++i) {
                int r = find(i);
                if (r == rootA || r == rootB)
                    group.push_back(&particles[i]);
            }

            // Build temporary merged cluster
            CCParticle tempCluster = mergeToCluster(group);
            double bx = tempCluster.getPx() / tempCluster.getE();
            double by = tempCluster.getPy() / tempCluster.getE();
            double bz = tempCluster.getPz() / tempCluster.getE();

            // Boost particles to the temporary cluster COM frame and check momentum
            for (auto* p : group) {
                CCParticle boosted = *p;
                boosted.LorentzBoost(bx, by, bz);
                if (boosted.getP() >= Momentum) {
                    return; // Reject merge
                }
            }
        }

        // Merge clusters -- set the root of cluster B to the root of cluster A
        parent[rootB] = rootA; // Cluster B now belongs to the same root as cluster A (to the same cluster)
    };

    // Proximity-based clustering for every unique pair
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            if (proximityCheckMST(particles[i], particles[j])) {
                if(CheckPairMomentum){
                    std::vector<CCParticle*> group = {&particles[i], &particles[j]};
                    if (passesMomentumCheck(group)) unite(i, j);
                }
                else {
                    unite(i, j);
                }
            }
        }
    }

    // Assign cluster IDs
    std::unordered_map<int, short> cluster_map;
    short next_cluster_id = 0;
    for (size_t i = 0; i < N; ++i) {
        int root = find(i);                             // Root of current particle (or cluster, if it belongs to the cluster)
        if (cluster_map.count(root) == 0) {             // If root is NOT in the map
            cluster_map[root] = next_cluster_id++;      // assign new ID to this root in the map
        }
        particles[i].setClusterID(cluster_map[root]);   // Assign same root (cluster) ID to the particle
    }

    // Merge clusters into new particles
    std::vector<CCParticle> new_particles;
    for (short cid = 0; cid < next_cluster_id; ++cid) {
        std::vector<CCParticle*> group;
        for (auto& p : particles) {
            if (p.getClusterID() == cid) {
                group.push_back(&p);
            }
        }
        if (group.size() > 1) {
            for (auto* p : group) p->setInCluster(true);
            new_particles.push_back(mergeToCluster(group));
        }
        else if (group.size() == 1) {
            for (auto* p : group){
                p -> setClusterID(-1);
                p -> clearChilds();
                p -> setInCluster(false);
            }
        }
    }

    // Single particles
    for (size_t i = 0; i < N; ++i) {
        if (!particles[i].getInCluster()) {
            particles[i].clearChilds();
            particles[i].setClusterID(-1);
            new_particles.push_back(particles[i]);
        }
    }

    return new_particles;
}


// This struct handles information of a move in makeSA() and makeSA2():
struct MoveRecord {
    int src_id = -1;                // source cluster ID
    int dst_id = -1;                // destination cluster ID
    size_t src_index = 0;           // ID of baryon to move in source cluster
    size_t dst_index = 0;           // ID of baryon to move in destination cluster
    bool removed_src = false;       // true if source cluster dissapear after the move
    bool created_dst = false;       // true if destination cluster is a new (non existing!) cluster
    CCParticle* moved = nullptr;    // point to the baryon what was moved
};


// ---------------------------------------------------------------------
// Simulated Annealing procedure implementation: 
//   must be applied after makeMST() procedure
// ---------------------------------------------------------------------
std::vector<CCParticle> CClusterizer::makeSA(const std::vector<CCParticle>& input){
    if (input.empty()) return {};
    // Make a working copy and a pointer view for internal routines
    std::vector<CCParticle>  baryons_copy = input;

    const size_t N = input.size();

    // Simulated annealing parameters 
    double T               =  sa_Tmax;        // Initial "temperature"
    const double Tmin      =  sa_Tmin;        // Minimal tempersture
    const double cool      =  sa_Cool;        // Cooling rate
    const bool   useMH     =  sa_MHCriterion; // Metropolis–Hastings criterion (use it or not)


    // Helper: binding energy of a single cluster (0 for singletons).
    auto cluster_energy = [&](const std::vector<CCParticle*>& cluster) -> double {
        size_t constituents = cluster.size();
        if (constituents > 1) return computeBindingEnergy(cluster, this) * constituents;
        return 0.0;
    };


    // Starting system
    std::map<int, std::vector<CCParticle*>> mapClusters;
    for (auto &p : baryons_copy) {
        int cls_id = p.getClusterID();
        mapClusters[cls_id].push_back(&p);
    }


    double totalE = 0.0;
    // Per-cluster energy also will be stored now
    std::map<int, double> clusterE;
    for (const auto& pair : mapClusters) {
        double e = cluster_energy(pair.second);
        clusterE[pair.first] = e;
        totalE += e;
    }

    double currE = totalE;                                            // 'Current' energy variable
    double bestE = totalE;                                            // 'Best' (lowest) energy found
    std::map<int, std::vector<CCParticle*>> bestSystem = mapClusters; // 'Best' system with the lowest binding energy

    // List of clusters IDs
    std::vector<int> activeClusters;
    activeClusters.reserve(mapClusters.size());
    for (const auto& pair : mapClusters){
        activeClusters.push_back(pair.first);
    }
    int maxClusterId = mapClusters.rbegin() -> first;

    // Helper function for removal cluster from activeClusters
    auto remove_active = [&](int id) {
        auto it = std::lower_bound(activeClusters.begin(), activeClusters.end(), id);
        if (it != activeClusters.end() && *it == id) activeClusters.erase(it);
    };
    // Helper function to add cluster there
    auto add_active = [&](int id) {
        auto it = std::lower_bound(activeClusters.begin(), activeClusters.end(), id);
        activeClusters.insert(it, id);
    };


    // Classical SA
    int sa_level = 0;
    while (T > Tmin) {                                // While current "temperature" is higher than minimal
        int steps = 0;
        // Number of tries/steps per temperature level:
        switch (sa_StepsMode) {
            case SAStepsMode::Fixed:       steps = sa_Steps; break;  // fixed
            case SAStepsMode::Linear:      steps = static_cast<int>(sa_Steps * (T / sa_Tmax)); break; // linear decrease
            case SAStepsMode::Exponential: steps = static_cast<int>(sa_Steps * std::pow(cool, sa_level)); break;  // exponential decrease
        }
        steps = std::max(1, steps); // To avoid zero

        // Probability of making new cluster/singleton
        double pNew = sa_Pnew;
        if(sa_PnewMode == SAProbMode::Linear) {
            double ratio = (T - sa_Tmin) / (sa_Tmax - sa_Tmin);
            pNew = sa_PnewMin + (sa_Pnew - sa_PnewMin) * ratio;
        }

        // Stagnation counter
        int stagnant = 0;
        const int stagnation_limit = std::max(sa_StagMin, steps / std::max(1, sa_StagDenom));


        //   loop over steps for each "temperature" level
        for (int s = 0; s < steps; ++s) {
            std::uniform_int_distribution<int> pickClusterIndex(0, activeClusters.size() - 1);

            double PROB = U01()(get_rng());

            // Only free particles (no clusters anymore) -- only merge step
            if(activeClusters.size() == N) {
                PROB = pNew * 1.1;
            }
            // Only one single cluster available -- only 'new singletop' step
            else if(activeClusters.size() == 1) {
                PROB = pNew * 0.1;
            }

            MoveRecord rec;
            int testBaryonIdx = 0;
            size_t inputClusterSize = 0;

            // Move some baryon from some cluster to the singleton (it will be free)
            if (PROB < pNew) {
                while (inputClusterSize < 2){
                    rec.src_id = activeClusters[pickClusterIndex(get_rng())];
                    inputClusterSize = mapClusters[rec.src_id].size();
                }
                std::uniform_int_distribution<int> pickClusterBaryon(0, int(mapClusters[rec.src_id].size()) - 1);
                testBaryonIdx = pickClusterBaryon(get_rng());

                int newClusterId = ++maxClusterId;

                rec.dst_id       = newClusterId;
                rec.created_dst  = true;
                rec.src_index    = static_cast<size_t>(testBaryonIdx);
                rec.dst_index    = 0;
                rec.moved        = mapClusters[rec.src_id][testBaryonIdx];

                mapClusters[newClusterId].push_back(rec.moved);
                mapClusters[rec.src_id].erase(mapClusters[rec.src_id].begin() + testBaryonIdx);
                add_active(rec.dst_id);
            }
            // Move some baryon, single or from some cluster to another cluster or existing singleton (so it will be cluster too)
            else {
                rec.src_id = activeClusters[pickClusterIndex(get_rng())];
                rec.dst_id = activeClusters[pickClusterIndex(get_rng())];
                inputClusterSize = mapClusters[rec.src_id].size();
                if (inputClusterSize > 1){
                    std::uniform_int_distribution<int> pickClusterBaryon(0, int(mapClusters[rec.src_id].size()) - 1);
                    testBaryonIdx = pickClusterBaryon(get_rng());
                }
                
                while (rec.src_id == rec.dst_id){
                    rec.dst_id = activeClusters[pickClusterIndex(get_rng())];
                }

                rec.src_index = static_cast<size_t>(testBaryonIdx);
                rec.dst_index = mapClusters[rec.dst_id].size();
                rec.moved     = mapClusters[rec.src_id][testBaryonIdx];

                mapClusters[rec.dst_id].push_back(rec.moved);

                if (inputClusterSize == 1){
                    mapClusters[rec.src_id].erase(mapClusters[rec.src_id].begin() + testBaryonIdx);
                    mapClusters.erase(rec.src_id);
                    rec.removed_src = true;
                    remove_active(rec.src_id);
                }
                else{
                    mapClusters[rec.src_id].erase(mapClusters[rec.src_id].begin() + testBaryonIdx);
                }
            }

            // Old binding energy of modified clusters
            double oldEin  = 0.0;
            double oldEout = 0.0;
            if (clusterE.count(rec.src_id)) oldEin  = clusterE[rec.src_id];
            if (clusterE.count(rec.dst_id)) oldEout = clusterE[rec.dst_id];

            // New binding energy of modified clusters
            double newEin  = 0.0;
            double newEout = 0.0;
            if (mapClusters.count(rec.src_id)){
                newEin = cluster_energy(mapClusters[rec.src_id]);
            }
            if (mapClusters.count(rec.dst_id)){
                newEout = cluster_energy(mapClusters[rec.dst_id]);
            }

            // New total binding energy of the system
            double newE = currE + (newEin - oldEin) + (newEout - oldEout);

            double dE   = newE - currE;
            bool accept = false;

            // New configuration has ***lower*** energy -> accept
            if (dE < 0.0) {
                accept = true;
            } 
            // New configuration has ***higher*** energy -> accept with probability exp(-dE/T)
            else if (useMH){
                if (U01()(get_rng()) < std::exp(-dE / T)) {
                    accept = true; // Accept uphill move to escape local minimum
                }
            }

            if(!accept) {
                if (rec.created_dst){
                    mapClusters[rec.dst_id].erase(mapClusters[rec.dst_id].begin() + rec.dst_index);
                    mapClusters.erase(rec.dst_id);
                    remove_active(rec.dst_id);
                    mapClusters[rec.src_id].insert(mapClusters[rec.src_id].begin() + rec.src_index, rec.moved);
                }
                else{
                    mapClusters[rec.dst_id].erase(mapClusters[rec.dst_id].begin() + rec.dst_index);
                    if (rec.removed_src){
                        mapClusters[rec.src_id].push_back(rec.moved);
                        add_active(rec.src_id);
                    }
                    else{
                        mapClusters[rec.src_id].insert(mapClusters[rec.src_id].begin() + rec.src_index, rec.moved);
                    }
                }
                // Stagnation increased here
                stagnant++;
                if (stagnant >= stagnation_limit) break;
                continue;
            }

            stagnant = 0;
            currE = newE;
            if (mapClusters.count(rec.src_id)){
                clusterE[rec.src_id] = newEin;
            } else {
                clusterE.erase(rec.src_id);
            }
            if (mapClusters.count(rec.dst_id)){
                clusterE[rec.dst_id] = newEout;
            } else {
                clusterE.erase(rec.dst_id);
            }
            if (currE < bestE){
                bestE      = currE;
                bestSystem = mapClusters;
            }
        }
        T *= cool; // cool down
        sa_level++;
    }

    std::vector<CCParticle> new_particles;
    for (const auto& pair : bestSystem) {
        int    finalClusterID    = pair.first;
        size_t finalClusterSize  = pair.second.size();
        if (finalClusterSize > 1){
            for (auto c: pair.second){
                c -> setClusterID(finalClusterID);
            }
            CCParticle merged = mergeToCluster(pair.second);
            merged.setUniqueID(999E5 + finalClusterID);
            new_particles.push_back(std::move(merged));
        }
        else{
            CCParticle single = *pair.second.front();
            single.clearChilds();
            single.setClusterID(-1);
            new_particles.push_back(std::move(single));
        }
    }

    return new_particles;
}




// ---------------------------------------------------------------------
// Simulated Annealing procedure implementation -- clusters growth only: 
//   must be applied after the makeSA() procedure
// ---------------------------------------------------------------------
std::vector<CCParticle> CClusterizer::makeSA2(const std::vector<CCParticle>& input){
    if (input.empty()) return {};
    // Make a working copy and a pointer view for internal routines

    std::vector<CCParticle>  particles_copy = input;
    const size_t N = particles_copy.size();

    // Simulated annealing parameters 
    double T               =  sa_Tmax;        // Initial "temperature"
    const double Tmin      =  sa_Tmin;        // Minimal tempersture
    const double cool      =  sa_Cool;        // Cooling rate
    const bool   useMH     =  sa_MHCriterion; // Metropolis–Hastings criterion (use it or not)


    // Helper: binding energy of a single cluster (0 for singletons).
    auto cluster_energy = [&](const std::vector<CCParticle*>& pvec) -> double {
        std::vector<CCParticle*> group;
        for (const auto& p : pvec) {
            auto   childsPtrs = p -> getChildsPtrs();
            size_t childsSize = childsPtrs.size();
            if (childsSize > 1){
                for(const auto &c: childsPtrs){
                    group.push_back(c);
                }
            }
            else{
                group.push_back(p);
            }
        }
        if(group.size() > 1){
            return computeBindingEnergy(group, this) * group.size();
        }
        return 0.0;
    };

    // Starting system
    std::map<int, std::vector<CCParticle*>>  mapSystem;
    for (auto &p : particles_copy) {
        int uid = p.getUniqueID();
        mapSystem[uid].push_back(&p);
    }
    std::map<int, double> clusterE;
    double totalE = 0.0;
    for (const auto& pair : mapSystem) {
        double e = cluster_energy(pair.second);
        clusterE[pair.first] = e;
        totalE += e;
    }

    double currE = totalE;                                          // 'Current' energy variable
    double bestE = totalE;                                          // 'Best' (lowest) energy found
    std::map<int, std::vector<CCParticle*>> bestSystem = mapSystem; // 'Best' system with the lowest binding energy
    std::vector<int> activeClusters;
    activeClusters.reserve(mapSystem.size());
    for (const auto& pair : mapSystem) activeClusters.push_back(pair.first);

    auto remove_active = [&](int id) {
        auto it = std::lower_bound(activeClusters.begin(), activeClusters.end(), id);
        if (it != activeClusters.end() && *it == id) activeClusters.erase(it);
    };

    auto add_active = [&](int id) {
        auto it = std::lower_bound(activeClusters.begin(), activeClusters.end(), id);
        activeClusters.insert(it, id);
    };

    // Classical SA
    int sa_level = 0;
    while (T > Tmin) {                                // While current "temperature" is higher than minimal
        int steps = 0;
        // Number of tries/steps per temperature level:
        switch (sa_StepsMode) {
            case SAStepsMode::Fixed:       steps = sa_Steps; break;  // fixed
            case SAStepsMode::Linear:      steps = static_cast<int>(sa_Steps * (T / sa_Tmax)); break; // linear decrease
            case SAStepsMode::Exponential: steps = static_cast<int>(sa_Steps * std::pow(cool, sa_level)); break;  // exponential decrease
        }
        steps = std::max(1, steps); // To avoid zero

        int stagnant = 0;
        const int stagnation_limit = std::max(sa_StagMin, steps / std::max(1, sa_StagDenom));

        //   loop over steps for each "temperature" level
        for (int s = 0; s < steps; ++s) {
            std::uniform_int_distribution<int> pickClusterIndex(0, activeClusters.size() - 1);

            MoveRecord rec;
            rec.src_id = activeClusters[pickClusterIndex(get_rng())];
            rec.dst_id = activeClusters[pickClusterIndex(get_rng())];
            int testParticleIdx = 0;

            size_t inputClusterSize = mapSystem[rec.src_id].size();
            if (inputClusterSize > 1){
                std::uniform_int_distribution<int> pickParticle(0, int(mapSystem[rec.src_id].size()) - 1);
                testParticleIdx = pickParticle(get_rng());
            }
            
            while (rec.src_id == rec.dst_id){
                rec.dst_id = activeClusters[pickClusterIndex(get_rng())];
            }

            rec.src_index = static_cast<size_t>(testParticleIdx);
            rec.dst_index = mapSystem[rec.dst_id].size();
            rec.moved     = mapSystem[rec.src_id][testParticleIdx];

            mapSystem[rec.dst_id].push_back(rec.moved);

            if (inputClusterSize == 1){
                mapSystem[rec.src_id].erase(mapSystem[rec.src_id].begin() + testParticleIdx);
                mapSystem.erase(rec.src_id);
                rec.removed_src = true;
                remove_active(rec.src_id);
            }
            else{
                mapSystem[rec.src_id].erase(mapSystem[rec.src_id].begin() + testParticleIdx);
            }

            double oldEin  = 0.0;
            double oldEout = 0.0;
            if (clusterE.count(rec.src_id)) oldEin  = clusterE[rec.src_id];
            if (clusterE.count(rec.dst_id)) oldEout = clusterE[rec.dst_id];

            double newEin  = 0.0;
            double newEout = 0.0;
            if (mapSystem.count(rec.src_id)){
                newEin = cluster_energy(mapSystem[rec.src_id]);
            }
            if (mapSystem.count(rec.dst_id)){
                newEout = cluster_energy(mapSystem[rec.dst_id]);
            }

            double newE = currE + (newEin - oldEin) + (newEout - oldEout);
            double dE   = newE - currE;
            bool accept = false;

            // New configuration has ***lower*** energy -> accept
            if (dE < 0.0) {
                accept = true;
            } 

            if(!accept) {
                mapSystem[rec.dst_id].erase(mapSystem[rec.dst_id].begin() + rec.dst_index);
                if (rec.removed_src){
                    mapSystem[rec.src_id].push_back(rec.moved);
                    add_active(rec.src_id);
                }
                else{
                    mapSystem[rec.src_id].insert(mapSystem[rec.src_id].begin() + rec.src_index, rec.moved);
                }
                stagnant++;
                if (stagnant >= stagnation_limit) break;
                continue;
            }

            stagnant = 0;
            currE = newE;
            if (mapSystem.count(rec.src_id)){
                clusterE[rec.src_id] = newEin;
            } else {
                clusterE.erase(rec.src_id);
            }
            if (mapSystem.count(rec.dst_id)){
                clusterE[rec.dst_id] = newEout;
            } else {
                clusterE.erase(rec.dst_id);
            }
            if (currE < bestE){
                bestE      = currE;
                bestSystem = mapSystem;
            }

        }
        T *= cool; // cool down
        sa_level++;
    }


    // Yes, it is long
    std::vector<CCParticle> new_particles;
    for (const auto& pair : bestSystem) {
        int    uid  = pair.first;
        auto   pvec = pair.second;
        size_t size = pvec.size();

        if (size == 1){
            auto p = *pvec.front();
            auto   childsPtrs = p.getChildsPtrs();
            size_t childsSize = childsPtrs.size();
            if (childsSize < 2){
                p.clearChilds();
                p.setClusterID(-1);
            }
            new_particles.push_back(p);
        }
        else{
            std::vector<CCParticle*> group;
            for (const auto& p : pvec) {
                auto   childsPtrs = p -> getChildsPtrs();
                size_t childsSize = childsPtrs.size();
                if (childsSize < 2){
                    p -> clearChilds();
                    p -> setClusterID(uid);
                    group.push_back(p);
                }
                else{
                    for (const auto& c : childsPtrs) {
                        c -> clearChilds();
                        c -> setClusterID(uid);
                        group.push_back(c);
                    }
                }
            }
            CCParticle merged = mergeToCluster(group);
            merged.setBindingEnergy(computeBindingEnergy(group, this));
            merged.setUniqueID(uid);
            new_particles.push_back(merged);
        }
    }
    
    return new_particles;
}




// ---------------------------------------------------------------------
// Simulated Annealing chain:  makeSA() -> makeS2A()
//   must be used to get CCL SA results after makeMST()
// ---------------------------------------------------------------------
std::vector<CCParticle> CClusterizer::makeSAchain(const std::vector<CCParticle>& input){
    if (input.empty()) return {};
    std::vector<CCParticle>  mst_pass1 = input;
    const size_t N = input.size();

    std::vector<CCParticle> free;
    std::vector<CCParticle> bound_stable;
    std::vector<CCParticle> bound_unstable;
    std::vector<CCParticle> final_particles;

    const double EBCUT = sa_EbindBound;

    int max_clust_id = 0;
    for (auto &p : mst_pass1){
        short  PTYPE = static_cast<short>(p.getType());
        double EBIND = p.getBindingEnergy();
        int    CL_ID = p.getClusterID();
        if (CL_ID > max_clust_id) max_clust_id = CL_ID;

        if(PTYPE <= 2){
            free.push_back(p);
        }
        else if(EBIND <= EBCUT){
            bound_stable.push_back(p);
        }
        else if(EBIND >  EBCUT){
            bound_unstable.push_back(p);
        }
    }


    if(bound_unstable.size() > 0){
        std::vector<CCParticle> local_pass;

        for (auto &p : bound_unstable){
            auto childs = p.getChilds();

            std::vector<CCParticle> SA_local = makeSA(childs);
            for (auto &p_loc : SA_local){
                if (static_cast<short>(p_loc.getType()) > 2){
                    max_clust_id++;
                    p_loc.setClusterID(max_clust_id);
                    auto childs_loc = p_loc.getChilds();
                    double E_BND = p_loc.getBindingEnergy();
                    if(E_BND <= EBCUT){
                        for (auto &c_loc : childs_loc) c_loc.setClusterID(max_clust_id);
                        local_pass.push_back(p_loc);
                    }
                    else{
                        for (auto &c_loc : childs_loc){
                            c_loc.setClusterID(-1);
                            c_loc.setInCluster(false);
                            c_loc.clearChilds();
                            local_pass.push_back(c_loc);
                        }
                    }
                }
            }
        }

        for (auto &p : free) local_pass.push_back(p);
        for (auto &p : bound_stable) local_pass.push_back(p);

        CClusterizer *cls = new CClusterizer();
        cls -> setAlphaBetaGamma(CClusterizer::getAlpha(), CClusterizer::getBeta(), CClusterizer::getGamma());
        cls -> setSkyrmeFormula(CClusterizer::getSkyrmeFormula());
        cls -> setComputeEbind(true);
        cls -> setHyPot(CClusterizer::getHyPot());
        int sa_steps2 = CClusterizer::getSASteps2();
        if (sa_steps2 < 0) sa_steps2 = CClusterizer::getSASteps() * 2;
        cls -> setSASteps(sa_steps2);
        cls -> setSATmin(CClusterizer::getSATmin());


        std::vector<CCParticle> SA_local_pass2 = cls -> makeSA2(local_pass);
        for (auto &p_loc : SA_local_pass2){
            if (static_cast<short>(p_loc.getType()) > 2){
                max_clust_id++;
                p_loc.setClusterID(max_clust_id);
                local_pass.push_back(p_loc);
                auto childs_loc = p_loc.getChilds();
                double E_BND = p_loc.getBindingEnergy();
                if(E_BND < 0){
                    for (auto &c_loc : childs_loc) c_loc.setClusterID(max_clust_id);
                    final_particles.push_back(p_loc);
                }
            }
        }
        delete(cls);
    }
    else{
        for (auto &p : free)           final_particles.push_back(p);
        for (auto &p : bound_stable)   final_particles.push_back(p);
    }


    return final_particles;
}

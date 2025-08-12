#include "libccl.h"

#include <algorithm>
#include <unordered_set>

void CClusterizer::updateStableVector(const std::vector<CCParticle>& input,
                                      const float step_time,
                                      std::vector<CCParticle>& output) {
    if (input.empty()) return;

    std::vector<CCParticle>  input_copy = input;
    const size_t N = input_copy.size();

    // For each input particle
    for (auto& p : input_copy) {

        if (static_cast<short>(p.getType()) < 3) continue; // Select only clusters
        // And select only clusters with Ebind < less than defined cut
        if(tr_UseEbind && p.getBindingEnergy() > tr_EbindCut) continue;

        auto childsPtrs = p.getChildsPtrs();               // Constituent particles of current cluster

        // Small check for the particle collision time larger than current time step
        // Maybe this part is not needed -- left for a while
        bool select = true;
        for(const auto &c: childsPtrs){
            if (c -> getCollisionTime() > step_time){
                select = false;
                break;
            }
        }
        if (!select) continue;

        // If there are no stored 'stable' clusters -- store first
        if(output.size() == 0) {
            output.push_back(p);
        }
        // There are stored 'stable' clusters
        else{
            bool is_new = true;
            std::vector<size_t> overlapping;
            size_t idx = 0;

            // Loop over stored clusters
            for (auto& o : output) {
                auto StoredClusterChildsPtrs = o.getChildsPtrs();
                std::unordered_set<long> stored_set;

                // Constituents of the stored cluster are collected
                for (auto* sc : StoredClusterChildsPtrs) {
                    stored_set.insert(sc -> getUniqueID());
                }
    
                // Overlap between current cluster constituents
                // and stored cluster constituents found
                bool overlap = false;
                for (auto* cc : childsPtrs) {
                    if (stored_set.count(cc -> getUniqueID())) {
                        overlap = true;
                        break;
                    }
                }
                // Current cluster is not new
                // There is possibility that several clusters with same constituens as in current were stored already
                // Store indeces of previously stored clusters
                if (overlap) {
                    is_new = false;
                    overlapping.push_back(idx);
                }
                ++idx;
            }

            // If current cluster is not new
            if (!is_new) {
                size_t max_size = 0;
                // Maximal size of the previously stored clusters
                for (auto i : overlapping) {
                    max_size = std::max(max_size, output[i].getChildsPtrs().size());
                }
                bool allow_reduce = false;

                // Loop over only these previously stored clusters
                for (auto i : overlapping) {
                    auto StoredClusterChildsPtrs = output[i].getChildsPtrs();
                    // Collect collisions times of the stored clusters constituents into the map: [unique ID  -- collision time]
                    std::unordered_map<long, double> stored_times;
                    for (auto* sc : StoredClusterChildsPtrs) {
                        stored_times[sc->getUniqueID()] = sc -> getCollisionTime();
                    }

                    // For the each constituent of the current cluster
                    for (auto* c : childsPtrs) {
                        auto uid = c -> getUniqueID();
                        // Check if collision time for the same particle by UID increased
                        if (stored_times.count(uid) && c -> getCollisionTime() > stored_times[uid]) {
                            allow_reduce = true;
                            break;
                        }
                    }
                    if (allow_reduce) break;
                }
                // If the collision time did not increased allow the stored cluster to became bigger
                // Allow stored cluster to became smaller only if the collision time increased 
                if (childsPtrs.size() > max_size || (childsPtrs.size() <= max_size && allow_reduce)) {
                    std::sort(overlapping.begin(), overlapping.end(), std::greater<size_t>());
                    for (auto i : overlapping) {
                        output.erase(output.begin() + i);
                    }
                    output.push_back(p);
                }
            }
            // If cluster is new -- just store it 
            else {
                output.push_back(p);
            }
        }
    }
}


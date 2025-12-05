#include "libccl.h"

#include <algorithm>
#include <unordered_set>

void CClusterizer::updateStableVector(const std::vector<CCParticle>& input,
                                      const float step_time,
                                      std::vector<CCParticle>& output) {
    if (input.empty()) return;

    std::vector<CCParticle> input_copy = input;
    const size_t N = input_copy.size();

    // Collect last collision times of all input baryons: bound and free
    std::unordered_map<int, double> latest_collision_time;
    for (auto& p : input_copy) {
        if (p.getType() > ParticleType::LambdaSigma) {
            for (auto* c : p.getChildsPtrs()) {
                latest_collision_time[c -> getUniqueID()] = c -> getCollisionTime();
            }
        }
        else {
            latest_collision_time[p.getUniqueID()] = p.getCollisionTime();
        }
    }

    // Filter the stored (stable candidates) clusters using last collision time information
    for (size_t i = 0; i < output.size();) {
        auto& stored_cluster = output[i];
        bool invalid = false;

        for (auto* c : stored_cluster.getChildsPtrs()) {
            auto it = latest_collision_time.find(c -> getUniqueID());
            if (it == latest_collision_time.end()) {
                // Baryon disappeared, stored cluster was not stable -- remove stored cluster
                invalid = true;
                break;
            }
            if (c -> getCollisionTime() < it -> second) {
                // Baryon had collision after stored cluster was foun -- remove stored cluster
                invalid = true;
                break;
            }
        }
        // Removal
        if (invalid) {
            output[i] = std::move(output.back());
            output.pop_back();
        }
        else {
            ++i;
        }
    }


    // For each input particle
    for (auto& p : input_copy) {

        if (static_cast<short>(p.getType()) < 3) continue; // Select only clusters
        // And select only clusters with Ebind < less than defined cut
        if(tr_UseEbind && p.getBindingEnergy() > tr_EbindCut) continue;

        // Constituent particles of current cluster
        auto childsPtrs = p.getChildsPtrs();

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

                // Allow stored cluster to became bigger
                if (childsPtrs.size() > max_size) {
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


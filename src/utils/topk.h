#pragma once

#include "spheni/spheni.h"
#include <queue>
#include <vector>

namespace spheni {

class TopK {
public:
    TopK(int k);
    
    void push(long long id, float score);
    std::vector<SearchHit> sorted_results();
    
private:
    int k_;

    struct CompareHit {
        bool operator()(const SearchHit& a, const SearchHit& b) const {
            return a.score > b.score;
            // smaller score has higher priority
        }
    };
    
    std::priority_queue<SearchHit, std::vector<SearchHit>, CompareHit> heap_;
};

}

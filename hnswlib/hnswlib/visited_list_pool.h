#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
 public:
    vl_type curV;
    vl_type curV_ft;
    // vl_type curVtop;
    vl_type *mass;
    // vl_type *top_mass;
    unsigned int numelements;
    // unsigned int top_numelements;

    VisitedList(int numelements1) {
        curV = -1;
        curV_ft = 1;
        // curVtop = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
        // top_mass = nullptr;
    }

    // VisitedList(int numelements1, int top_numelements_) {
    //     curV = -1;
    //     curVtop = -1;
    //     numelements = numelements1;
    //     top_numelements = top_numelements_;
    //     mass = new vl_type[numelements];
    //     top_mass = new vl_type[top_numelements];
    // }

    void reset() {
        curV++;
        // curVtop++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            // if(top_mass) memset(top_mass, 0, sizeof(vl_type) * (top_numelements));
            curV++;
        }
        curV_ft = curV + 1;
        if (curV_ft == 0) {
            curV_ft++;
        }
    }

    ~VisitedList() { delete[] mass;}
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;
    // int top_numelements;
    bool is_two_level = false;

 public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    // VisitedListPool(int initmaxpools, int numelements1, int top_numelements_) {
    //     is_two_level = true;
    //     numelements = numelements1;
    //     top_numelements = top_numelements_;
    //     for (int i = 0; i < initmaxpools; i++)
    //         pool.push_front(new VisitedList(numelements, top_numelements));
    // }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                if (is_two_level) {
                    // rez = new VisitedList(numelements, top_numelements);
                    std::cout << "Error: two-level visited list is not supported now." << std::endl;
                    exit(-1);
                } else {
                    rez = new VisitedList(numelements);
                }
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib

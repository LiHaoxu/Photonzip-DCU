#pragma once

// [PhotonZip] Reusable device scratch memory for the MANS DCU backend.
//
// Every compress/decompress call used to hipMalloc and hipFree about a dozen buffers, which
// dominated the cost of small calls (HDF5 chunks). A WorkspaceLease hands out a set of
// grow-only buffers ("slots") from a pool per (pool id, device). Concurrent host threads get
// different workspaces, and a workspace is only returned to the pool after the call has
// synchronised, so reuse is safe. Buffers are kept for the lifetime of the process (freeing
// them from static destructors could run after the HIP runtime has been torn down).

#include <hip/hip_runtime.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mans::dcu {

enum WorkspacePool : int { kMansPool = 0, kAdmPool = 1, kAnsPool = 2, kMansHostPool = 3 };

class Workspace {
public:
    void* get(std::size_t slot, std::size_t bytes) {
        if (slot >= slots_.size()) slots_.resize(slot + 1);
        Slot& s = slots_[slot];
        if (bytes == 0) bytes = 1;
        if (bytes > s.bytes) {
            if (s.ptr) (void)hipFree(s.ptr);
            s.ptr = nullptr;
            s.bytes = 0;
            const hipError_t status = hipMalloc(&s.ptr, bytes);
            if (status != hipSuccess) {
                throw std::runtime_error(std::string("hipMalloc MANS DCU workspace: ") + hipGetErrorString(status));
            }
            s.bytes = bytes;
        }
        return s.ptr;
    }

    template <typename T>
    T* get(std::size_t slot, std::size_t count) {
        return static_cast<T*>(get(slot, count * sizeof(T)));
    }

private:
    struct Slot {
        void* ptr = nullptr;
        std::size_t bytes = 0;
    };
    std::vector<Slot> slots_;
};

class WorkspaceLease {
public:
    explicit WorkspaceLease(WorkspacePool pool) {
        int device = 0;
        (void)hipGetDevice(&device);
        key_ = {static_cast<int>(pool), device};
        std::lock_guard<std::mutex> lock(mutex());
        auto& free_list = pools()[key_];
        if (free_list.empty()) {
            ws_ = new Workspace();  // intentionally never deleted, see above
        } else {
            ws_ = free_list.back();
            free_list.pop_back();
        }
    }
    ~WorkspaceLease() {
        std::lock_guard<std::mutex> lock(mutex());
        pools()[key_].push_back(ws_);
    }
    WorkspaceLease(const WorkspaceLease&) = delete;
    WorkspaceLease& operator=(const WorkspaceLease&) = delete;

    Workspace* operator->() const { return ws_; }

private:
    static std::mutex& mutex() {
        static std::mutex m;
        return m;
    }
    static std::map<std::pair<int, int>, std::vector<Workspace*>>& pools() {
        static auto* p = new std::map<std::pair<int, int>, std::vector<Workspace*>>();
        return *p;
    }

    std::pair<int, int> key_;
    Workspace* ws_ = nullptr;
};

}  // namespace mans::dcu

#pragma once

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace hnswlib {

// Dynamic point scheduling, shared by insertion and synchronous maintenance.
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t num_threads, Function fn) {
    if (end <= start) return;
    if (num_threads == 0)
        num_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    num_threads = std::min(num_threads, end - start);
    if (num_threads == 1) {
        for (size_t id = start; id < end; ++id) fn(id, 0);
        return;
    }
    std::atomic<size_t> next(start);
    std::atomic<bool> failed(false);
    std::exception_ptr error;
    std::mutex error_mutex;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    try {
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads.emplace_back([&, thread_id] {
                while (!failed.load(std::memory_order_relaxed)) {
                    size_t id = next.fetch_add(1, std::memory_order_relaxed);
                    if (id >= end) break;
                    try {
                        fn(id, thread_id);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!error) error = std::current_exception();
                        failed.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace hnswlib

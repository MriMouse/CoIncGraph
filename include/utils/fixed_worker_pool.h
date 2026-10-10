#ifndef SEPGRAPH_FIXED_WORKER_POOL_H
#define SEPGRAPH_FIXED_WORKER_POOL_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace sepgraph {
namespace concurrency {

class FixedWorkerPool {
public:
    explicit FixedWorkerPool(size_t worker_count)
        : next_(0), count_(0), generation_(0), completed_(0), stopping_(false) {
        workers_.reserve(worker_count);
        for (size_t worker = 0; worker < worker_count; ++worker) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    FixedWorkerPool(const FixedWorkerPool &) = delete;
    FixedWorkerPool &operator=(const FixedWorkerPool &) = delete;

    ~FixedWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_ready_.notify_all();
        for (auto &worker : workers_) worker.join();
    }

    size_t WorkerCount() const { return workers_.size(); }

    template <typename Function>
    void Run(size_t count, Function function, size_t grain = 16) {
        if (count == 0) return;
        if (workers_.empty()) {
            for (size_t index = 0; index < count; ++index) function(index);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            function_ = function;
            next_.store(0, std::memory_order_relaxed);
            count_ = count;
            grain_ = std::max<size_t>(1, grain);
            completed_ = 0;
            error_ = nullptr;
            ++generation_;
        }
        work_ready_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        work_done_.wait(lock, [this]() {
            return completed_ == workers_.size();
        });
        function_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:

    void WorkerLoop() {
        size_t observed_generation = 0;
        while (true) {
            std::function<void(size_t)> function;
            size_t count = 0;
            size_t grain = 16;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [this, observed_generation]() {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) return;
                observed_generation = generation_;
                function = function_;
                count = count_;
                grain = grain_;
            }
            while (true) {
                const size_t begin = next_.fetch_add(grain,
                                                     std::memory_order_relaxed);
                if (begin >= count) break;
                const size_t end = std::min(begin + grain, count);
                for (size_t index = begin; index < end; ++index) {
                    try {
                        function(index);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (!error_) error_ = std::current_exception();
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++completed_;
                if (completed_ == workers_.size()) work_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::atomic<size_t> next_;
    size_t count_;
    size_t grain_ = 16;
    size_t generation_;
    size_t completed_;
    bool stopping_;
    std::function<void(size_t)> function_;
    std::exception_ptr error_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
};

} // namespace concurrency
} // namespace sepgraph

#endif

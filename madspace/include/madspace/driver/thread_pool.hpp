#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace madspace {

class ThreadPool {
public:
    using JobFunc = std::function<std::optional<std::size_t>()>;
    ThreadPool(int thread_count = -1);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void set_thread_count(int new_count);
    std::size_t thread_count() const { return _thread_count; }
    void submit(JobFunc job);
    void submit(std::vector<JobFunc>& jobs);
    std::optional<std::size_t> wait();
    std::vector<std::size_t> wait_multiple();
    std::size_t add_listener(std::function<void(std::size_t)> listener);
    void remove_listener(std::size_t id);

    static std::size_t thread_index() { return _thread_index; }

private:
    static inline thread_local std::size_t _thread_index = 0;
    static const std::size_t QUEUE_SIZE_PER_THREAD = 16384;

    void thread_loop(std::size_t index);
    bool fill_done_cache();
    // Called with *lock* held and _exception set: cancels the queue, waits for
    // the running jobs and rethrows on the caller's thread.
    [[noreturn]] void rethrow_job_exception(std::unique_lock<std::mutex>& lock);

    std::mutex _mutex;
    std::condition_variable _cv_run, _cv_done;
    std::size_t _thread_count;
    std::vector<std::thread> _threads;
    std::deque<JobFunc> _job_queue;
    std::deque<std::size_t> _done_queue;
    std::vector<std::size_t> _done_buffer;
    std::size_t _busy_threads = 0;
    // The first exception thrown by a job, rethrown by wait()/wait_multiple().
    std::exception_ptr _exception;
    std::size_t _listener_id = 0;
    std::unordered_map<std::size_t, std::function<void(std::size_t)>> _listeners;
};

// Collects the results of jobs run on a ThreadPool by a caller that counts its
// jobs in flight and waits for exactly that many results. Every job submitted
// through submit() posts exactly one result -- its id, or the exception it threw
// -- so such a caller can never block on a job that died.
class ResultQueue {
public:
    struct Result {
        std::size_t id;
        // Set if the job threw instead of completing
        std::exception_ptr exception;
    };

    // Runs *job* on *pool*, then posts *id*, or the exception *job* threw.
    void submit(ThreadPool& pool, std::size_t id, std::function<void()> job);
    void push(std::size_t id);
    void push_exception(std::size_t id, std::exception_ptr exception);
    // Blocks until a result is available, calling *poll* (e.g. a check for a
    // requested abort, which may throw) every *poll_interval* while waiting.
    Result wait(
        const std::function<void()>& poll = {},
        std::chrono::milliseconds poll_interval = std::chrono::milliseconds(100)
    );
    // Cancels the jobs submitted but not started yet, waits for *count* further
    // results and drops them, exceptions included. A running job checks
    // cancelled() between its steps to stop early. The wait cannot be cut short
    // -- the jobs still write into their caller's state -- but *poll* keeps
    // being called, and the first exception it throws (an abort requested
    // during the drain) is returned instead of being lost.
    std::exception_ptr discard(
        std::size_t count,
        const std::function<void()>& poll = {},
        std::chrono::milliseconds poll_interval = std::chrono::milliseconds(100)
    );
    bool cancelled() const { return _cancelled; }

private:
    std::atomic<bool> _cancelled = false;
    std::mutex _mutex;
    std::condition_variable _cv;
    std::deque<Result> _queue;
    std::vector<Result> _buffer;
};

template <typename T>
class ThreadResource {
public:
    ThreadResource() = default;
    ThreadResource(
        ThreadPool& pool,
        std::function<T()> constructor,
        std::optional<std::function<void(T&)>> destructor = std::nullopt,
        bool lazy = true
    ) :
        _pool(&pool),
        _constructor(std::move(constructor)),
        _destructor(destructor),
        _listener_id(pool.add_listener([this, lazy](std::size_t thread_count) {
            while (_resources.size() < thread_count) {
                _resources.emplace_back();
                if (!lazy) {
                    construct(_resources.back());
                }
            }
        })) {
        for (std::size_t i = 0; i == 0 || i < pool.thread_count(); ++i) {
            _resources.emplace_back();
            if (!lazy || i == 0) {
                construct(_resources.back());
            }
        }
    }
    ~ThreadResource() { reset(); }
    ThreadResource(ThreadResource&& other) noexcept :
        _pool(std::move(other._pool)),
        _resources(std::move(other._resources)),
        _constructor(std::move(other._constructor)),
        _listener_id(std::move(other._listener_id)),
        _destructor(std::move(other._destructor)) {
        other._pool = nullptr;
    }

    ThreadResource& operator=(ThreadResource&& other) noexcept {
        reset();
        _pool = std::move(other._pool);
        _resources = std::move(other._resources);
        _constructor = std::move(other._constructor);
        _listener_id = std::move(other._listener_id);
        _destructor = std::move(other._destructor);
        other._pool = nullptr;
        return *this;
    }
    ThreadResource(const ThreadResource&) = delete;
    ThreadResource& operator=(const ThreadResource&) = delete;
    T& get() { return construct(_resources.at(ThreadPool::thread_index())); }
    const T& get() const {
        return construct(_resources.at(ThreadPool::thread_index()));
    }
    void reset() {
        if (_pool) {
            if (_destructor) {
                for (auto& [flag, item] : _resources) {
                    if (item) {
                        _destructor.value()(*item);
                    }
                }
            }
            _pool->remove_listener(_listener_id);
        }
    }

private:
    T& construct(std::pair<std::once_flag, std::optional<T>>& slot) const {
        auto& [flag, item] = slot;
        std::call_once(flag, [&] {
            std::unique_lock<std::mutex> lock(construction_mutex());
            item.emplace(_constructor());
        });
        return *item;
    }
    static std::mutex& construction_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    ThreadPool* _pool = nullptr;
    mutable std::deque<std::pair<std::once_flag, std::optional<T>>> _resources;
    std::function<T()> _constructor;
    std::size_t _listener_id;
    std::optional<std::function<void(T&)>> _destructor;
};

inline ThreadPool& default_thread_pool() {
    static ThreadPool instance;
    return instance;
}

} // namespace madspace

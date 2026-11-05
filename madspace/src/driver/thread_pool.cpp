#include "madspace/driver/thread_pool.hpp"

#include <stdexcept>

#include "madspace/util.hpp"

using namespace madspace;

ThreadPool::ThreadPool(int thread_count) { set_thread_count(thread_count); }

ThreadPool::~ThreadPool() { set_thread_count(0); }

void ThreadPool::set_thread_count(int new_count) {
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _thread_count = new_count < 0 ? std::thread::hardware_concurrency() : new_count;
    }

    if (_threads.size() < _thread_count) {
        for (int i = _threads.size(); i < _thread_count; ++i) {
            _threads.emplace_back(&ThreadPool::thread_loop, this, i);
        }
    } else if (_threads.size() > _thread_count) {
        _cv_run.notify_all();
        std::for_each(
            _threads.begin() + _thread_count, _threads.end(), [](auto& thread) {
                thread.join();
            }
        );
        _threads.erase(_threads.begin() + _thread_count, _threads.end());
    }

    for (auto& [id, listener] : _listeners) {
        listener(_thread_count);
    }
}

void ThreadPool::submit(JobFunc job) {
    std::unique_lock<std::mutex> lock(_mutex);
    _job_queue.push_back(job);
    if (!_done_queue.empty()) {
        _done_buffer.insert(
            _done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend()
        );
        _done_queue.clear();
    }
    lock.unlock();
    _cv_run.notify_one();
}

void ThreadPool::submit(std::vector<JobFunc>& jobs) {
    if (jobs.empty()) {
        return;
    }
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto& job : jobs) {
        _job_queue.push_back(std::move(job));
    }
    if (!_done_queue.empty()) {
        _done_buffer.insert(
            _done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend()
        );
        _done_queue.clear();
    }
    lock.unlock();
    _cv_run.notify_all();
}

void ThreadPool::rethrow_job_exception(std::unique_lock<std::mutex>& lock) {
    _job_queue.clear();
    _cv_done.wait(lock, [&] { return _busy_threads == 0; });
    auto exception = _exception;
    _exception = nullptr;
    _done_queue.clear();
    _done_buffer.clear();
    std::rethrow_exception(exception);
}

bool ThreadPool::fill_done_cache() {
    if (!_done_buffer.empty()) {
        return true;
    }

    std::unique_lock<std::mutex> lock(_mutex);
    if (_exception) {
        rethrow_job_exception(lock);
    }
    if (_done_queue.empty()) {
        if (_job_queue.empty() && _busy_threads == 0) {
            return false;
        }
        _cv_done.wait(lock, [&] { return !_done_queue.empty() || _exception; });
        if (_exception) {
            rethrow_job_exception(lock);
        }
    }
    _done_buffer.insert(_done_buffer.begin(), _done_queue.rbegin(), _done_queue.rend());
    _done_queue.clear();
    return true;
}

std::optional<std::size_t> ThreadPool::wait() {
    if (!fill_done_cache()) {
        return std::nullopt;
    }
    std::size_t result = _done_buffer.back();
    _done_buffer.pop_back();
    return result;
}

std::vector<std::size_t> ThreadPool::wait_multiple() {
    if (!fill_done_cache()) {
        return {};
    }
    std::vector<std::size_t> ret(_done_buffer.rbegin(), _done_buffer.rend());
    _done_buffer.clear();
    return ret;
}

std::size_t ThreadPool::add_listener(std::function<void(std::size_t)> listener) {
    _listeners[_listener_id] = listener;
    return _listener_id++;
}

void ThreadPool::remove_listener(std::size_t id) {
    if (auto search = _listeners.find(id); search != _listeners.end()) {
        _listeners.erase(search);
    } else {
        throw std::invalid_argument("Listener id not found");
    }
}

void ThreadPool::thread_loop(std::size_t index) {
    _thread_index = index;
    std::unique_lock<std::mutex> lock(_mutex);
    while (true) {
        _cv_run.wait(lock, [&] {
            return !_job_queue.empty() || index >= _thread_count;
        });
        if (index >= _thread_count) {
            return;
        }
        auto job = _job_queue.front();
        _job_queue.pop_front();
        ++_busy_threads;
        lock.unlock();
        std::optional<std::size_t> result;
        try {
            result = job();
        } catch (...) {
            // Keep the first exception for the waiter and drop the queued jobs,
            // which capture the waiter's locals.
            lock.lock();
            if (!_exception) {
                _exception = std::current_exception();
            }
            _job_queue.clear();
            --_busy_threads;
            _cv_done.notify_all();
            continue;
        }
        lock.lock();
        --_busy_threads;
        if (result) {
            _done_queue.push_back(*result);
        }
        // notify_all, not notify_one: rethrow_job_exception() waits here for
        // _busy_threads to drain, and a job returning nullopt must wake it too.
        _cv_done.notify_all();
    }
}

void ResultQueue::submit(ThreadPool& pool, std::size_t id, std::function<void()> job) {
    pool.submit([this, id, job = std::move(job)]() -> std::optional<std::size_t> {
        try {
            if (_cancelled) {
                throw std::runtime_error("job cancelled");
            }
            job();
        } catch (...) {
            push_exception(id, std::current_exception());
            return std::nullopt;
        }
        push(id);
        return std::nullopt;
    });
}

void ResultQueue::push(std::size_t id) {
    std::unique_lock<std::mutex> lock(_mutex);
    _queue.push_back({id, nullptr});
    _cv.notify_one();
}

void ResultQueue::push_exception(std::size_t id, std::exception_ptr exception) {
    std::unique_lock<std::mutex> lock(_mutex);
    _queue.push_back({id, exception});
    _cv.notify_one();
}

ResultQueue::Result ResultQueue::wait(
    const std::function<void()>& poll, std::chrono::milliseconds poll_interval
) {
    // Once per call too, not only while blocked: results may keep coming.
    if (poll) {
        poll();
    }
    if (_buffer.empty()) {
        std::unique_lock<std::mutex> lock(_mutex);
        auto ready = [&] { return !_queue.empty(); };
        if (!poll) {
            _cv.wait(lock, ready);
        }
        while (!_cv.wait_for(lock, poll_interval, ready)) {
            // Without the lock: poll may throw, or take its time.
            lock.unlock();
            poll();
            lock.lock();
        }
        _buffer.insert(_buffer.begin(), _queue.rbegin(), _queue.rend());
        _queue.clear();
    }
    Result result = _buffer.back();
    _buffer.pop_back();
    return result;
}

std::exception_ptr ResultQueue::discard(
    std::size_t count,
    const std::function<void()>& poll,
    std::chrono::milliseconds poll_interval
) {
    _cancelled = true;
    std::exception_ptr aborted;
    // never throws: the drain has to run to completion
    std::function<void()> guarded_poll;
    if (poll) {
        guarded_poll = [&] {
            if (aborted) {
                return;
            }
            try {
                poll();
            } catch (...) {
                aborted = std::current_exception();
            }
        };
    }
    for (std::size_t i = 0; i < count; ++i) {
        wait(guarded_poll, poll_interval);
    }
    _cancelled = false;
    return aborted;
}

#pragma once

#include "madspace/driver/tensor.hpp"
#include "madspace/driver/thread_pool.hpp"
#include "simd.hpp"

namespace madspace {
namespace cpu {

class CpuDevice : public Device {
public:
    static constexpr bool is_concurrent = false;

    std::pair<void*, Tensor> allocate(std::size_t size, AllocHint hint) const override {
        std::pair<void*, Tensor> ret{new std::byte[size], Tensor()};
        if (needs_zero_init(hint)) {
            std::memset(ret.first, 0, size);
        }
        return ret;
    }

    void free(void* ptr) const override { delete[] static_cast<std::byte*>(ptr); }

    void memcpy(void* to, void* from, std::size_t size) const override {
        auto to_u8 = static_cast<std::byte*>(to);
        auto from_u8 = static_cast<std::byte*>(from);
        std::copy(from_u8, from_u8 + size, to_u8);
    }

    void tensor_copy(const Tensor& source, Tensor& target) const override;
    void tensor_zero(Tensor& tensor) const override;
    void tensor_add(const Tensor& source, Tensor& target) const override;
    void tensor_cpu(const Tensor& source, Tensor& target) const override {}
    DevicePtr device_ptr() const override { return &instance(); }
    DeviceType device_type() const override { return DeviceType::cpu; }
    void activate() const override {}
    void adam_step(
        const Tensor& gradient,
        Tensor& parameter,
        Tensor& exp_avg,
        Tensor& exp_avg_sq,
        double step_size,
        double beta1,
        double beta2,
        double eps,
        double bias_corr2_sqrt
    ) const override;

    template <typename F>
    void foreach (std::size_t batch_size, F func, bool single_job = false) const {
        func(batch_size, 0);
    }

    template <typename F>
    void submit(F func) const {
        func();
    }

    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(CpuDevice&) = delete;
    static const CpuDevice& instance() {
        static CpuDevice device;
        return device;
    }

protected:
    CpuDevice() = default;
};

class AsyncCpuDevice : public CpuDevice {
public:
    static constexpr bool is_concurrent = true;

    AsyncCpuDevice(
        std::size_t instr_index,
        std::size_t& instr_job_count,
        bool& barrier_state,
        std::vector<std::function<std::size_t()>>& funcs_after_barrier,
        ThreadPool& thread_pool
    ) :
        _instr_index(instr_index),
        _instr_job_count(instr_job_count),
        _barrier_state(barrier_state),
        _funcs_after_barrier(funcs_after_barrier),
        _thread_pool(thread_pool) {}

    void tensor_copy(const Tensor& source, Tensor& target) const override;
    void tensor_zero(Tensor& tensor) const override;
    void tensor_add(const Tensor& source, Tensor& target) const override;

    std::tuple<std::size_t, std::size_t>
    job_count_and_size(std::size_t batch_size, bool single_job = false) const {
        if (batch_size == 0) {
            return {0, 0};
        }
        // return {1, batch_size};
        if (single_job) {
            return {1, batch_size};
        }

        std::size_t batch_size_vec = (batch_size + simd_vec_size - 1) / simd_vec_size;
        std::size_t min_batch_size = 64 / simd_vec_size;
        std::size_t thread_count = _thread_pool.thread_count();
        // std::size_t job_count = (batch_size + min_batch_size - 1) / min_batch_size;
        std::size_t job_count = batch_size_vec < thread_count * min_batch_size
            ? (batch_size_vec + min_batch_size - 1) / min_batch_size
            : thread_count;
        std::size_t job_size =
            (batch_size_vec + job_count - 1) / job_count * simd_vec_size;
        // correct rounding errors to vector size
        job_count = (batch_size + job_size - 1) / job_size;
        return {job_count, job_size};
    }

    template <typename F>
    void foreach (std::size_t batch_size, F func, bool single_job = false) const {
        auto [job_count, job_size] = job_count_and_size(batch_size, single_job);
        std::size_t result = _instr_index;
        std::vector<ThreadPool::JobFunc> jobs;
        jobs.reserve(job_count);
        if (!_barrier_state) {
            _instr_job_count += job_count;
        }
        for (std::size_t i = 0; i < job_count; ++i) {
            std::size_t offset = i * job_size;
            std::size_t count = std::min(job_size, batch_size - offset);
            auto job_func = [count, offset, func, result]() mutable {
                func(count, offset);
                return result;
            };
            if (_barrier_state) {
                _funcs_after_barrier.push_back(job_func);
            } else {
                // _thread_pool.submit(job_func);
                jobs.push_back(job_func);
            }
        }
        if (!_barrier_state) {
            _thread_pool.submit(jobs);
        }
    }

    template <typename F>
    void submit(F func) const {
        std::size_t result = _instr_index;
        auto job_func = [func, result]() mutable {
            func();
            return result;
        };
        if (_barrier_state) {
            _funcs_after_barrier.push_back(job_func);
        } else {
            ++_instr_job_count;
            _thread_pool.submit(job_func);
        }
    }

    void sync_barrier() const override {
        if (_instr_job_count > 0) {
            _barrier_state = true;
        }
    }

private:
    int _instr_index;
    std::size_t& _instr_job_count;
    bool& _barrier_state;
    std::vector<std::function<std::size_t()>>& _funcs_after_barrier;
    ThreadPool& _thread_pool;
};

extern "C" int device_count();
extern "C" DevicePtr get_device(int index);

} // namespace cpu
} // namespace madspace

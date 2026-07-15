#include "api.h"

#include "CPPProcess.h"
#include "MemoryAccessMomenta.h"
#include <cmath>

using namespace mg5amcCpu;

namespace {

void* initialize_impl(
    const fptype* momenta,
    const fptype* couplings,
    fptype* matrix_elements,
    fptype* numerators,
    fptype* denominators,
    std::size_t count
) {
    bool is_good_hel[CPPProcess::ncomb];
    sigmaKin_getGoodHel(
        momenta, couplings, matrix_elements, numerators, denominators, is_good_hel, count
    );
    sigmaKin_setGoodHel(is_good_hel);
    return nullptr;
}

void initialize(
    const fptype* momenta,
    const fptype* couplings,
    fptype* matrix_elements,
    fptype* numerators,
    fptype* denominators,
    std::size_t count
) {
    // static local initialization is called exactly once in a thread-safe way
    static void* dummy = initialize_impl(
        momenta, couplings, matrix_elements, numerators, denominators, count
    );
}

void transpose_momenta(
    const double* momenta_in, fptype* momenta_out, std::size_t i_event, std::size_t stride
) {
    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    std::size_t i_page = i_event / page_size;
    std::size_t i_vector = i_event % page_size;

    for (std::size_t i_part = 0; i_part < CPPProcess::npar; ++i_part) {
        for(std::size_t i_mom = 0; i_mom < 4; ++i_mom) {
            momenta_out[
                i_page * CPPProcess::npar * 4 * page_size +
                i_part * 4 * page_size + i_mom * page_size + i_vector
            ] = momenta_in[
                stride * (CPPProcess::npar * i_mom + i_part) + i_event
            ];
        }
    }
}

}

extern "C" {

const SubProcessInfo* subprocess_info() {
    static SubProcessInfo info = {
        /* on_gpu          = */ false,
        /* particle_count  = */ CPPProcess::npar,
        /* diagram_count   = */ CPPProcess::ndiagrams,
        /* helicity_count  = */ CPPProcess::ncomb
    };
    return &info;
}

void* init_subprocess(const char* param_card_path) {
    CPPProcess process;
    process.initProc(param_card_path);
    // We don't actually need the CPPProcess instance for anything as it initializes a
    // global variable. So here we just return a boolean that is used to store whether
    // the good helicities are initialized
    return new bool(false);
}

void compute_matrix_element(
    void* subprocess,
    size_t count,
    size_t stride,
    const double* momenta_in,
    const int* flavor_in,
    double* m2_out
) {
    // need to round to round to double page size for some reason
    std::size_t page_size2 = 2 * MemoryAccessMomentaBase::neppM;
    std::size_t rounded_count = (count + page_size2 - 1) / page_size2 * page_size2;

    std::vector<fptype> momenta(rounded_count * CPPProcess::npar * 4);
    std::vector<fptype> couplings(
        rounded_count * mg5amcCpu::Parameters_sm_dependentCouplings::ndcoup * 2
    );
    // alpha s from the paramcard is discarded, so we just use 0.118 for now
    std::vector<fptype> g_s(rounded_count, 1.2177157847767195); // sqrt(4 pi alpha_s)
    std::vector<fptype> helicity_random(rounded_count, 0.5);
    std::vector<fptype> color_random(rounded_count, 0.5);
    std::vector<fptype> matrix_elements(rounded_count);
    std::vector<unsigned int> channel_index(rounded_count, 2);
    std::vector<fptype> numerators(rounded_count * CPPProcess::ndiagrams);
    std::vector<fptype> denominators(rounded_count);
    std::vector<int> helicity_index(rounded_count);
    std::vector<int> color_index(rounded_count);

    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        transpose_momenta(momenta_in, momenta.data(), i_event, stride);
    }
    computeDependentCouplings(
        g_s.data(), couplings.data(), rounded_count
    );

    bool& is_initialized = *static_cast<bool*>(subprocess);
    if (!is_initialized) {
        initialize(
            momenta.data(),
            couplings.data(),
            matrix_elements.data(),
            numerators.data(),
            denominators.data(),
            rounded_count
        );
        is_initialized = true;
    }

    sigmaKin(
        momenta.data(),
        couplings.data(),
        helicity_random.data(),
        color_random.data(),
        matrix_elements.data(),
        channel_index.data(),
        numerators.data(),
        denominators.data(),
        color_index.data(),
        helicity_index.data(),
        rounded_count
    );

    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        std::size_t i_page = i_event / page_size;
        std::size_t i_vector = i_event % page_size;

        double denominator = denominators[i_event];
        double factor = denominator / numerators[
            i_page * page_size * CPPProcess::ndiagrams +
            1 * page_size + i_vector
        ];
        m2_out[i_event] = factor * matrix_elements[i_event];
    }
}

void compute_matrix_element_multichannel(
    void* subprocess,
    size_t count,
    size_t stride,
    const double* momenta_in,
    const double* alpha_s_in,
    const double* random_in,
    const int* flavor_in,
    double* m2_out,
    double* amp2_out,
    int* diagram_out,
    int* color_out,
    int* helicity_out
) {
    // need to round to round to double page size for some reason
    std::size_t page_size2 = 2 * MemoryAccessMomentaBase::neppM;
    std::size_t rounded_count = (count + page_size2 - 1) / page_size2 * page_size2;

    std::vector<fptype> momenta(rounded_count * CPPProcess::npar * 4);
    std::vector<fptype> couplings(
        rounded_count * mg5amcCpu::Parameters_sm_dependentCouplings::ndcoup * 2
    );
    std::vector<fptype> g_s(rounded_count);
    std::vector<fptype> helicity_random(rounded_count);
    std::vector<fptype> color_random(rounded_count);
    std::vector<fptype> matrix_elements(rounded_count);
    std::vector<unsigned int> channel_index(rounded_count, 2);
    std::vector<fptype> numerators(rounded_count * CPPProcess::ndiagrams);
    std::vector<fptype> denominators(rounded_count);
    std::vector<int> helicity_index(rounded_count);
    std::vector<int> color_index(rounded_count);

    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        transpose_momenta(momenta_in, momenta.data(), i_event, stride);
        helicity_random[i_event] = random_in[i_event];
        color_random[i_event] = random_in[i_event + stride];
        g_s[i_event] = sqrt(4 * M_PI * alpha_s_in[i_event]);
    }
    computeDependentCouplings(
        g_s.data(), couplings.data(), rounded_count
    );

    bool& is_initialized = *static_cast<bool*>(subprocess);
    if (!is_initialized) {
        initialize(
            momenta.data(),
            couplings.data(),
            matrix_elements.data(),
            numerators.data(),
            denominators.data(),
            rounded_count
        );
        is_initialized = true;
    }

    sigmaKin(
        momenta.data(),
        couplings.data(),
        helicity_random.data(),
        color_random.data(),
        matrix_elements.data(),
        channel_index.data(),
        numerators.data(),
        denominators.data(),
        color_index.data(),
        helicity_index.data(),
        rounded_count
    );

    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    for (std::size_t i_event = 0; i_event < count; ++i_event) {
        std::size_t i_page = i_event / page_size;
        std::size_t i_vector = i_event % page_size;

        double denominator = denominators[i_event];
        double factor = denominator / numerators[
            i_page * page_size * CPPProcess::ndiagrams +
            1 * page_size + i_vector
        ];
        m2_out[i_event] = factor * matrix_elements[i_event];
        for (std::size_t i_diag = 0; i_diag < CPPProcess::ndiagrams; ++i_diag) {
            amp2_out[stride * i_diag + i_event] = numerators[
                i_page * page_size * CPPProcess::ndiagrams +
                i_diag * page_size + i_vector
            ] / denominator;
        }
        diagram_out[i_event] = 0;
        color_out[i_event] = color_index[i_event] - 1;
        helicity_out[i_event] = helicity_index[i_event] - 1;
    }
}

void free_subprocess(void* subprocess) {
    delete static_cast<bool*>(subprocess);
}

}

#include "CPPProcess.h"
#include "api.h"
#include <vector>

extern "C" {

UmamiStatus umami_get_meta(UmamiMetaKey meta_key, void* result) {
    switch (meta_key) {
    case UMAMI_META_DEVICE: {
        *static_cast<UmamiDevice*>(result) = UMAMI_DEVICE_CPU;
        break;
    } case UMAMI_META_PARTICLE_COUNT:
        *static_cast<int*>(result) = CPPProcess::nexternal;
        break;
    case UMAMI_META_DIAGRAM_COUNT:
        *static_cast<int*>(result) = CPPProcess::ndiagrams;
        break;
    case UMAMI_META_HELICITY_COUNT:
        *static_cast<int*>(result) = CPPProcess::ncomb;
        break;
    case UMAMI_META_COLOR_COUNT:
        return UMAMI_ERROR_UNSUPPORTED_META;
    default:
        return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_initialize(UmamiHandle* handle, char const* param_card_path) {
    CPPProcess* process = new CPPProcess(param_card_path);
    std::vector<double*>& momenta = process->getMomenta();
    for (int i = 0; i < CPPProcess::nexternal; ++i) momenta.push_back(new double[4]());
    *handle = process;
    return UMAMI_SUCCESS;
}


UmamiStatus umami_set_parameter(
    UmamiHandle handle,
    char const* name,
    double parameter_real,
    double parameter_imag
) {
    return UMAMI_ERROR_NOT_IMPLEMENTED;
}


UmamiStatus umami_get_parameter(
    UmamiHandle handle,
    char const* name,
    double* parameter_real,
    double* parameter_imag
) {
    return UMAMI_ERROR_NOT_IMPLEMENTED;
}

UmamiStatus umami_matrix_element(
    UmamiHandle handle,
    size_t count,
    size_t stride,
    size_t offset,
    size_t input_count,
    UmamiInputKey const* input_keys,
    void const* const* inputs,
    size_t output_count,
    UmamiOutputKey const* output_keys,
    void* const* outputs
) {
    const double* momenta_in = nullptr;
    const double* alpha_s_in = nullptr;
    const int* flavor_in = nullptr;
    const double* random_color_in = nullptr;
    const double* random_helicity_in = nullptr;
    const double* random_diagram_in = nullptr;

    for (std::size_t i = 0; i < input_count; ++i) {
        const void* input = inputs[i];
        switch (input_keys[i]) {
        case UMAMI_IN_MOMENTA:
            momenta_in = static_cast<const double*>(input);
            break;
        case UMAMI_IN_ALPHA_S:
            alpha_s_in = static_cast<const double*>(input);
            break;
        case UMAMI_IN_FLAVOR_INDEX:
            flavor_in = static_cast<const int*>(input);
            break;
        case UMAMI_IN_RANDOM_COLOR:
            random_color_in = static_cast<const double*>(input);
            break;
        case UMAMI_IN_RANDOM_HELICITY:
            random_helicity_in = static_cast<const double*>(input);
            break;
        case UMAMI_IN_RANDOM_DIAGRAM:
            random_diagram_in = static_cast<const double*>(input);
            break;
        default:
            return UMAMI_ERROR_UNSUPPORTED_INPUT;
        }
    }
    if (!momenta_in) return UMAMI_ERROR_MISSING_INPUT;

    double* m2_out = nullptr;
    double* amp2_out = nullptr;
    int* diagram_out = nullptr;
    int* color_out = nullptr;
    int* helicity_out = nullptr;
    for (std::size_t i = 0; i < output_count; ++i) {
        void* output = outputs[i];
        switch (output_keys[i]) {
        case UMAMI_OUT_MATRIX_ELEMENT:
            m2_out = static_cast<double*>(output);
            break;
        case UMAMI_OUT_DIAGRAM_AMP2:
            amp2_out = static_cast<double*>(output);
            break;
        case UMAMI_OUT_COLOR_INDEX:
            color_out = static_cast<int*>(output);
            break;
        case UMAMI_OUT_HELICITY_INDEX:
            helicity_out = static_cast<int*>(output);
            break;
        case UMAMI_OUT_DIAGRAM_INDEX:
            diagram_out = static_cast<int*>(output);
            break;
        default:
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
        }
    }

    CPPProcess* process = static_cast<CPPProcess*>(handle);

    std::vector<double*>& process_momenta = process->getMomenta();
    for (size_t i_batch = 0; i_batch < count; ++i_batch) {
        for (size_t i_part = 0; i_part < CPPProcess::nexternal; ++i_part) {
            for(size_t i_mom = 0; i_mom < 4; ++i_mom) {
                process_momenta[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::nexternal * i_mom + i_part) + i_batch];
            }
        }
        process->getParameters().aS = alpha_s_in ? alpha_s_in[i_batch] : 0.118;
        double m2 = process->sigmaKin(flavor_in[i_batch]);
        if (m2_out) m2_out[i_batch] = m2;
        if (color_out) color_out[i_batch] = 0;
        if (diagram_out) diagram_out[i_batch] = 0;
        if (helicity_out) helicity_out[i_batch] = 0;
        if (amp2_out) {
            double chan_total = 0.;
            const double* amp2 = process->getAmp2();
            for(size_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
                double amp2_item = amp2[i_amp];
                amp2_out[i_amp * stride + i_batch] = amp2_item;
                chan_total += amp2_item;
            }
            for(size_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
                amp2_out[i_amp * stride + i_batch] /= chan_total;
            }
        }
    }

    return UMAMI_SUCCESS;
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
    int* helicity_out,
    void* cuda_stream
) {
    CPPProcess* process = static_cast<CPPProcess*>(subprocess);

    std::vector<double*>& process_momenta = process->getMomenta();
    for (size_t i_batch = 0; i_batch < count; ++i_batch) {
        for (size_t i_part = 0; i_part < CPPProcess::nexternal; ++i_part) {
            for(size_t i_mom = 0; i_mom < 4; ++i_mom) {
                process_momenta[i_part][i_mom] =
                    momenta_in[stride * (CPPProcess::nexternal * i_mom + i_part) + i_batch];
            }
        }
        process->getParameters().aS = alpha_s_in[i_batch];
        m2_out[i_batch] = process->sigmaKin(flavor_in[i_batch]);
        color_out[i_batch] = 0;
        diagram_out[i_batch] = 0;
        helicity_out[i_batch] = 0;
        double chan_total = 0.;
        const double* amp2 = process->getAmp2();
        for(size_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
            double amp2_item = amp2[i_amp];
            amp2_out[i_amp * stride + i_batch] = amp2_item;
            chan_total += amp2_item;
        }
        for(size_t i_amp = 0; i_amp < CPPProcess::ndiagrams; ++i_amp) {
            amp2_out[i_amp * stride + i_batch] /= chan_total;
        }
    }
}

void free_subprocess(void* subprocess) {
    CPPProcess* process = static_cast<CPPProcess*>(subprocess);
    std::vector<double*>& momenta = process->getMomenta();
    for (int i = 0; i < CPPProcess::nexternal; ++i) delete[] momenta[i];
    delete process;
}

UmamiStatus umami_free(UmamiHandle handle) {
    CPPProcess* process = static_cast<CPPProcess*>(handle);
    std::vector<double*>& momenta = process->getMomenta();
    for (int i = 0; i < CPPProcess::nexternal; ++i) delete[] momenta[i];
    delete process;
    return UMAMI_SUCCESS;
}

}

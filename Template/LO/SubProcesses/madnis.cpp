#include <array>
#include <vector>
#include <iostream>
#include <cmath>
#include <cfloat>

extern "C" {
    void configure_code_cc_(bool, bool, double&); 
    void madnis_api_(double*, int&, int&, bool&, double&);
    void get_momenta_(double*);
    void get_multichannel_(double*, int*);
}

// get weight
double Weight(double *rans, int ndim, int channel) {
    double result;
    bool cut = true;
    madnis_api_(rans, ndim, channel, cut, result);
    return result;
}

// momenta
std::vector<double> Momenta(int npart) {
    std::vector<double> momenta(npart*4);
    get_momenta_(momenta.data());
    return momenta;
}

// alpha
void GetAlpha(std::vector<double>* alpha, int* used_channel) {
    get_multichannel_(alpha->data(), used_channel);
    return;
}

extern "C" void madgraph_init() {
    double dconfig = 1.0;
    configure_code_cc_(true, true, dconfig);
}

extern "C" void call_madgraph(
    double* z, 
    int* chans, 
    int nbatch, 
    int ndim,
    int npart,
    int nchan, 
    double* w_out, 
    double* mom_out, 
    double* alpha_out,
    int* used_channels_out){
    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
        int channel = chans[ibatch];
        double weight = Weight(z + ibatch*ndim, ndim, channel + 1);

        // Get momenta
        auto mom = Momenta(npart);
        for(size_t imom = 0; imom < 4*npart; ++imom){
            mom_out[ibatch*4*npart + imom] = mom[imom];
        }

        // Get corresponding alpha and make sure its not NaN
        std::vector<double> alpha(nchan);
        int used_channel;
        GetAlpha(&alpha, &used_channel);
        for(size_t ialpha = 0; ialpha < nchan; ++ialpha){
            if (std::isnan(alpha[ialpha]) || weight == 0.0) {
                alpha_out[ibatch*nchan + ialpha] = 1.0 / nchan;
            } else {
                alpha_out[ibatch*nchan + ialpha] = alpha[ialpha];
            }
        }

        // Return correct weight and used_channel
        used_channels_out[ibatch] = used_channel - 1;
        w_out[ibatch] = weight/alpha_out[ibatch*nchan + used_channel - 1];

    }
}

#include <array>
#include <vector>
#include <iostream>
#include <cmath>
#include <cfloat>

extern "C" {
    void madnis_nlo_initialise_(); 
    void madnis_nlo_terminate_();
    void madnis_get_channel_(int*);
    void madnis_get_nchans_(int*);
    void madnis_set_channel_(int*, double*);
    void madnis_nlo_evaluate_(double*, double*, int*, double*);
}

// get integrands
double GetIntegrandNLO(double *rans) {
    double result[26]; // max number extracted from mind_module.f90
    int ifl = 0;
    double vegas_wgt = 1.0; // handle externally
    madnis_nlo_evaluate_(rans, &vegas_wgt, &ifl, result);
    // only return relevant part for now
    return result[1];
}

// returns the number of channels
// important to know the pool I can sample over
extern "C" int GetNChannels() {
    int nchan_out;
    madnis_get_nchans_(&nchan_out);
    return nchan_out;
}

// sets specific channel i,
// and needs jacobian of sampling as input
void SetChannel(int* ichan) {
    double jac = 1.0;
    madnis_set_channel_(ichan, &jac);
    return;
}

extern "C" void madgraph_nlo_init() {
    madnis_nlo_initialise_();
}

extern "C" void call_magraph_nlo(
    // random numbers and channel
    double* rand,
    //double* chans,
    // integer inputs
    int nbatch, // batch size
    int ndim, //dimension of the ps-space
    // outputs
    double* w_out, 
    int* used_chan_out){

    // Initialize the ichan array (or scalar as required by Fortran function)
    int ichan = 0;  // Adjust as needed (for example, if it's an array, initialize accordingly)
    
    // Call madnis_get_channel_ to fetch the channel state
    madnis_get_channel_(&ichan);
    
    // Print the channel state to check if it's initialized correctly
    std::cout << "ICHAN: " << ichan << std::endl;

    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
        // Set channel as done in input
        //int channel = chans[ibatch] + 1; // Fortran starts with 1
        //SetChannel(&channel);

        // get NLO weight
        double weight_nlo = GetIntegrandNLO(rand + ibatch*ndim);

        // Return correct weight and used_channel
        int used_channel = ichan;
        used_chan_out[ibatch] = used_channel - 1;
        w_out[ibatch] = weight_nlo;

    }
}

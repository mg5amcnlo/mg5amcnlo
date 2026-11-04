/* SPDX-License-Identifier: 0BSD */
/* Generated from the same records as metadata.json. */
#include "umami_provider.h"
#define UMAMI_HAS_ALPHA_S 0
#define UMAMI_HAS_COLOR_FLOW 1
#define UMAMI_ARTIFACT_ID "a74e458c6e7fa85d55e2e6ae494a2328a76fd69be74e40c867a35887e2fbd449"
static const double umami_masses[] = {0, 0, 91.188000000000002};
static const int umami_p0_pdgs[] = {1, -1, 23};
static const char *const umami_p0_helicities[] = {"h:-1,-1,-1", "h:-1,-1,+0", "h:-1,-1,+1", "h:-1,+1,-1", "h:-1,+1,+0", "h:-1,+1,+1", "h:+1,-1,-1", "h:+1,-1,+0", "h:+1,-1,+1", "h:+1,+1,-1", "h:+1,+1,+0", "h:+1,+1,+1"};
static const UmamiParameter umami_p0_parameters[] = {
    {"normalization.alpha_ew", 0.0075467711139788835, 0, true, false},
    {"normalization.alpha_s_me_check", 0.11799999999999999, 0, true, false},
    {"particle.6.mass", 173, 0, true, false},
    {"particle.6.width", 1.4915, 0, true, false},
    {"particle.23.mass", 91.188000000000002, 0, true, false},
    {"particle.23.width", 2.4414039999999999, 0, true, false},
    {"particle.24.mass", 80.419002445756163, 0, true, false},
    {"particle.24.width", 2.0476000000000001, 0, true, false},
    {"particle.25.mass", 125, 0, true, false},
    {"particle.25.width", 0.0063823389999999999, 0, true, false},
    {"coupling.10.1_23_1.component_0", -1.0244420275940371, 0, true, true},
    {"coupling.10.1_23_1.component_1", 0.17818666745287456, 0, true, true},
};
static const UmamiProcess umami_processes[] = {
    {"d_dbar_to_z", umami_p0_pdgs, NULL, "lc", 12, umami_p0_helicities, 12, umami_p0_parameters},
};
static const size_t umami_c0_f0_permutation[] = {0, 1, 2};
static const int umami_c0_f0_helicity_runtime[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const size_t umami_c0_f0_helicity_export[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const UmamiContribution umami_c0_f0_contributions[] = {
    {0, "flow:2,1", 0, umami_c0_f0_permutation, 1, umami_c0_f0_helicity_runtime, umami_c0_f0_helicity_export},
};
static const UmamiFlavour umami_c0_flavours[] = {
    {1, umami_c0_f0_contributions},
};
static const UmamiChannel umami_channels[] = {
    {1, umami_c0_flavours},
};
static const UmamiProvider umami_provider = {
    "p0", 3, 2, 12, 1, 12,
    umami_masses, 1, umami_processes,
    1, umami_channels, false
};

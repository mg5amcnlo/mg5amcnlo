int i,j;
%(flavor_mask_decl)s

double BWCUTOFF = 15;
// Calculate all wavefunctions
%(flavor_mask_setup)s
%(wavefunction_calls)s

// Calculate all amplitudes
%(amplitude_calls)s

double p[nexternal][4];
int i;
 
// Convert Pythia 4-vectors to double[]
for(i=0;i < nexternal;i++){
    p[i][0] = pME[i].e();
    p[i][1] = pME[i].px();
    p[i][2] = pME[i].py();
    p[i][3] = pME[i].pz();
}

// Calculate all wavefunctions
%(wavefunction_calls)s

// Calculate all amplitudes
%(amplitude_calls)s


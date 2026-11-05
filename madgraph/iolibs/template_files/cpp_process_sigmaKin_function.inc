// Local variables and constants
std::complex<double> **wfs;
// Helicities for the process
%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominator = %(den_factors)s;
// Flavor lookup table
%(flavor_table)s

const int* flavor = &flavor_table[flavor_id][0];

ntry[flavor_id]++;

// Define permutation
int perm[nexternal];
for(int i = 0; i < nexternal; i++){
    perm[i]=i;
}

double matrix_element = 0.;

if (sum_hel[flavor_id] == 0 || ntry[flavor_id] < 10){
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel ++){
        if (goodhel[flavor_id][ihel] || ntry[flavor_id] < 2){
            calculate_wavefunctions(perm, helicities[ihel], flavor);
            %(get_matrix_t_lines)s
            matrix_element += t;
            // Store which helicities give non-zero result
            if (t != 0. && !goodhel[flavor_id][ihel]){
                goodhel[flavor_id][ihel]=true;
                ngood[flavor_id] ++;
                igood[flavor_id][ngood[flavor_id]] = ihel;
            }
        }
    }
    jhel[flavor_id] = 0;
    sum_hel[flavor_id]=min(sum_hel[flavor_id], ngood[flavor_id]);
} else {
    // Only use the "good" helicities
    for(int j=0; j < sum_hel[flavor_id]; j++){
        jhel[flavor_id]++;
        if (jhel[flavor_id] >= ngood[flavor_id]) jhel[flavor_id]=0;
        double hwgt = double(ngood[flavor_id])/double(sum_hel[flavor_id]);
        int ihel = igood[flavor_id][jhel[flavor_id]];
        calculate_wavefunctions(perm, helicities[ihel], flavor);
        %(get_matrix_t_lines)s
        matrix_element += t*hwgt;
    }
}

return matrix_element * broken_sym(flavor) / denominator;

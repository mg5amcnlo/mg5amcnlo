import model SM_HeavyN_NLO
define mu = mu+ mu- vm vm~
define fermion = ve vm vt e- mu- ta- ve~ vm~ vt~ e+ mu+ ta+ j
generate p p > mu+ n1, n1 > mu- j j
output heavyNscan
launch
shower=pythia8
analysis=rivet
set nevents 200
set use_syst False
set mn1 scan:[10**(0.5*i+1) for i in range(0,4)]
set mn2 999999
set mn3 999999
set vmun1 scan:[10**(0-0.1*i) for i in range(0,3)]
set vmun2 0.
set vmun3 0.
set ven1 0.
set ven2 0.
set ven3 0.
set vtan1 0.
set vtan2 0.
set vtan3 0.
set wn1 0.01
set wn2 10
set wn3 10
set no_parton_cut
set fast_rivet
set run_contur True
set xaxis_var mn1
set yaxis_var vmun1
set yaxis_relvar vmun1*vmun1
set yaxis_label sqvmun1

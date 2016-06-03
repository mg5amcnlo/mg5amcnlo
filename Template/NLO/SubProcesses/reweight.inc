c Save reweight information for aMC@NLO computations

      include 'reweight0.inc'
      include 'reweight1.inc'

      logical doreweight
      common/c_doreweight/doreweight

c Check that save weights are consistent with the reference weight wgtref
      logical check_reweight
      parameter (check_reweight=.true.)

c Controls the scale/kinematics storage:
c  iwgtinfo=1 --> store muR and muF for both event and counterevents
c  iwgtinfo=2 --> store muR and muF for the kinematic configuration already
c                 in <event>, and the other kinematic configuration
c  iwgtinfo=3 --> store the two kinematic configurations
c  iwgtinfo=4 (default) --> same as 3, but also wgtbpower is included
c  iwgtinfo=5 (new default) --> needed when combining nFKSprocesses
c If computing parton-level NLO, iwgtinfo=2 must not be used
      integer iwgtinfo
      parameter (iwgtinfo=-5)

c
c Auxiliary quantities
c

c Temporary quantities relevant to degenerate terms
      double precision wgtdegrem_xi,wgtdegrem_lxi,wgtdegrem_muF
      common/cwgtaux1/wgtdegrem_xi,wgtdegrem_lxi,wgtdegrem_muF

c Temporary quantities relevant to n-body terms
      double precision wgtnstmp,wgtwnstmpmuf,wgtwnstmpmur,wgtref_nbody,wgtvirttmp,wgtnstmp_avgvirt
      common/cwgtaux2/wgtnstmp,wgtwnstmpmuf,wgtwnstmpmur,wgtref_nbody,wgtvirttmp,wgtnstmp_avgvirt

c Temporary weights for pure-NLO computations: event, counterevents+Born, 
c and Born, with the same conventions as in fks_singular
      double precision wgtNLO11,wgtNLO12,wgtNLO20
      common/cwgtaux3/wgtNLO11,wgtNLO12,wgtNLO20

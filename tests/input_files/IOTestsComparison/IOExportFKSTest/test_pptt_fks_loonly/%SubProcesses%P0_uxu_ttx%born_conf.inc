      INTEGER IFR
      INTEGER LMAXCONFIGSB_USED
      PARAMETER (LMAXCONFIGSB_USED=1)
      INTEGER MAX_BRANCHB_USED
      PARAMETER (MAX_BRANCHB_USED=1)
      INTEGER MAPCONFIG(0 : LMAXCONFIGSB_USED)
      INTEGER IFOREST(2, -MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER SPROP(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER TPRID(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      LOGICAL GFORCEBW(-MAX_BRANCHB_USED : -1, LMAXCONFIGSB_USED)

C     Here are the congifurations
C     Process: u~ u > t t~ [ LOonly = QED QCD ] QCD^2<=6 QED^2<=0
C     Process: c~ c > t t~ [ LOonly = QED QCD ] QCD^2<=6 QED^2<=0
C     Process: d~ d > t t~ [ LOonly = QED QCD ] QCD^2<=6 QED^2<=0
C     Process: s~ s > t t~ [ LOonly = QED QCD ] QCD^2<=6 QED^2<=0
C     Diagram 1, Amplitude 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP(  -1,   1)/      21/
C     Number of configs
      DATA MAPCONFIG(0)/   1/

C     Here are the BWs
      DATA GFORCEBW(-1,1)/.FALSE./

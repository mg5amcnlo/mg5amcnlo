      INTEGER IFR
      INTEGER LMAXCONFIGSB_USED
      PARAMETER (LMAXCONFIGSB_USED=2)
      INTEGER MAX_BRANCHB_USED
      PARAMETER (MAX_BRANCHB_USED=2)
      INTEGER MAPCONFIG(0 : LMAXCONFIGSB_USED)
      INTEGER IFOREST(2, -MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER SPROP(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER TPRID(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      LOGICAL GFORCEBW(-MAX_BRANCHB_USED : -1, LMAXCONFIGSB_USED)

C     Here are the congifurations
C     Process: a g > t t~ [ real = QED QCD ] QCD^2<=4 QED^2<=2
C     Diagram 1, Amplitude 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(IFR, -1,   1),IFR=1,2)/  1,  3/
      DATA TPRID(  -1,   1)/       6/
      DATA (IFOREST(IFR, -2,   1),IFR=1,2)/ -1,  4/
C     Diagram 2, Amplitude 2
      DATA MAPCONFIG(   2)/   2/
      DATA (IFOREST(IFR, -1,   2),IFR=1,2)/  1,  4/
      DATA TPRID(  -1,   2)/       6/
      DATA (IFOREST(IFR, -2,   2),IFR=1,2)/ -1,  3/
C     Number of configs
      DATA MAPCONFIG(0)/   2/

C     Here are the BWs

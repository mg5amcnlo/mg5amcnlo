      INTEGER IFR
      INTEGER LMAXCONFIGSB_USED
      PARAMETER (LMAXCONFIGSB_USED=1)
      INTEGER MAX_BRANCHB_USED
      PARAMETER (MAX_BRANCHB_USED=2)
      INTEGER MAPCONFIG(0 : LMAXCONFIGSB_USED)
      INTEGER IFOREST(2, -MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER SPROP(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER TPRID(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      LOGICAL GFORCEBW(-MAX_BRANCHB_USED : -1, LMAXCONFIGSB_USED)

C     Here are the congifurations
C     Process: t > b u d~ [ real = QED QCD ] QCD^2<=2 QED^2<=4
C     Process: t > b c s~ [ real = QED QCD ] QCD^2<=2 QED^2<=4
C     Diagram 1, Amplitude 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP(  -1,   1)/      24/
      DATA (IFOREST(IFR, -2,   1),IFR=1,2)/ -1,  2/
      DATA SPROP(  -2,   1)/       6/
C     Number of configs
      DATA MAPCONFIG(0)/   1/

C     Here are the BWs
      DATA GFORCEBW(-1,1)/.FALSE./
      DATA GFORCEBW(-2,1)/.FALSE./

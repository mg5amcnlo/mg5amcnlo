      INTEGER IFR
      INTEGER LMAXCONFIGSB_USED
      PARAMETER (LMAXCONFIGSB_USED=3)
      INTEGER MAX_BRANCHB_USED
      PARAMETER (MAX_BRANCHB_USED=1)
      INTEGER MAPCONFIG(0 : LMAXCONFIGSB_USED)
      INTEGER IFOREST(2, -MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER SPROP(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      INTEGER TPRID(-MAX_BRANCHB_USED:-1, LMAXCONFIGSB_USED)
      LOGICAL GFORCEBW(-MAX_BRANCHB_USED : -1, LMAXCONFIGSB_USED)

C     Here are the congifurations
C     Process: d d~ > t t~ [ real = QED QCD ] QCD^2<=4 QED^2<=2
C     Process: s s~ > t t~ [ real = QED QCD ] QCD^2<=4 QED^2<=2
C     Diagram 1, Amplitude 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP(  -1,   1)/      22/
C     Diagram 2, Amplitude 2
      DATA MAPCONFIG(   2)/   2/
      DATA (IFOREST(IFR, -1,   2),IFR=1,2)/  4,  3/
      DATA SPROP(  -1,   2)/      21/
C     Diagram 3, Amplitude 3
      DATA MAPCONFIG(   3)/   3/
      DATA (IFOREST(IFR, -1,   3),IFR=1,2)/  4,  3/
      DATA SPROP(  -1,   3)/      23/
C     Number of configs
      DATA MAPCONFIG(0)/   3/

C     Here are the BWs
      DATA GFORCEBW(-1,1)/.FALSE./
      DATA GFORCEBW(-1,2)/.FALSE./
      DATA GFORCEBW(-1,3)/.FALSE./

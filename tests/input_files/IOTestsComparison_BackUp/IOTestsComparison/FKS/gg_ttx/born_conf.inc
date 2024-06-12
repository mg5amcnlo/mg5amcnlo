C     Diagram 1, Amplitude 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(I, -1,   1),I=1,2)/  4,  3/
      DATA SPROP(  -1,   1)/      21/
C     Diagram 2, Amplitude 2
      DATA MAPCONFIG(   2)/   2/
      DATA (IFOREST(I, -1,   2),I=1,2)/  1,  3/
      DATA TPRID(  -1,   2)/       6/
      DATA (IFOREST(I, -2,   2),I=1,2)/ -1,  4/
C     Diagram 3, Amplitude 3
      DATA MAPCONFIG(   3)/   3/
      DATA (IFOREST(I, -1,   3),I=1,2)/  1,  4/
      DATA TPRID(  -1,   3)/       6/
      DATA (IFOREST(I, -2,   3),I=1,2)/ -1,  3/
C     Number of configs
      DATA MAPCONFIG(0)/   3/

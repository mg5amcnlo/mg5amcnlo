      INTEGER IFR,LMAXCONFIGS_USED,MAX_BRANCH_USED
      PARAMETER (LMAXCONFIGS_USED=  15)
      PARAMETER (MAX_BRANCH_USED =   3)
      INTEGER MAPCONFIG_D(  8,0:LMAXCONFIGS_USED)
      INTEGER IFOREST_D(  8,2,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
      INTEGER SPROP_D(  8,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
      INTEGER TPRID_D(  8,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
      DOUBLE PRECISION PMASS_D(  8,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED
     $ )
      DOUBLE PRECISION PWIDTH_D(  8,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USE
     $ D)
      INTEGER POW_D(  8,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
C     
C     nFKSprocess 1
C     Diagram 1 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   1)/   1/
      DATA (IFOREST_D(  1, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  1,  -1,   1)/      21/
      DATA (IFOREST_D(  1, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  1,  -2,   1)/      21/
C     Diagram 2 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   2)/   2/
      DATA (IFOREST_D(  1, IFR, -1,   2),IFR=1,2)/  5,  3/
      DATA SPROP_D(  1,  -1,   2)/       6/
      DATA (IFOREST_D(  1, IFR, -2,   2),IFR=1,2)/  4, -1/
      DATA SPROP_D(  1,  -2,   2)/      21/
C     Diagram 3 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   3)/   3/
      DATA (IFOREST_D(  1, IFR, -1,   3),IFR=1,2)/  5,  4/
      DATA SPROP_D(  1,  -1,   3)/      -6/
      DATA (IFOREST_D(  1, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA SPROP_D(  1,  -2,   3)/      21/
C     Diagram 4 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   4)/   4/
      DATA (IFOREST_D(  1, IFR, -1,   4),IFR=1,2)/  1,  3/
      DATA TPRID_D(  1,  -1,   4)/       6/
      DATA (IFOREST_D(  1, IFR, -2,   4),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  1,  -2,   4)/       6/
      DATA (IFOREST_D(  1, IFR, -3,   4),IFR=1,2)/ -2,  4/
C     Diagram 5 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   5)/   5/
      DATA (IFOREST_D(  1, IFR, -1,   5),IFR=1,2)/  1,  3/
      DATA TPRID_D(  1,  -1,   5)/       6/
      DATA (IFOREST_D(  1, IFR, -2,   5),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  1,  -2,   5)/      21/
      DATA (IFOREST_D(  1, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Diagram 6 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   6)/   6/
      DATA (IFOREST_D(  1, IFR, -1,   6),IFR=1,2)/  5,  4/
      DATA SPROP_D(  1,  -1,   6)/      -6/
      DATA (IFOREST_D(  1, IFR, -2,   6),IFR=1,2)/  1,  3/
      DATA TPRID_D(  1,  -2,   6)/       6/
      DATA (IFOREST_D(  1, IFR, -3,   6),IFR=1,2)/ -2, -1/
C     Diagram 7 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   7)/   7/
      DATA (IFOREST_D(  1, IFR, -1,   7),IFR=1,2)/  1,  4/
      DATA TPRID_D(  1,  -1,   7)/       6/
      DATA (IFOREST_D(  1, IFR, -2,   7),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  1,  -2,   7)/       6/
      DATA (IFOREST_D(  1, IFR, -3,   7),IFR=1,2)/ -2,  3/
C     Diagram 8 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   8)/   8/
      DATA (IFOREST_D(  1, IFR, -1,   8),IFR=1,2)/  1,  4/
      DATA TPRID_D(  1,  -1,   8)/       6/
      DATA (IFOREST_D(  1, IFR, -2,   8),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  1,  -2,   8)/      21/
      DATA (IFOREST_D(  1, IFR, -3,   8),IFR=1,2)/ -2,  5/
C     Diagram 9 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   9)/   9/
      DATA (IFOREST_D(  1, IFR, -1,   9),IFR=1,2)/  5,  3/
      DATA SPROP_D(  1,  -1,   9)/       6/
      DATA (IFOREST_D(  1, IFR, -2,   9),IFR=1,2)/  1,  4/
      DATA TPRID_D(  1,  -2,   9)/       6/
      DATA (IFOREST_D(  1, IFR, -3,   9),IFR=1,2)/ -2, -1/
C     Diagram 10 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,  10)/  10/
      DATA (IFOREST_D(  1, IFR, -1,  10),IFR=1,2)/  1,  5/
      DATA TPRID_D(  1,  -1,  10)/      21/
      DATA (IFOREST_D(  1, IFR, -2,  10),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  1,  -2,  10)/       6/
      DATA (IFOREST_D(  1, IFR, -3,  10),IFR=1,2)/ -2,  3/
C     Diagram 11 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,  11)/  11/
      DATA (IFOREST_D(  1, IFR, -1,  11),IFR=1,2)/  1,  5/
      DATA TPRID_D(  1,  -1,  11)/      21/
      DATA (IFOREST_D(  1, IFR, -2,  11),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  1,  -2,  11)/       6/
      DATA (IFOREST_D(  1, IFR, -3,  11),IFR=1,2)/ -2,  4/
C     Diagram 12 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,  12)/  12/
      DATA (IFOREST_D(  1, IFR, -1,  12),IFR=1,2)/  4,  3/
      DATA SPROP_D(  1,  -1,  12)/      21/
      DATA (IFOREST_D(  1, IFR, -2,  12),IFR=1,2)/  1,  5/
      DATA TPRID_D(  1,  -2,  12)/      21/
      DATA (IFOREST_D(  1, IFR, -3,  12),IFR=1,2)/ -2, -1/
C     Diagram 13 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,  13)/  13/
      DATA (IFOREST_D(  1, IFR, -1,  13),IFR=1,2)/  5,  4/
      DATA SPROP_D(  1,  -1,  13)/      -6/
      DATA (IFOREST_D(  1, IFR, -2,  13),IFR=1,2)/  1, -1/
      DATA TPRID_D(  1,  -2,  13)/       6/
      DATA (IFOREST_D(  1, IFR, -3,  13),IFR=1,2)/ -2,  3/
C     Diagram 14 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,  14)/  14/
      DATA (IFOREST_D(  1, IFR, -1,  14),IFR=1,2)/  5,  3/
      DATA SPROP_D(  1,  -1,  14)/       6/
      DATA (IFOREST_D(  1, IFR, -2,  14),IFR=1,2)/  1, -1/
      DATA TPRID_D(  1,  -2,  14)/       6/
      DATA (IFOREST_D(  1, IFR, -3,  14),IFR=1,2)/ -2,  4/
C     Diagram 15 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,  15)/  15/
      DATA (IFOREST_D(  1, IFR, -1,  15),IFR=1,2)/  4,  3/
      DATA SPROP_D(  1,  -1,  15)/      21/
      DATA (IFOREST_D(  1, IFR, -2,  15),IFR=1,2)/  1, -1/
      DATA TPRID_D(  1,  -2,  15)/      21/
      DATA (IFOREST_D(  1, IFR, -3,  15),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 1
      DATA MAPCONFIG_D(  1,0)/  15/
C     
C     nFKSprocess 2
C     Diagram 1 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   1)/   1/
      DATA (IFOREST_D(  2, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  2,  -1,   1)/      21/
      DATA (IFOREST_D(  2, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  2,  -2,   1)/      21/
C     Diagram 2 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   2)/   2/
      DATA (IFOREST_D(  2, IFR, -1,   2),IFR=1,2)/  5,  3/
      DATA SPROP_D(  2,  -1,   2)/       6/
      DATA (IFOREST_D(  2, IFR, -2,   2),IFR=1,2)/  4, -1/
      DATA SPROP_D(  2,  -2,   2)/      21/
C     Diagram 3 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   3)/   3/
      DATA (IFOREST_D(  2, IFR, -1,   3),IFR=1,2)/  5,  4/
      DATA SPROP_D(  2,  -1,   3)/      -6/
      DATA (IFOREST_D(  2, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA SPROP_D(  2,  -2,   3)/      21/
C     Diagram 4 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   4)/   4/
      DATA (IFOREST_D(  2, IFR, -1,   4),IFR=1,2)/  1,  3/
      DATA TPRID_D(  2,  -1,   4)/       6/
      DATA (IFOREST_D(  2, IFR, -2,   4),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  2,  -2,   4)/       6/
      DATA (IFOREST_D(  2, IFR, -3,   4),IFR=1,2)/ -2,  4/
C     Diagram 5 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   5)/   5/
      DATA (IFOREST_D(  2, IFR, -1,   5),IFR=1,2)/  1,  3/
      DATA TPRID_D(  2,  -1,   5)/       6/
      DATA (IFOREST_D(  2, IFR, -2,   5),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  2,  -2,   5)/      21/
      DATA (IFOREST_D(  2, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Diagram 6 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   6)/   6/
      DATA (IFOREST_D(  2, IFR, -1,   6),IFR=1,2)/  5,  4/
      DATA SPROP_D(  2,  -1,   6)/      -6/
      DATA (IFOREST_D(  2, IFR, -2,   6),IFR=1,2)/  1,  3/
      DATA TPRID_D(  2,  -2,   6)/       6/
      DATA (IFOREST_D(  2, IFR, -3,   6),IFR=1,2)/ -2, -1/
C     Diagram 7 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   7)/   7/
      DATA (IFOREST_D(  2, IFR, -1,   7),IFR=1,2)/  1,  4/
      DATA TPRID_D(  2,  -1,   7)/       6/
      DATA (IFOREST_D(  2, IFR, -2,   7),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  2,  -2,   7)/       6/
      DATA (IFOREST_D(  2, IFR, -3,   7),IFR=1,2)/ -2,  3/
C     Diagram 8 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   8)/   8/
      DATA (IFOREST_D(  2, IFR, -1,   8),IFR=1,2)/  1,  4/
      DATA TPRID_D(  2,  -1,   8)/       6/
      DATA (IFOREST_D(  2, IFR, -2,   8),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  2,  -2,   8)/      21/
      DATA (IFOREST_D(  2, IFR, -3,   8),IFR=1,2)/ -2,  5/
C     Diagram 9 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   9)/   9/
      DATA (IFOREST_D(  2, IFR, -1,   9),IFR=1,2)/  5,  3/
      DATA SPROP_D(  2,  -1,   9)/       6/
      DATA (IFOREST_D(  2, IFR, -2,   9),IFR=1,2)/  1,  4/
      DATA TPRID_D(  2,  -2,   9)/       6/
      DATA (IFOREST_D(  2, IFR, -3,   9),IFR=1,2)/ -2, -1/
C     Diagram 10 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,  10)/  10/
      DATA (IFOREST_D(  2, IFR, -1,  10),IFR=1,2)/  1,  5/
      DATA TPRID_D(  2,  -1,  10)/      21/
      DATA (IFOREST_D(  2, IFR, -2,  10),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  2,  -2,  10)/       6/
      DATA (IFOREST_D(  2, IFR, -3,  10),IFR=1,2)/ -2,  3/
C     Diagram 11 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,  11)/  11/
      DATA (IFOREST_D(  2, IFR, -1,  11),IFR=1,2)/  1,  5/
      DATA TPRID_D(  2,  -1,  11)/      21/
      DATA (IFOREST_D(  2, IFR, -2,  11),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  2,  -2,  11)/       6/
      DATA (IFOREST_D(  2, IFR, -3,  11),IFR=1,2)/ -2,  4/
C     Diagram 12 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,  12)/  12/
      DATA (IFOREST_D(  2, IFR, -1,  12),IFR=1,2)/  4,  3/
      DATA SPROP_D(  2,  -1,  12)/      21/
      DATA (IFOREST_D(  2, IFR, -2,  12),IFR=1,2)/  1,  5/
      DATA TPRID_D(  2,  -2,  12)/      21/
      DATA (IFOREST_D(  2, IFR, -3,  12),IFR=1,2)/ -2, -1/
C     Diagram 13 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,  13)/  13/
      DATA (IFOREST_D(  2, IFR, -1,  13),IFR=1,2)/  5,  4/
      DATA SPROP_D(  2,  -1,  13)/      -6/
      DATA (IFOREST_D(  2, IFR, -2,  13),IFR=1,2)/  1, -1/
      DATA TPRID_D(  2,  -2,  13)/       6/
      DATA (IFOREST_D(  2, IFR, -3,  13),IFR=1,2)/ -2,  3/
C     Diagram 14 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,  14)/  14/
      DATA (IFOREST_D(  2, IFR, -1,  14),IFR=1,2)/  5,  3/
      DATA SPROP_D(  2,  -1,  14)/       6/
      DATA (IFOREST_D(  2, IFR, -2,  14),IFR=1,2)/  1, -1/
      DATA TPRID_D(  2,  -2,  14)/       6/
      DATA (IFOREST_D(  2, IFR, -3,  14),IFR=1,2)/ -2,  4/
C     Diagram 15 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,  15)/  15/
      DATA (IFOREST_D(  2, IFR, -1,  15),IFR=1,2)/  4,  3/
      DATA SPROP_D(  2,  -1,  15)/      21/
      DATA (IFOREST_D(  2, IFR, -2,  15),IFR=1,2)/  1, -1/
      DATA TPRID_D(  2,  -2,  15)/      21/
      DATA (IFOREST_D(  2, IFR, -3,  15),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 2
      DATA MAPCONFIG_D(  2,0)/  15/
C     
C     nFKSprocess 3
C     Diagram 1 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   1)/   1/
      DATA (IFOREST_D(  3, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  3,  -1,   1)/      21/
      DATA (IFOREST_D(  3, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  3,  -2,   1)/      21/
C     Diagram 2 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   2)/   2/
      DATA (IFOREST_D(  3, IFR, -1,   2),IFR=1,2)/  5,  3/
      DATA SPROP_D(  3,  -1,   2)/       6/
      DATA (IFOREST_D(  3, IFR, -2,   2),IFR=1,2)/  4, -1/
      DATA SPROP_D(  3,  -2,   2)/      21/
C     Diagram 3 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   3)/   3/
      DATA (IFOREST_D(  3, IFR, -1,   3),IFR=1,2)/  5,  4/
      DATA SPROP_D(  3,  -1,   3)/      -6/
      DATA (IFOREST_D(  3, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA SPROP_D(  3,  -2,   3)/      21/
C     Diagram 4 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   4)/   4/
      DATA (IFOREST_D(  3, IFR, -1,   4),IFR=1,2)/  1,  3/
      DATA TPRID_D(  3,  -1,   4)/       6/
      DATA (IFOREST_D(  3, IFR, -2,   4),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  3,  -2,   4)/       6/
      DATA (IFOREST_D(  3, IFR, -3,   4),IFR=1,2)/ -2,  4/
C     Diagram 5 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   5)/   5/
      DATA (IFOREST_D(  3, IFR, -1,   5),IFR=1,2)/  1,  3/
      DATA TPRID_D(  3,  -1,   5)/       6/
      DATA (IFOREST_D(  3, IFR, -2,   5),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  3,  -2,   5)/      21/
      DATA (IFOREST_D(  3, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Diagram 6 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   6)/   6/
      DATA (IFOREST_D(  3, IFR, -1,   6),IFR=1,2)/  5,  4/
      DATA SPROP_D(  3,  -1,   6)/      -6/
      DATA (IFOREST_D(  3, IFR, -2,   6),IFR=1,2)/  1,  3/
      DATA TPRID_D(  3,  -2,   6)/       6/
      DATA (IFOREST_D(  3, IFR, -3,   6),IFR=1,2)/ -2, -1/
C     Diagram 7 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   7)/   7/
      DATA (IFOREST_D(  3, IFR, -1,   7),IFR=1,2)/  1,  4/
      DATA TPRID_D(  3,  -1,   7)/       6/
      DATA (IFOREST_D(  3, IFR, -2,   7),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  3,  -2,   7)/       6/
      DATA (IFOREST_D(  3, IFR, -3,   7),IFR=1,2)/ -2,  3/
C     Diagram 8 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   8)/   8/
      DATA (IFOREST_D(  3, IFR, -1,   8),IFR=1,2)/  1,  4/
      DATA TPRID_D(  3,  -1,   8)/       6/
      DATA (IFOREST_D(  3, IFR, -2,   8),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  3,  -2,   8)/      21/
      DATA (IFOREST_D(  3, IFR, -3,   8),IFR=1,2)/ -2,  5/
C     Diagram 9 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   9)/   9/
      DATA (IFOREST_D(  3, IFR, -1,   9),IFR=1,2)/  5,  3/
      DATA SPROP_D(  3,  -1,   9)/       6/
      DATA (IFOREST_D(  3, IFR, -2,   9),IFR=1,2)/  1,  4/
      DATA TPRID_D(  3,  -2,   9)/       6/
      DATA (IFOREST_D(  3, IFR, -3,   9),IFR=1,2)/ -2, -1/
C     Diagram 10 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,  10)/  10/
      DATA (IFOREST_D(  3, IFR, -1,  10),IFR=1,2)/  1,  5/
      DATA TPRID_D(  3,  -1,  10)/      21/
      DATA (IFOREST_D(  3, IFR, -2,  10),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  3,  -2,  10)/       6/
      DATA (IFOREST_D(  3, IFR, -3,  10),IFR=1,2)/ -2,  3/
C     Diagram 11 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,  11)/  11/
      DATA (IFOREST_D(  3, IFR, -1,  11),IFR=1,2)/  1,  5/
      DATA TPRID_D(  3,  -1,  11)/      21/
      DATA (IFOREST_D(  3, IFR, -2,  11),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  3,  -2,  11)/       6/
      DATA (IFOREST_D(  3, IFR, -3,  11),IFR=1,2)/ -2,  4/
C     Diagram 12 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,  12)/  12/
      DATA (IFOREST_D(  3, IFR, -1,  12),IFR=1,2)/  4,  3/
      DATA SPROP_D(  3,  -1,  12)/      21/
      DATA (IFOREST_D(  3, IFR, -2,  12),IFR=1,2)/  1,  5/
      DATA TPRID_D(  3,  -2,  12)/      21/
      DATA (IFOREST_D(  3, IFR, -3,  12),IFR=1,2)/ -2, -1/
C     Diagram 13 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,  13)/  13/
      DATA (IFOREST_D(  3, IFR, -1,  13),IFR=1,2)/  5,  4/
      DATA SPROP_D(  3,  -1,  13)/      -6/
      DATA (IFOREST_D(  3, IFR, -2,  13),IFR=1,2)/  1, -1/
      DATA TPRID_D(  3,  -2,  13)/       6/
      DATA (IFOREST_D(  3, IFR, -3,  13),IFR=1,2)/ -2,  3/
C     Diagram 14 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,  14)/  14/
      DATA (IFOREST_D(  3, IFR, -1,  14),IFR=1,2)/  5,  3/
      DATA SPROP_D(  3,  -1,  14)/       6/
      DATA (IFOREST_D(  3, IFR, -2,  14),IFR=1,2)/  1, -1/
      DATA TPRID_D(  3,  -2,  14)/       6/
      DATA (IFOREST_D(  3, IFR, -3,  14),IFR=1,2)/ -2,  4/
C     Diagram 15 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,  15)/  15/
      DATA (IFOREST_D(  3, IFR, -1,  15),IFR=1,2)/  4,  3/
      DATA SPROP_D(  3,  -1,  15)/      21/
      DATA (IFOREST_D(  3, IFR, -2,  15),IFR=1,2)/  1, -1/
      DATA TPRID_D(  3,  -2,  15)/      21/
      DATA (IFOREST_D(  3, IFR, -3,  15),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 3
      DATA MAPCONFIG_D(  3,0)/  15/
C     
C     nFKSprocess 4
C     Diagram 1 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   1)/   1/
      DATA (IFOREST_D(  4, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  4,  -1,   1)/      21/
      DATA (IFOREST_D(  4, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  4,  -2,   1)/      21/
C     Diagram 2 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   2)/   2/
      DATA (IFOREST_D(  4, IFR, -1,   2),IFR=1,2)/  5,  3/
      DATA SPROP_D(  4,  -1,   2)/       6/
      DATA (IFOREST_D(  4, IFR, -2,   2),IFR=1,2)/  4, -1/
      DATA SPROP_D(  4,  -2,   2)/      21/
C     Diagram 3 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   3)/   3/
      DATA (IFOREST_D(  4, IFR, -1,   3),IFR=1,2)/  5,  4/
      DATA SPROP_D(  4,  -1,   3)/      -6/
      DATA (IFOREST_D(  4, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA SPROP_D(  4,  -2,   3)/      21/
C     Diagram 4 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   4)/   4/
      DATA (IFOREST_D(  4, IFR, -1,   4),IFR=1,2)/  1,  3/
      DATA TPRID_D(  4,  -1,   4)/       6/
      DATA (IFOREST_D(  4, IFR, -2,   4),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  4,  -2,   4)/       6/
      DATA (IFOREST_D(  4, IFR, -3,   4),IFR=1,2)/ -2,  4/
C     Diagram 5 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   5)/   5/
      DATA (IFOREST_D(  4, IFR, -1,   5),IFR=1,2)/  1,  3/
      DATA TPRID_D(  4,  -1,   5)/       6/
      DATA (IFOREST_D(  4, IFR, -2,   5),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  4,  -2,   5)/      21/
      DATA (IFOREST_D(  4, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Diagram 6 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   6)/   6/
      DATA (IFOREST_D(  4, IFR, -1,   6),IFR=1,2)/  5,  4/
      DATA SPROP_D(  4,  -1,   6)/      -6/
      DATA (IFOREST_D(  4, IFR, -2,   6),IFR=1,2)/  1,  3/
      DATA TPRID_D(  4,  -2,   6)/       6/
      DATA (IFOREST_D(  4, IFR, -3,   6),IFR=1,2)/ -2, -1/
C     Diagram 7 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   7)/   7/
      DATA (IFOREST_D(  4, IFR, -1,   7),IFR=1,2)/  1,  4/
      DATA TPRID_D(  4,  -1,   7)/       6/
      DATA (IFOREST_D(  4, IFR, -2,   7),IFR=1,2)/ -1,  5/
      DATA TPRID_D(  4,  -2,   7)/       6/
      DATA (IFOREST_D(  4, IFR, -3,   7),IFR=1,2)/ -2,  3/
C     Diagram 8 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   8)/   8/
      DATA (IFOREST_D(  4, IFR, -1,   8),IFR=1,2)/  1,  4/
      DATA TPRID_D(  4,  -1,   8)/       6/
      DATA (IFOREST_D(  4, IFR, -2,   8),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  4,  -2,   8)/      21/
      DATA (IFOREST_D(  4, IFR, -3,   8),IFR=1,2)/ -2,  5/
C     Diagram 9 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   9)/   9/
      DATA (IFOREST_D(  4, IFR, -1,   9),IFR=1,2)/  5,  3/
      DATA SPROP_D(  4,  -1,   9)/       6/
      DATA (IFOREST_D(  4, IFR, -2,   9),IFR=1,2)/  1,  4/
      DATA TPRID_D(  4,  -2,   9)/       6/
      DATA (IFOREST_D(  4, IFR, -3,   9),IFR=1,2)/ -2, -1/
C     Diagram 10 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,  10)/  10/
      DATA (IFOREST_D(  4, IFR, -1,  10),IFR=1,2)/  1,  5/
      DATA TPRID_D(  4,  -1,  10)/      21/
      DATA (IFOREST_D(  4, IFR, -2,  10),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  4,  -2,  10)/       6/
      DATA (IFOREST_D(  4, IFR, -3,  10),IFR=1,2)/ -2,  3/
C     Diagram 11 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,  11)/  11/
      DATA (IFOREST_D(  4, IFR, -1,  11),IFR=1,2)/  1,  5/
      DATA TPRID_D(  4,  -1,  11)/      21/
      DATA (IFOREST_D(  4, IFR, -2,  11),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  4,  -2,  11)/       6/
      DATA (IFOREST_D(  4, IFR, -3,  11),IFR=1,2)/ -2,  4/
C     Diagram 12 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,  12)/  12/
      DATA (IFOREST_D(  4, IFR, -1,  12),IFR=1,2)/  4,  3/
      DATA SPROP_D(  4,  -1,  12)/      21/
      DATA (IFOREST_D(  4, IFR, -2,  12),IFR=1,2)/  1,  5/
      DATA TPRID_D(  4,  -2,  12)/      21/
      DATA (IFOREST_D(  4, IFR, -3,  12),IFR=1,2)/ -2, -1/
C     Diagram 13 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,  13)/  13/
      DATA (IFOREST_D(  4, IFR, -1,  13),IFR=1,2)/  5,  4/
      DATA SPROP_D(  4,  -1,  13)/      -6/
      DATA (IFOREST_D(  4, IFR, -2,  13),IFR=1,2)/  1, -1/
      DATA TPRID_D(  4,  -2,  13)/       6/
      DATA (IFOREST_D(  4, IFR, -3,  13),IFR=1,2)/ -2,  3/
C     Diagram 14 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,  14)/  14/
      DATA (IFOREST_D(  4, IFR, -1,  14),IFR=1,2)/  5,  3/
      DATA SPROP_D(  4,  -1,  14)/       6/
      DATA (IFOREST_D(  4, IFR, -2,  14),IFR=1,2)/  1, -1/
      DATA TPRID_D(  4,  -2,  14)/       6/
      DATA (IFOREST_D(  4, IFR, -3,  14),IFR=1,2)/ -2,  4/
C     Diagram 15 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,  15)/  15/
      DATA (IFOREST_D(  4, IFR, -1,  15),IFR=1,2)/  4,  3/
      DATA SPROP_D(  4,  -1,  15)/      21/
      DATA (IFOREST_D(  4, IFR, -2,  15),IFR=1,2)/  1, -1/
      DATA TPRID_D(  4,  -2,  15)/      21/
      DATA (IFOREST_D(  4, IFR, -3,  15),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 4
      DATA MAPCONFIG_D(  4,0)/  15/
C     
C     nFKSprocess 5
C     Diagram 1 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   1)/   1/
      DATA (IFOREST_D(  5, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  5,  -1,   1)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  5,  -2,   1)/      -1/
C     Diagram 2 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   2)/   2/
      DATA (IFOREST_D(  5, IFR, -1,   2),IFR=1,2)/  1,  5/
      DATA TPRID_D(  5,  -1,   2)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   2),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  5,  -2,   2)/       6/
      DATA (IFOREST_D(  5, IFR, -3,   2),IFR=1,2)/ -2,  3/
C     Diagram 3 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   3)/   3/
      DATA (IFOREST_D(  5, IFR, -1,   3),IFR=1,2)/  1,  5/
      DATA TPRID_D(  5,  -1,   3)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  5,  -2,   3)/       6/
      DATA (IFOREST_D(  5, IFR, -3,   3),IFR=1,2)/ -2,  4/
C     Diagram 4 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   4)/   4/
      DATA (IFOREST_D(  5, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  5,  -1,   4)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  5,  -2,   4)/      21/
      DATA (IFOREST_D(  5, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   5)/   5/
      DATA (IFOREST_D(  5, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  5,  -1,   5)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  5,  -2,   5)/       1/
      DATA (IFOREST_D(  5, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 5
      DATA MAPCONFIG_D(  5,0)/   5/
C     
C     nFKSprocess 6
C     Diagram 1 for nFKSprocess 6
      DATA MAPCONFIG_D(  6,   1)/   1/
      DATA (IFOREST_D(  6, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  6,  -1,   1)/      21/
      DATA (IFOREST_D(  6, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  6,  -2,   1)/       1/
C     Diagram 2 for nFKSprocess 6
      DATA MAPCONFIG_D(  6,   2)/   2/
      DATA (IFOREST_D(  6, IFR, -1,   2),IFR=1,2)/  1,  5/
      DATA TPRID_D(  6,  -1,   2)/      21/
      DATA (IFOREST_D(  6, IFR, -2,   2),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  6,  -2,   2)/       6/
      DATA (IFOREST_D(  6, IFR, -3,   2),IFR=1,2)/ -2,  3/
C     Diagram 3 for nFKSprocess 6
      DATA MAPCONFIG_D(  6,   3)/   3/
      DATA (IFOREST_D(  6, IFR, -1,   3),IFR=1,2)/  1,  5/
      DATA TPRID_D(  6,  -1,   3)/      21/
      DATA (IFOREST_D(  6, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  6,  -2,   3)/       6/
      DATA (IFOREST_D(  6, IFR, -3,   3),IFR=1,2)/ -2,  4/
C     Diagram 4 for nFKSprocess 6
      DATA MAPCONFIG_D(  6,   4)/   4/
      DATA (IFOREST_D(  6, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  6,  -1,   4)/      21/
      DATA (IFOREST_D(  6, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  6,  -2,   4)/      21/
      DATA (IFOREST_D(  6, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 6
      DATA MAPCONFIG_D(  6,   5)/   5/
      DATA (IFOREST_D(  6, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  6,  -1,   5)/      21/
      DATA (IFOREST_D(  6, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  6,  -2,   5)/       1/
      DATA (IFOREST_D(  6, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 6
      DATA MAPCONFIG_D(  6,0)/   5/
C     
C     nFKSprocess 7
C     Diagram 1 for nFKSprocess 7
      DATA MAPCONFIG_D(  7,   1)/   1/
      DATA (IFOREST_D(  7, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  7,  -1,   1)/      21/
      DATA (IFOREST_D(  7, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  7,  -2,   1)/      -1/
C     Diagram 2 for nFKSprocess 7
      DATA MAPCONFIG_D(  7,   2)/   2/
      DATA (IFOREST_D(  7, IFR, -1,   2),IFR=1,2)/  1,  3/
      DATA TPRID_D(  7,  -1,   2)/       6/
      DATA (IFOREST_D(  7, IFR, -2,   2),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  7,  -2,   2)/      21/
      DATA (IFOREST_D(  7, IFR, -3,   2),IFR=1,2)/ -2,  5/
C     Diagram 3 for nFKSprocess 7
      DATA MAPCONFIG_D(  7,   3)/   3/
      DATA (IFOREST_D(  7, IFR, -1,   3),IFR=1,2)/  1,  4/
      DATA TPRID_D(  7,  -1,   3)/       6/
      DATA (IFOREST_D(  7, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  7,  -2,   3)/      21/
      DATA (IFOREST_D(  7, IFR, -3,   3),IFR=1,2)/ -2,  5/
C     Diagram 4 for nFKSprocess 7
      DATA MAPCONFIG_D(  7,   4)/   4/
      DATA (IFOREST_D(  7, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  7,  -1,   4)/      21/
      DATA (IFOREST_D(  7, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  7,  -2,   4)/       1/
      DATA (IFOREST_D(  7, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 7
      DATA MAPCONFIG_D(  7,   5)/   5/
      DATA (IFOREST_D(  7, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  7,  -1,   5)/      21/
      DATA (IFOREST_D(  7, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  7,  -2,   5)/      21/
      DATA (IFOREST_D(  7, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 7
      DATA MAPCONFIG_D(  7,0)/   5/
C     
C     nFKSprocess 8
C     Diagram 1 for nFKSprocess 8
      DATA MAPCONFIG_D(  8,   1)/   1/
      DATA (IFOREST_D(  8, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  8,  -1,   1)/      21/
      DATA (IFOREST_D(  8, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  8,  -2,   1)/       1/
C     Diagram 2 for nFKSprocess 8
      DATA MAPCONFIG_D(  8,   2)/   2/
      DATA (IFOREST_D(  8, IFR, -1,   2),IFR=1,2)/  1,  3/
      DATA TPRID_D(  8,  -1,   2)/       6/
      DATA (IFOREST_D(  8, IFR, -2,   2),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  8,  -2,   2)/      21/
      DATA (IFOREST_D(  8, IFR, -3,   2),IFR=1,2)/ -2,  5/
C     Diagram 3 for nFKSprocess 8
      DATA MAPCONFIG_D(  8,   3)/   3/
      DATA (IFOREST_D(  8, IFR, -1,   3),IFR=1,2)/  1,  4/
      DATA TPRID_D(  8,  -1,   3)/       6/
      DATA (IFOREST_D(  8, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  8,  -2,   3)/      21/
      DATA (IFOREST_D(  8, IFR, -3,   3),IFR=1,2)/ -2,  5/
C     Diagram 4 for nFKSprocess 8
      DATA MAPCONFIG_D(  8,   4)/   4/
      DATA (IFOREST_D(  8, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  8,  -1,   4)/      21/
      DATA (IFOREST_D(  8, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  8,  -2,   4)/       1/
      DATA (IFOREST_D(  8, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 8
      DATA MAPCONFIG_D(  8,   5)/   5/
      DATA (IFOREST_D(  8, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  8,  -1,   5)/      21/
      DATA (IFOREST_D(  8, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  8,  -2,   5)/      21/
      DATA (IFOREST_D(  8, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 8
      DATA MAPCONFIG_D(  8,0)/   5/
C     
C     
      PMASS_D (  1, -1,   1) = ZERO
      PWIDTH_D(  1, -1,   1) = ZERO
      POW_D   (  1, -1,   1) = 2
      PMASS_D (  1, -2,   1) = ZERO
      PWIDTH_D(  1, -2,   1) = ZERO
      POW_D   (  1, -2,   1) = 2
      PMASS_D (  1, -1,   2) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   2) = ABS(MDL_WT)
      POW_D   (  1, -1,   2) = 1
      PMASS_D (  1, -2,   2) = ZERO
      PWIDTH_D(  1, -2,   2) = ZERO
      POW_D   (  1, -2,   2) = 2
      PMASS_D (  1, -1,   3) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   3) = ABS(MDL_WT)
      POW_D   (  1, -1,   3) = 1
      PMASS_D (  1, -2,   3) = ZERO
      PWIDTH_D(  1, -2,   3) = ZERO
      POW_D   (  1, -2,   3) = 2
      PMASS_D (  1, -1,   4) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   4) = ABS(MDL_WT)
      POW_D   (  1, -1,   4) = 1
      PMASS_D (  1, -2,   4) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,   4) = ABS(MDL_WT)
      POW_D   (  1, -2,   4) = 1
      PMASS_D (  1, -1,   5) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   5) = ABS(MDL_WT)
      POW_D   (  1, -1,   5) = 1
      PMASS_D (  1, -2,   5) = ZERO
      PWIDTH_D(  1, -2,   5) = ZERO
      POW_D   (  1, -2,   5) = 2
      PMASS_D (  1, -1,   6) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   6) = ABS(MDL_WT)
      POW_D   (  1, -1,   6) = 1
      PMASS_D (  1, -2,   6) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,   6) = ABS(MDL_WT)
      POW_D   (  1, -2,   6) = 1
      PMASS_D (  1, -1,   7) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   7) = ABS(MDL_WT)
      POW_D   (  1, -1,   7) = 1
      PMASS_D (  1, -2,   7) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,   7) = ABS(MDL_WT)
      POW_D   (  1, -2,   7) = 1
      PMASS_D (  1, -1,   8) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   8) = ABS(MDL_WT)
      POW_D   (  1, -1,   8) = 1
      PMASS_D (  1, -2,   8) = ZERO
      PWIDTH_D(  1, -2,   8) = ZERO
      POW_D   (  1, -2,   8) = 2
      PMASS_D (  1, -1,   9) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,   9) = ABS(MDL_WT)
      POW_D   (  1, -1,   9) = 1
      PMASS_D (  1, -2,   9) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,   9) = ABS(MDL_WT)
      POW_D   (  1, -2,   9) = 1
      PMASS_D (  1, -1,  10) = ZERO
      PWIDTH_D(  1, -1,  10) = ZERO
      POW_D   (  1, -1,  10) = 2
      PMASS_D (  1, -2,  10) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,  10) = ABS(MDL_WT)
      POW_D   (  1, -2,  10) = 1
      PMASS_D (  1, -1,  11) = ZERO
      PWIDTH_D(  1, -1,  11) = ZERO
      POW_D   (  1, -1,  11) = 2
      PMASS_D (  1, -2,  11) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,  11) = ABS(MDL_WT)
      POW_D   (  1, -2,  11) = 1
      PMASS_D (  1, -1,  12) = ZERO
      PWIDTH_D(  1, -1,  12) = ZERO
      POW_D   (  1, -1,  12) = 2
      PMASS_D (  1, -2,  12) = ZERO
      PWIDTH_D(  1, -2,  12) = ZERO
      POW_D   (  1, -2,  12) = 2
      PMASS_D (  1, -1,  13) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,  13) = ABS(MDL_WT)
      POW_D   (  1, -1,  13) = 1
      PMASS_D (  1, -2,  13) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,  13) = ABS(MDL_WT)
      POW_D   (  1, -2,  13) = 1
      PMASS_D (  1, -1,  14) = ABS(MDL_MT)
      PWIDTH_D(  1, -1,  14) = ABS(MDL_WT)
      POW_D   (  1, -1,  14) = 1
      PMASS_D (  1, -2,  14) = ABS(MDL_MT)
      PWIDTH_D(  1, -2,  14) = ABS(MDL_WT)
      POW_D   (  1, -2,  14) = 1
      PMASS_D (  1, -1,  15) = ZERO
      PWIDTH_D(  1, -1,  15) = ZERO
      POW_D   (  1, -1,  15) = 2
      PMASS_D (  1, -2,  15) = ZERO
      PWIDTH_D(  1, -2,  15) = ZERO
      POW_D   (  1, -2,  15) = 2
C     
      PMASS_D (  2, -1,   1) = ZERO
      PWIDTH_D(  2, -1,   1) = ZERO
      POW_D   (  2, -1,   1) = 2
      PMASS_D (  2, -2,   1) = ZERO
      PWIDTH_D(  2, -2,   1) = ZERO
      POW_D   (  2, -2,   1) = 2
      PMASS_D (  2, -1,   2) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   2) = ABS(MDL_WT)
      POW_D   (  2, -1,   2) = 1
      PMASS_D (  2, -2,   2) = ZERO
      PWIDTH_D(  2, -2,   2) = ZERO
      POW_D   (  2, -2,   2) = 2
      PMASS_D (  2, -1,   3) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   3) = ABS(MDL_WT)
      POW_D   (  2, -1,   3) = 1
      PMASS_D (  2, -2,   3) = ZERO
      PWIDTH_D(  2, -2,   3) = ZERO
      POW_D   (  2, -2,   3) = 2
      PMASS_D (  2, -1,   4) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   4) = ABS(MDL_WT)
      POW_D   (  2, -1,   4) = 1
      PMASS_D (  2, -2,   4) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,   4) = ABS(MDL_WT)
      POW_D   (  2, -2,   4) = 1
      PMASS_D (  2, -1,   5) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   5) = ABS(MDL_WT)
      POW_D   (  2, -1,   5) = 1
      PMASS_D (  2, -2,   5) = ZERO
      PWIDTH_D(  2, -2,   5) = ZERO
      POW_D   (  2, -2,   5) = 2
      PMASS_D (  2, -1,   6) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   6) = ABS(MDL_WT)
      POW_D   (  2, -1,   6) = 1
      PMASS_D (  2, -2,   6) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,   6) = ABS(MDL_WT)
      POW_D   (  2, -2,   6) = 1
      PMASS_D (  2, -1,   7) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   7) = ABS(MDL_WT)
      POW_D   (  2, -1,   7) = 1
      PMASS_D (  2, -2,   7) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,   7) = ABS(MDL_WT)
      POW_D   (  2, -2,   7) = 1
      PMASS_D (  2, -1,   8) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   8) = ABS(MDL_WT)
      POW_D   (  2, -1,   8) = 1
      PMASS_D (  2, -2,   8) = ZERO
      PWIDTH_D(  2, -2,   8) = ZERO
      POW_D   (  2, -2,   8) = 2
      PMASS_D (  2, -1,   9) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,   9) = ABS(MDL_WT)
      POW_D   (  2, -1,   9) = 1
      PMASS_D (  2, -2,   9) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,   9) = ABS(MDL_WT)
      POW_D   (  2, -2,   9) = 1
      PMASS_D (  2, -1,  10) = ZERO
      PWIDTH_D(  2, -1,  10) = ZERO
      POW_D   (  2, -1,  10) = 2
      PMASS_D (  2, -2,  10) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,  10) = ABS(MDL_WT)
      POW_D   (  2, -2,  10) = 1
      PMASS_D (  2, -1,  11) = ZERO
      PWIDTH_D(  2, -1,  11) = ZERO
      POW_D   (  2, -1,  11) = 2
      PMASS_D (  2, -2,  11) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,  11) = ABS(MDL_WT)
      POW_D   (  2, -2,  11) = 1
      PMASS_D (  2, -1,  12) = ZERO
      PWIDTH_D(  2, -1,  12) = ZERO
      POW_D   (  2, -1,  12) = 2
      PMASS_D (  2, -2,  12) = ZERO
      PWIDTH_D(  2, -2,  12) = ZERO
      POW_D   (  2, -2,  12) = 2
      PMASS_D (  2, -1,  13) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,  13) = ABS(MDL_WT)
      POW_D   (  2, -1,  13) = 1
      PMASS_D (  2, -2,  13) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,  13) = ABS(MDL_WT)
      POW_D   (  2, -2,  13) = 1
      PMASS_D (  2, -1,  14) = ABS(MDL_MT)
      PWIDTH_D(  2, -1,  14) = ABS(MDL_WT)
      POW_D   (  2, -1,  14) = 1
      PMASS_D (  2, -2,  14) = ABS(MDL_MT)
      PWIDTH_D(  2, -2,  14) = ABS(MDL_WT)
      POW_D   (  2, -2,  14) = 1
      PMASS_D (  2, -1,  15) = ZERO
      PWIDTH_D(  2, -1,  15) = ZERO
      POW_D   (  2, -1,  15) = 2
      PMASS_D (  2, -2,  15) = ZERO
      PWIDTH_D(  2, -2,  15) = ZERO
      POW_D   (  2, -2,  15) = 2
C     
      PMASS_D (  3, -1,   1) = ZERO
      PWIDTH_D(  3, -1,   1) = ZERO
      POW_D   (  3, -1,   1) = 2
      PMASS_D (  3, -2,   1) = ZERO
      PWIDTH_D(  3, -2,   1) = ZERO
      POW_D   (  3, -2,   1) = 2
      PMASS_D (  3, -1,   2) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   2) = ABS(MDL_WT)
      POW_D   (  3, -1,   2) = 1
      PMASS_D (  3, -2,   2) = ZERO
      PWIDTH_D(  3, -2,   2) = ZERO
      POW_D   (  3, -2,   2) = 2
      PMASS_D (  3, -1,   3) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   3) = ABS(MDL_WT)
      POW_D   (  3, -1,   3) = 1
      PMASS_D (  3, -2,   3) = ZERO
      PWIDTH_D(  3, -2,   3) = ZERO
      POW_D   (  3, -2,   3) = 2
      PMASS_D (  3, -1,   4) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   4) = ABS(MDL_WT)
      POW_D   (  3, -1,   4) = 1
      PMASS_D (  3, -2,   4) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,   4) = ABS(MDL_WT)
      POW_D   (  3, -2,   4) = 1
      PMASS_D (  3, -1,   5) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   5) = ABS(MDL_WT)
      POW_D   (  3, -1,   5) = 1
      PMASS_D (  3, -2,   5) = ZERO
      PWIDTH_D(  3, -2,   5) = ZERO
      POW_D   (  3, -2,   5) = 2
      PMASS_D (  3, -1,   6) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   6) = ABS(MDL_WT)
      POW_D   (  3, -1,   6) = 1
      PMASS_D (  3, -2,   6) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,   6) = ABS(MDL_WT)
      POW_D   (  3, -2,   6) = 1
      PMASS_D (  3, -1,   7) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   7) = ABS(MDL_WT)
      POW_D   (  3, -1,   7) = 1
      PMASS_D (  3, -2,   7) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,   7) = ABS(MDL_WT)
      POW_D   (  3, -2,   7) = 1
      PMASS_D (  3, -1,   8) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   8) = ABS(MDL_WT)
      POW_D   (  3, -1,   8) = 1
      PMASS_D (  3, -2,   8) = ZERO
      PWIDTH_D(  3, -2,   8) = ZERO
      POW_D   (  3, -2,   8) = 2
      PMASS_D (  3, -1,   9) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,   9) = ABS(MDL_WT)
      POW_D   (  3, -1,   9) = 1
      PMASS_D (  3, -2,   9) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,   9) = ABS(MDL_WT)
      POW_D   (  3, -2,   9) = 1
      PMASS_D (  3, -1,  10) = ZERO
      PWIDTH_D(  3, -1,  10) = ZERO
      POW_D   (  3, -1,  10) = 2
      PMASS_D (  3, -2,  10) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,  10) = ABS(MDL_WT)
      POW_D   (  3, -2,  10) = 1
      PMASS_D (  3, -1,  11) = ZERO
      PWIDTH_D(  3, -1,  11) = ZERO
      POW_D   (  3, -1,  11) = 2
      PMASS_D (  3, -2,  11) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,  11) = ABS(MDL_WT)
      POW_D   (  3, -2,  11) = 1
      PMASS_D (  3, -1,  12) = ZERO
      PWIDTH_D(  3, -1,  12) = ZERO
      POW_D   (  3, -1,  12) = 2
      PMASS_D (  3, -2,  12) = ZERO
      PWIDTH_D(  3, -2,  12) = ZERO
      POW_D   (  3, -2,  12) = 2
      PMASS_D (  3, -1,  13) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,  13) = ABS(MDL_WT)
      POW_D   (  3, -1,  13) = 1
      PMASS_D (  3, -2,  13) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,  13) = ABS(MDL_WT)
      POW_D   (  3, -2,  13) = 1
      PMASS_D (  3, -1,  14) = ABS(MDL_MT)
      PWIDTH_D(  3, -1,  14) = ABS(MDL_WT)
      POW_D   (  3, -1,  14) = 1
      PMASS_D (  3, -2,  14) = ABS(MDL_MT)
      PWIDTH_D(  3, -2,  14) = ABS(MDL_WT)
      POW_D   (  3, -2,  14) = 1
      PMASS_D (  3, -1,  15) = ZERO
      PWIDTH_D(  3, -1,  15) = ZERO
      POW_D   (  3, -1,  15) = 2
      PMASS_D (  3, -2,  15) = ZERO
      PWIDTH_D(  3, -2,  15) = ZERO
      POW_D   (  3, -2,  15) = 2
C     
      PMASS_D (  4, -1,   1) = ZERO
      PWIDTH_D(  4, -1,   1) = ZERO
      POW_D   (  4, -1,   1) = 2
      PMASS_D (  4, -2,   1) = ZERO
      PWIDTH_D(  4, -2,   1) = ZERO
      POW_D   (  4, -2,   1) = 2
      PMASS_D (  4, -1,   2) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   2) = ABS(MDL_WT)
      POW_D   (  4, -1,   2) = 1
      PMASS_D (  4, -2,   2) = ZERO
      PWIDTH_D(  4, -2,   2) = ZERO
      POW_D   (  4, -2,   2) = 2
      PMASS_D (  4, -1,   3) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   3) = ABS(MDL_WT)
      POW_D   (  4, -1,   3) = 1
      PMASS_D (  4, -2,   3) = ZERO
      PWIDTH_D(  4, -2,   3) = ZERO
      POW_D   (  4, -2,   3) = 2
      PMASS_D (  4, -1,   4) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   4) = ABS(MDL_WT)
      POW_D   (  4, -1,   4) = 1
      PMASS_D (  4, -2,   4) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,   4) = ABS(MDL_WT)
      POW_D   (  4, -2,   4) = 1
      PMASS_D (  4, -1,   5) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   5) = ABS(MDL_WT)
      POW_D   (  4, -1,   5) = 1
      PMASS_D (  4, -2,   5) = ZERO
      PWIDTH_D(  4, -2,   5) = ZERO
      POW_D   (  4, -2,   5) = 2
      PMASS_D (  4, -1,   6) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   6) = ABS(MDL_WT)
      POW_D   (  4, -1,   6) = 1
      PMASS_D (  4, -2,   6) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,   6) = ABS(MDL_WT)
      POW_D   (  4, -2,   6) = 1
      PMASS_D (  4, -1,   7) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   7) = ABS(MDL_WT)
      POW_D   (  4, -1,   7) = 1
      PMASS_D (  4, -2,   7) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,   7) = ABS(MDL_WT)
      POW_D   (  4, -2,   7) = 1
      PMASS_D (  4, -1,   8) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   8) = ABS(MDL_WT)
      POW_D   (  4, -1,   8) = 1
      PMASS_D (  4, -2,   8) = ZERO
      PWIDTH_D(  4, -2,   8) = ZERO
      POW_D   (  4, -2,   8) = 2
      PMASS_D (  4, -1,   9) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,   9) = ABS(MDL_WT)
      POW_D   (  4, -1,   9) = 1
      PMASS_D (  4, -2,   9) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,   9) = ABS(MDL_WT)
      POW_D   (  4, -2,   9) = 1
      PMASS_D (  4, -1,  10) = ZERO
      PWIDTH_D(  4, -1,  10) = ZERO
      POW_D   (  4, -1,  10) = 2
      PMASS_D (  4, -2,  10) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,  10) = ABS(MDL_WT)
      POW_D   (  4, -2,  10) = 1
      PMASS_D (  4, -1,  11) = ZERO
      PWIDTH_D(  4, -1,  11) = ZERO
      POW_D   (  4, -1,  11) = 2
      PMASS_D (  4, -2,  11) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,  11) = ABS(MDL_WT)
      POW_D   (  4, -2,  11) = 1
      PMASS_D (  4, -1,  12) = ZERO
      PWIDTH_D(  4, -1,  12) = ZERO
      POW_D   (  4, -1,  12) = 2
      PMASS_D (  4, -2,  12) = ZERO
      PWIDTH_D(  4, -2,  12) = ZERO
      POW_D   (  4, -2,  12) = 2
      PMASS_D (  4, -1,  13) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,  13) = ABS(MDL_WT)
      POW_D   (  4, -1,  13) = 1
      PMASS_D (  4, -2,  13) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,  13) = ABS(MDL_WT)
      POW_D   (  4, -2,  13) = 1
      PMASS_D (  4, -1,  14) = ABS(MDL_MT)
      PWIDTH_D(  4, -1,  14) = ABS(MDL_WT)
      POW_D   (  4, -1,  14) = 1
      PMASS_D (  4, -2,  14) = ABS(MDL_MT)
      PWIDTH_D(  4, -2,  14) = ABS(MDL_WT)
      POW_D   (  4, -2,  14) = 1
      PMASS_D (  4, -1,  15) = ZERO
      PWIDTH_D(  4, -1,  15) = ZERO
      POW_D   (  4, -1,  15) = 2
      PMASS_D (  4, -2,  15) = ZERO
      PWIDTH_D(  4, -2,  15) = ZERO
      POW_D   (  4, -2,  15) = 2
C     
      PMASS_D (  5, -1,   1) = ZERO
      PWIDTH_D(  5, -1,   1) = ZERO
      POW_D   (  5, -1,   1) = 2
      PMASS_D (  5, -2,   1) = ZERO
      PWIDTH_D(  5, -2,   1) = ZERO
      POW_D   (  5, -2,   1) = 1
      PMASS_D (  5, -1,   2) = ZERO
      PWIDTH_D(  5, -1,   2) = ZERO
      POW_D   (  5, -1,   2) = 2
      PMASS_D (  5, -2,   2) = ABS(MDL_MT)
      PWIDTH_D(  5, -2,   2) = ABS(MDL_WT)
      POW_D   (  5, -2,   2) = 1
      PMASS_D (  5, -1,   3) = ZERO
      PWIDTH_D(  5, -1,   3) = ZERO
      POW_D   (  5, -1,   3) = 2
      PMASS_D (  5, -2,   3) = ABS(MDL_MT)
      PWIDTH_D(  5, -2,   3) = ABS(MDL_WT)
      POW_D   (  5, -2,   3) = 1
      PMASS_D (  5, -1,   4) = ZERO
      PWIDTH_D(  5, -1,   4) = ZERO
      POW_D   (  5, -1,   4) = 2
      PMASS_D (  5, -2,   4) = ZERO
      PWIDTH_D(  5, -2,   4) = ZERO
      POW_D   (  5, -2,   4) = 2
      PMASS_D (  5, -1,   5) = ZERO
      PWIDTH_D(  5, -1,   5) = ZERO
      POW_D   (  5, -1,   5) = 2
      PMASS_D (  5, -2,   5) = ZERO
      PWIDTH_D(  5, -2,   5) = ZERO
      POW_D   (  5, -2,   5) = 1
C     
      PMASS_D (  6, -1,   1) = ZERO
      PWIDTH_D(  6, -1,   1) = ZERO
      POW_D   (  6, -1,   1) = 2
      PMASS_D (  6, -2,   1) = ZERO
      PWIDTH_D(  6, -2,   1) = ZERO
      POW_D   (  6, -2,   1) = 1
      PMASS_D (  6, -1,   2) = ZERO
      PWIDTH_D(  6, -1,   2) = ZERO
      POW_D   (  6, -1,   2) = 2
      PMASS_D (  6, -2,   2) = ABS(MDL_MT)
      PWIDTH_D(  6, -2,   2) = ABS(MDL_WT)
      POW_D   (  6, -2,   2) = 1
      PMASS_D (  6, -1,   3) = ZERO
      PWIDTH_D(  6, -1,   3) = ZERO
      POW_D   (  6, -1,   3) = 2
      PMASS_D (  6, -2,   3) = ABS(MDL_MT)
      PWIDTH_D(  6, -2,   3) = ABS(MDL_WT)
      POW_D   (  6, -2,   3) = 1
      PMASS_D (  6, -1,   4) = ZERO
      PWIDTH_D(  6, -1,   4) = ZERO
      POW_D   (  6, -1,   4) = 2
      PMASS_D (  6, -2,   4) = ZERO
      PWIDTH_D(  6, -2,   4) = ZERO
      POW_D   (  6, -2,   4) = 2
      PMASS_D (  6, -1,   5) = ZERO
      PWIDTH_D(  6, -1,   5) = ZERO
      POW_D   (  6, -1,   5) = 2
      PMASS_D (  6, -2,   5) = ZERO
      PWIDTH_D(  6, -2,   5) = ZERO
      POW_D   (  6, -2,   5) = 1
C     
      PMASS_D (  7, -1,   1) = ZERO
      PWIDTH_D(  7, -1,   1) = ZERO
      POW_D   (  7, -1,   1) = 2
      PMASS_D (  7, -2,   1) = ZERO
      PWIDTH_D(  7, -2,   1) = ZERO
      POW_D   (  7, -2,   1) = 1
      PMASS_D (  7, -1,   2) = ABS(MDL_MT)
      PWIDTH_D(  7, -1,   2) = ABS(MDL_WT)
      POW_D   (  7, -1,   2) = 1
      PMASS_D (  7, -2,   2) = ZERO
      PWIDTH_D(  7, -2,   2) = ZERO
      POW_D   (  7, -2,   2) = 2
      PMASS_D (  7, -1,   3) = ABS(MDL_MT)
      PWIDTH_D(  7, -1,   3) = ABS(MDL_WT)
      POW_D   (  7, -1,   3) = 1
      PMASS_D (  7, -2,   3) = ZERO
      PWIDTH_D(  7, -2,   3) = ZERO
      POW_D   (  7, -2,   3) = 2
      PMASS_D (  7, -1,   4) = ZERO
      PWIDTH_D(  7, -1,   4) = ZERO
      POW_D   (  7, -1,   4) = 2
      PMASS_D (  7, -2,   4) = ZERO
      PWIDTH_D(  7, -2,   4) = ZERO
      POW_D   (  7, -2,   4) = 1
      PMASS_D (  7, -1,   5) = ZERO
      PWIDTH_D(  7, -1,   5) = ZERO
      POW_D   (  7, -1,   5) = 2
      PMASS_D (  7, -2,   5) = ZERO
      PWIDTH_D(  7, -2,   5) = ZERO
      POW_D   (  7, -2,   5) = 2
C     
      PMASS_D (  8, -1,   1) = ZERO
      PWIDTH_D(  8, -1,   1) = ZERO
      POW_D   (  8, -1,   1) = 2
      PMASS_D (  8, -2,   1) = ZERO
      PWIDTH_D(  8, -2,   1) = ZERO
      POW_D   (  8, -2,   1) = 1
      PMASS_D (  8, -1,   2) = ABS(MDL_MT)
      PWIDTH_D(  8, -1,   2) = ABS(MDL_WT)
      POW_D   (  8, -1,   2) = 1
      PMASS_D (  8, -2,   2) = ZERO
      PWIDTH_D(  8, -2,   2) = ZERO
      POW_D   (  8, -2,   2) = 2
      PMASS_D (  8, -1,   3) = ABS(MDL_MT)
      PWIDTH_D(  8, -1,   3) = ABS(MDL_WT)
      POW_D   (  8, -1,   3) = 1
      PMASS_D (  8, -2,   3) = ZERO
      PWIDTH_D(  8, -2,   3) = ZERO
      POW_D   (  8, -2,   3) = 2
      PMASS_D (  8, -1,   4) = ZERO
      PWIDTH_D(  8, -1,   4) = ZERO
      POW_D   (  8, -1,   4) = 2
      PMASS_D (  8, -2,   4) = ZERO
      PWIDTH_D(  8, -2,   4) = ZERO
      POW_D   (  8, -2,   4) = 1
      PMASS_D (  8, -1,   5) = ZERO
      PWIDTH_D(  8, -1,   5) = ZERO
      POW_D   (  8, -1,   5) = 2
      PMASS_D (  8, -2,   5) = ZERO
      PWIDTH_D(  8, -2,   5) = ZERO
      POW_D   (  8, -2,   5) = 2

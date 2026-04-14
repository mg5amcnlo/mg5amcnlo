      INTEGER IFR,LMAXCONFIGS_USED,MAX_BRANCH_USED
      PARAMETER (LMAXCONFIGS_USED=   5)
      PARAMETER (MAX_BRANCH_USED =   3)
      INTEGER MAPCONFIG_D(  6,0:LMAXCONFIGS_USED)
      INTEGER IFOREST_D(  6,2,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
      INTEGER SPROP_D(  6,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
      INTEGER TPRID_D(  6,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
      DOUBLE PRECISION PMASS_D(  6,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED
     $ )
      DOUBLE PRECISION PWIDTH_D(  6,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USE
     $ D)
      INTEGER POW_D(  6,-MAX_BRANCH_USED:-1,LMAXCONFIGS_USED)
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
      DATA (IFOREST_D(  1, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  1,  -1,   4)/      21/
      DATA (IFOREST_D(  1, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  1,  -2,   4)/       2/
      DATA (IFOREST_D(  1, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 1
      DATA MAPCONFIG_D(  1,   5)/   5/
      DATA (IFOREST_D(  1, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  1,  -1,   5)/      21/
      DATA (IFOREST_D(  1, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  1,  -2,   5)/       2/
      DATA (IFOREST_D(  1, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 1
      DATA MAPCONFIG_D(  1,0)/   5/
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
      DATA (IFOREST_D(  2, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  2,  -1,   4)/      21/
      DATA (IFOREST_D(  2, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  2,  -2,   4)/       2/
      DATA (IFOREST_D(  2, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 2
      DATA MAPCONFIG_D(  2,   5)/   5/
      DATA (IFOREST_D(  2, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  2,  -1,   5)/      21/
      DATA (IFOREST_D(  2, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  2,  -2,   5)/       2/
      DATA (IFOREST_D(  2, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 2
      DATA MAPCONFIG_D(  2,0)/   5/
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
      DATA (IFOREST_D(  3, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  3,  -1,   4)/      21/
      DATA (IFOREST_D(  3, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  3,  -2,   4)/       2/
      DATA (IFOREST_D(  3, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 3
      DATA MAPCONFIG_D(  3,   5)/   5/
      DATA (IFOREST_D(  3, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  3,  -1,   5)/      21/
      DATA (IFOREST_D(  3, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  3,  -2,   5)/       2/
      DATA (IFOREST_D(  3, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 3
      DATA MAPCONFIG_D(  3,0)/   5/
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
      DATA (IFOREST_D(  4, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  4,  -1,   4)/      21/
      DATA (IFOREST_D(  4, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  4,  -2,   4)/       2/
      DATA (IFOREST_D(  4, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 4
      DATA MAPCONFIG_D(  4,   5)/   5/
      DATA (IFOREST_D(  4, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  4,  -1,   5)/      21/
      DATA (IFOREST_D(  4, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  4,  -2,   5)/       2/
      DATA (IFOREST_D(  4, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 4
      DATA MAPCONFIG_D(  4,0)/   5/
C     
C     nFKSprocess 5
C     Diagram 1 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   1)/   1/
      DATA (IFOREST_D(  5, IFR, -1,   1),IFR=1,2)/  4,  3/
      DATA SPROP_D(  5,  -1,   1)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   1),IFR=1,2)/  5, -1/
      DATA SPROP_D(  5,  -2,   1)/      -2/
C     Diagram 2 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   2)/   2/
      DATA (IFOREST_D(  5, IFR, -1,   2),IFR=1,2)/  1,  3/
      DATA TPRID_D(  5,  -1,   2)/       6/
      DATA (IFOREST_D(  5, IFR, -2,   2),IFR=1,2)/ -1,  4/
      DATA TPRID_D(  5,  -2,   2)/      21/
      DATA (IFOREST_D(  5, IFR, -3,   2),IFR=1,2)/ -2,  5/
C     Diagram 3 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   3)/   3/
      DATA (IFOREST_D(  5, IFR, -1,   3),IFR=1,2)/  1,  4/
      DATA TPRID_D(  5,  -1,   3)/       6/
      DATA (IFOREST_D(  5, IFR, -2,   3),IFR=1,2)/ -1,  3/
      DATA TPRID_D(  5,  -2,   3)/      21/
      DATA (IFOREST_D(  5, IFR, -3,   3),IFR=1,2)/ -2,  5/
C     Diagram 4 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   4)/   4/
      DATA (IFOREST_D(  5, IFR, -1,   4),IFR=1,2)/  4,  3/
      DATA SPROP_D(  5,  -1,   4)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   4),IFR=1,2)/  1,  5/
      DATA TPRID_D(  5,  -2,   4)/       2/
      DATA (IFOREST_D(  5, IFR, -3,   4),IFR=1,2)/ -2, -1/
C     Diagram 5 for nFKSprocess 5
      DATA MAPCONFIG_D(  5,   5)/   5/
      DATA (IFOREST_D(  5, IFR, -1,   5),IFR=1,2)/  4,  3/
      DATA SPROP_D(  5,  -1,   5)/      21/
      DATA (IFOREST_D(  5, IFR, -2,   5),IFR=1,2)/  1, -1/
      DATA TPRID_D(  5,  -2,   5)/      21/
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
      DATA SPROP_D(  6,  -2,   1)/       2/
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
      DATA TPRID_D(  6,  -2,   5)/       2/
      DATA (IFOREST_D(  6, IFR, -3,   5),IFR=1,2)/ -2,  5/
C     Number of configs for nFKSprocess 6
      DATA MAPCONFIG_D(  6,0)/   5/
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
      PMASS_D (  1, -1,   4) = ZERO
      PWIDTH_D(  1, -1,   4) = ZERO
      POW_D   (  1, -1,   4) = 2
      PMASS_D (  1, -2,   4) = ZERO
      PWIDTH_D(  1, -2,   4) = ZERO
      POW_D   (  1, -2,   4) = 1
      PMASS_D (  1, -1,   5) = ZERO
      PWIDTH_D(  1, -1,   5) = ZERO
      POW_D   (  1, -1,   5) = 2
      PMASS_D (  1, -2,   5) = ZERO
      PWIDTH_D(  1, -2,   5) = ZERO
      POW_D   (  1, -2,   5) = 1
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
      PMASS_D (  2, -1,   4) = ZERO
      PWIDTH_D(  2, -1,   4) = ZERO
      POW_D   (  2, -1,   4) = 2
      PMASS_D (  2, -2,   4) = ZERO
      PWIDTH_D(  2, -2,   4) = ZERO
      POW_D   (  2, -2,   4) = 1
      PMASS_D (  2, -1,   5) = ZERO
      PWIDTH_D(  2, -1,   5) = ZERO
      POW_D   (  2, -1,   5) = 2
      PMASS_D (  2, -2,   5) = ZERO
      PWIDTH_D(  2, -2,   5) = ZERO
      POW_D   (  2, -2,   5) = 1
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
      PMASS_D (  3, -1,   4) = ZERO
      PWIDTH_D(  3, -1,   4) = ZERO
      POW_D   (  3, -1,   4) = 2
      PMASS_D (  3, -2,   4) = ZERO
      PWIDTH_D(  3, -2,   4) = ZERO
      POW_D   (  3, -2,   4) = 1
      PMASS_D (  3, -1,   5) = ZERO
      PWIDTH_D(  3, -1,   5) = ZERO
      POW_D   (  3, -1,   5) = 2
      PMASS_D (  3, -2,   5) = ZERO
      PWIDTH_D(  3, -2,   5) = ZERO
      POW_D   (  3, -2,   5) = 1
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
      PMASS_D (  4, -1,   4) = ZERO
      PWIDTH_D(  4, -1,   4) = ZERO
      POW_D   (  4, -1,   4) = 2
      PMASS_D (  4, -2,   4) = ZERO
      PWIDTH_D(  4, -2,   4) = ZERO
      POW_D   (  4, -2,   4) = 1
      PMASS_D (  4, -1,   5) = ZERO
      PWIDTH_D(  4, -1,   5) = ZERO
      POW_D   (  4, -1,   5) = 2
      PMASS_D (  4, -2,   5) = ZERO
      PWIDTH_D(  4, -2,   5) = ZERO
      POW_D   (  4, -2,   5) = 1
C     
      PMASS_D (  5, -1,   1) = ZERO
      PWIDTH_D(  5, -1,   1) = ZERO
      POW_D   (  5, -1,   1) = 2
      PMASS_D (  5, -2,   1) = ZERO
      PWIDTH_D(  5, -2,   1) = ZERO
      POW_D   (  5, -2,   1) = 1
      PMASS_D (  5, -1,   2) = ABS(MDL_MT)
      PWIDTH_D(  5, -1,   2) = ABS(MDL_WT)
      POW_D   (  5, -1,   2) = 1
      PMASS_D (  5, -2,   2) = ZERO
      PWIDTH_D(  5, -2,   2) = ZERO
      POW_D   (  5, -2,   2) = 2
      PMASS_D (  5, -1,   3) = ABS(MDL_MT)
      PWIDTH_D(  5, -1,   3) = ABS(MDL_WT)
      POW_D   (  5, -1,   3) = 1
      PMASS_D (  5, -2,   3) = ZERO
      PWIDTH_D(  5, -2,   3) = ZERO
      POW_D   (  5, -2,   3) = 2
      PMASS_D (  5, -1,   4) = ZERO
      PWIDTH_D(  5, -1,   4) = ZERO
      POW_D   (  5, -1,   4) = 2
      PMASS_D (  5, -2,   4) = ZERO
      PWIDTH_D(  5, -2,   4) = ZERO
      POW_D   (  5, -2,   4) = 1
      PMASS_D (  5, -1,   5) = ZERO
      PWIDTH_D(  5, -1,   5) = ZERO
      POW_D   (  5, -1,   5) = 2
      PMASS_D (  5, -2,   5) = ZERO
      PWIDTH_D(  5, -2,   5) = ZERO
      POW_D   (  5, -2,   5) = 2
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

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     
C     NB: VECSIZE_MEMMAX is defined in vector.inc
C     NB: vector.inc must be included before coupl.inc
C     

      DOUBLE PRECISION G, ALL_G
      COMMON/STRONG/ G, ALL_G

      DOUBLE COMPLEX GAL(2)
      COMMON/WEAK/ GAL

      DOUBLE PRECISION MU_R, ALL_MU_R
      COMMON/RSCALE/ MU_R, ALL_MU_R

      DOUBLE PRECISION NF
      PARAMETER(NF=4D0)
      DOUBLE PRECISION NL
      PARAMETER(NL=2D0)

      DOUBLE PRECISION MDL_MB,MDL_MH,MDL_MT,MDL_MTA,MDL_MW,MDL_MZ

      COMMON/MASSES/ MDL_MB,MDL_MH,MDL_MT,MDL_MTA,MDL_MW,MDL_MZ


      DOUBLE PRECISION MDL_WH,MDL_WT,MDL_WW,MDL_WZ

      COMMON/WIDTHS/ MDL_WH,MDL_WT,MDL_WW,MDL_WZ


      DOUBLE COMPLEX, TARGET :: GC_37, GC_51, GC_58, GC_60, GC_68,
     $  GC_72, GC_81, GC_100, GC_FFV_0, GC_FFV_1, GC_FFV_2, GC_FFV_3

      TYPE(FLV_COUPLING) FLV_3, FLV_4, FLV_5, FLV_6

      DOUBLE COMPLEX, TARGET :: GC_10, GC_11

      COMMON/COUPLINGS/ GC_10, GC_11, GC_37, GC_51, GC_58, GC_60,
     $  GC_68, GC_72, GC_81, GC_100, GC_FFV_0, GC_FFV_1, GC_FFV_2,
     $  GC_FFV_3, FLV_3, FLV_4, FLV_5, FLV_6


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      REAL*16 MP__G
      COMMON/MP_STRONG/ MP__G

      COMPLEX*32 MP__GAL(2)
      COMMON/MP_WEAK/ MP__GAL

      COMPLEX*32 MP__MU_R
      COMMON/MP_RSCALE/ MP__MU_R


      REAL*16 MP__MDL_MT,MP__MDL_MW,MP__MDL_MZ,MP__MDL_MTA,MP__MDL_MB
     $ ,MP__MDL_MH

      COMMON/MP_MASSES/ MP__MDL_MT,MP__MDL_MW,MP__MDL_MZ,MP__MDL_MTA
     $ ,MP__MDL_MB,MP__MDL_MH


      REAL*16 MP__MDL_WZ,MP__MDL_WH,MP__MDL_WW,MP__MDL_WT

      COMMON/MP_WIDTHS/ MP__MDL_WZ,MP__MDL_WH,MP__MDL_WW,MP__MDL_WT


      COMPLEX*32 MP__R2_GGHB,MP__R2_GGHT,MP__R2_GGHHB,MP__R2_GGHHT
     $ ,MP__GC_5,MP__GC_30,MP__GC_33,MP__GC_37

      COMMON/MP_COUPLINGS/ MP__R2_GGHB,MP__R2_GGHT,MP__R2_GGHHB
     $ ,MP__R2_GGHHT,MP__GC_5,MP__GC_30,MP__GC_33,MP__GC_37



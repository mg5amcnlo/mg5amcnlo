ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      REAL*16 G
      COMMON/MP_STRONG/ G

      COMPLEX*32 GAL(2)
      COMMON/MP_WEAK/ GAL

      COMPLEX*32 MU_R
      COMMON/MP_RSCALE/ MU_R


      REAL*16 MDL_MT,MDL_MW,MDL_MZ,MDL_MTA,MDL_MB,MDL_MH

      COMMON/MP_MASSES/ MDL_MT,MDL_MW,MDL_MZ,MDL_MTA,MDL_MB,MDL_MH


      REAL*16 MDL_WZ,MDL_WH,MDL_WW,MDL_WT

      COMMON/MP_WIDTHS/ MDL_WZ,MDL_WH,MDL_WW,MDL_WT


      COMPLEX*32 R2_GGHB,R2_GGHT,R2_GGHHB,R2_GGHHT,GC_5,GC_30,GC_33
     $ ,GC_37

      COMMON/MP_COUPLINGS/ R2_GGHB,R2_GGHT,R2_GGHHB,R2_GGHHT,GC_5
     $ ,GC_30,GC_33,GC_37



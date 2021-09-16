ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      REAL*16 G
      COMMON/MP_STRONG/ G

      COMPLEX*32 GAL(2)
      COMMON/MP_WEAK/ GAL

      COMPLEX*32 MU_R
      COMMON/MP_RSCALE/ MU_R


      REAL*16 MDL_MT,MDL_MH,MDL_MZ,MDL_MB,MDL_MW,MDL_MTA

      COMMON/MP_MASSES/ MDL_MT,MDL_MH,MDL_MZ,MDL_MB,MDL_MW,MDL_MTA


      REAL*16 MDL_WW,MDL_WH,MDL_WZ,MDL_WT

      COMMON/MP_WIDTHS/ MDL_WW,MDL_WH,MDL_WZ,MDL_WT


      COMPLEX*32 GC_5,R2_GGHB,R2_GGHT,R2_GGHHB,R2_GGHHT,GC_30,GC_33
     $ ,GC_37

      COMMON/MP_COUPLINGS/ GC_5,R2_GGHB,R2_GGHT,R2_GGHHB,R2_GGHHT
     $ ,GC_30,GC_33,GC_37



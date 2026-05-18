ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      REAL*16 G
      COMMON/MP_STRONG/ G

      COMPLEX*32 GAL(2)
      COMMON/MP_WEAK/ GAL

      COMPLEX*32 MU_R
      COMMON/MP_RSCALE/ MU_R


      REAL*16 MDL_MB,MDL_MH,MDL_MT,MDL_MTA,MDL_MW,MDL_MZ

      COMMON/MP_MASSES/ MDL_MB,MDL_MH,MDL_MT,MDL_MTA,MDL_MW,MDL_MZ


      REAL*16 MDL_WH,MDL_WT,MDL_WW,MDL_WZ

      COMMON/MP_WIDTHS/ MDL_WH,MDL_WT,MDL_WW,MDL_WZ


      COMPLEX*32 GC_2,GC_3,GC_21,GC_22,GC_23,GC_24

      COMPLEX*32 GC_5,UVWFCT_B_0_1EPS,R2_UUA,R2_DDZ_V3,R2_UUZ_V2
     $ ,UVWFCT_T_0

      COMMON/MP_COUPLINGS/ GC_5,UVWFCT_B_0_1EPS,R2_UUA,R2_DDZ_V3
     $ ,R2_UUZ_V2,UVWFCT_T_0,GC_2,GC_3,GC_21,GC_22,GC_23,GC_24



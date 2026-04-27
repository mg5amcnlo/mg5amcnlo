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


      DOUBLE COMPLEX GC_11

      DOUBLE COMPLEX GC_4, GC_5, R2_GQQ, R2_QQQ, UV_GQQG_1EPS,
     $  UV_GQQB_1EPS, UVWFCT_G_1_1EPS, R2_BXTW, UV_GQQB, UV_GQQT,
     $  UVWFCT_G_1, UVWFCT_G_2

      COMMON/COUPLINGS/ GC_4, GC_5, R2_GQQ, R2_QQQ, UV_GQQG_1EPS,
     $  UV_GQQB_1EPS, UVWFCT_G_1_1EPS, R2_BXTW, UV_GQQB, UV_GQQT,
     $  UVWFCT_G_1, UVWFCT_G_2, GC_11


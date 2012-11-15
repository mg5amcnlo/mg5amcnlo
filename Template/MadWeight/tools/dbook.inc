C
C     Global 
C
      integer NHistoVar,NHistobyvar,NHistobin
C
C========================== PART TO DEFINE ==========================================
      parameter (NHistoVar=1)      ! Number of differential weight to compute
      parameter (NHistobyvar=200)  ! This number must 200
      parameter (NHistobin=1)     ! Maximum binning in a histogram 
C========================= END PART TO DEFINE =======================================

	 
      REAL*8 HIST(-1*Nhistobyvar:Nhistobyvar,NHistobin,NHistovar)
     &      ,XHIS(-1*Nhistobyvar:Nhistobyvar,NHistobin,NHistovar)
     &      ,HDEL(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &      ,HMIN(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &      ,HMAX(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &      ,USCORE(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &      ,OSCORE(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &      ,HAVG(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &      ,HINT(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &      ,HSIG(-1*Nhistobyvar:Nhistobyvar,NHistovar)

      INTEGER NBIN(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &       ,IHIS(-1*Nhistobyvar:Nhistobyvar,NHistobin,NHistovar)
     &       ,IUSCORE(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &       ,IOSCORE(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &       ,IENT(-1*Nhistobyvar:Nhistobyvar,NHistovar)
     &       ,NHIST(NHistoVar)

      CHARACTER*3  BOOK(-1*Nhistobyvar:Nhistobyvar,NHistovar)
      CHARACTER*20 TITLE(-1*Nhistobyvar:Nhistobyvar,NHistovar)
c    
      COMMON/HISTO/HIST,XHIS,HDEL,HMIN,HMAX,USCORE,OSCORE
     &            ,HAVG,HINT,HSIG,NBIN,IHIS,IUSCORE,IOSCORE,IENT,NHIST
     &            ,BOOK,TITLE

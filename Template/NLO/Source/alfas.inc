c***********************************************************************
c     this files contains the common blocks for the 
c     the alpha_s settings
c
c     asmz   = alpha_s(Mz) is set based on the pdf chosen in setcuts.f
c     nloop  = order of the running of alpha_s based on the pdf chosen  
c***********************************************************************
      integer nloop
      double precision asmz
      common/a_block/asmz,nloop


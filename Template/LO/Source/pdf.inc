c***********************************************************************
c     this files contains the common blocks for the 
c     pdf and the alpha_s settings
c
c     pdlabel= string identifying the pdf    
c     asmz   = alpha_s(Mz) is set based on the pdf chosen in setcuts.f
c     nloop  = order of the running of alpha_s based on the pdf chosen  
c***********************************************************************
      double precision asmz
      common/couple/asmz
      integer nloop
      common/running/nloop
      character*7 pdlabel,epa_label
      integer lhaid
      common/to_pdf/lhaid,pdlabel,epa_label

	 	

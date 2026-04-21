
c***********************************************************************
c     this files contains the common blocks for the 
c     pdf and the alpha_s settings
c
c     pdlabel= string identifying the pdf    
c***********************************************************************
      character*7 pdlabel,epa_label
      character*7 pdsublabel(2)
      integer lhaid, pdfscheme
      common/to_pdf/lhaid,pdfscheme,pdlabel,epa_label,pdsublabel
 	

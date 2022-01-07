      subroutine DLUM_%(N_me)d(lum)                                
C ****************************************************                
%(info_lines)s
C     RETURNS PARTON LUMINOSITIES FOR MADFKS                             
C
%(process_lines)s
C
C ****************************************************                
      IMPLICIT NONE                                                   
C                                                                     
C CONSTANTS                                                           
C                                                                     
      include 'genps.inc'                                            
      include "nexternal.inc"
      DOUBLE PRECISION       CONV                                     
      PARAMETER (CONV=389379660D0)  !CONV TO PICOBARNS             
C                                                                     
C ARGUMENTS                                                           
C                                                                     
      DOUBLE PRECISION lum                         
C                                                                     
C LOCAL VARIABLES                                                     
C                                                                     
      INTEGER I, ICROSS,LP                                      
%(pdf_vars)s
C                                                                     
C EXTERNAL FUNCTIONS                                                  
C                                                                     
      DOUBLE PRECISION PDG2PDF                          
C                                                                     
C GLOBAL VARIABLES                                                    
C                                                                     
      INTEGER              IPROC                                      
      DOUBLE PRECISION PD(0:MAXPROC)                                  
      COMMON /SubProc/ PD, IPROC                                      
      include "coupl.inc"                                             
      include "run.inc"                                               
      integer imirror
      common/cmirror/imirror
C
C     STUFF FOR DRESSED EE COLLISIONS
C     
      include 'eepdf.inc'
      double precision ee_comp_prod
%(ee_comp_vars)s
      integer i_ee
c
c
c
c Common blocks
      character*7         pdlabel,epa_label
      integer       lhaid
      common/to_pdf/lhaid,pdlabel,epa_label
C                                                                     
C DATA                                                                
C                                                                     
%(pdf_data)s
      DATA ICROSS/1/                                          
C ----------                                                          
C BEGIN CODE                                                          
C ----------                                                          
      lum = 0D0                                                      
%(pdf_lines)s
      do i=1,IPROC
        if (nincoming.eq.2) then
           lum = lum + pd(i) * conv 
        else
           lum = lum + pd(i)
        endif
      enddo
      return
      END                                                             

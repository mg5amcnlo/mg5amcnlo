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
      DOUBLE PRECISION PP(0:3,NEXTERNAL), lum                         
C                                                                     
C LOCAL VARIABLES                                                     
C                                                                     
      INTEGER I, ICROSS,ITYPE,LP                                      
      DOUBLE PRECISION P1(0:3,NEXTERNAL)                              
%(pdf_vars)s
      DOUBLE PRECISION XPQ(-7:7)                                      
C                                                                     
C EXTERNAL FUNCTIONS                                                  
C                                                                     
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF                          
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
C DATA                                                                
C                                                                     
%(pdf_data)s
      DATA ICROSS/1/                                          
C ----------                                                          
C BEGIN CODE                                                          
C ----------                                                          
      lum = 0D0                                                      
      if (imirror.eq.2) then
%(pdf_lines_mirr)s
      else
%(pdf_lines)s
      endif
      do i=1,IPROC
        if (nincoming.eq.2) then
           lum = lum + pd(i) * conv 
        else
           lum = lum + pd(i)
        endif
      enddo
      return
      END                                                             

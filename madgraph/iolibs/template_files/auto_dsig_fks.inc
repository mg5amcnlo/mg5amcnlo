      DOUBLE PRECISION FUNCTION DLUM()                                
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
      PARAMETER (CONV=389379.66*1000)  !CONV TO PICOBARNS             
      REAL*8     PI                                                   
      PARAMETER (PI=3.1415926d0)                                      
C                                                                     
C ARGUMENTS                                                           
C                                                                     
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT                         
C                                                                     
C LOCAL VARIABLES                                                     
C                                                                     
      INTEGER I, ICROSS,ITYPE,LP                                      
      DOUBLE PRECISION P1(0:3,NEXTERNAL)                              
      DOUBLE PRECISION u1,ub1,d1,db1,c1,cb1,s1,sb1,b1,bb1             
      DOUBLE PRECISION u2,ub2,d2,db2,c2,cb2,s2,sb2,b2,bb2             
      DOUBLE PRECISION g1,g2                                          
      DOUBLE PRECISION a1,a2                                          
      DOUBLE PRECISION XPQ(-7:7)                                      
      DOUBLE PRECISION rwgt,DSIGUU                                    
      double precision xi_i_fks,y_ij_fks                              
C                                                                     
C EXTERNAL FUNCTIONS                                                  
C                                                                     
C      LOGICAL PASSCUTS                                               
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF                          
C                                                                     
C GLOBAL VARIABLES                                                    
C                                                                     
      INTEGER              IPROC                                      
      DOUBLE PRECISION PD(0:MAXPROC)                                  
      COMMON /SubProc/ PD, IPROC                                      
      include "coupl.inc"                                             
      include "run.inc"                                               
C                                                                     
C DATA                                                                
C                                                                     
      DATA u1,ub1,d1,db1,c1,cb1,s1,sb1,b1,bb1/10*1d0/                 
      DATA u2,ub2,d2,db2,c2,cb2,s2,sb2,b2,bb2/10*1d0/                 
      DATA a1,g1/2*1d0/                                               
      DATA a2,g2/2*1d0/                                               
      DATA IPROC,ICROSS/1,1/                                          
C ----------                                                          
C BEGIN CODE                                                          
C ----------                                                          
      DLUM = 0D0                                                      
C      IF (PASSCUTS(PP,rwgt)) THEN                                    
%(pdf_lines)s
         DSIGUU=1d0                                                   
         If (dsiguu .lt. 1d199) then                                  
         dlum = pd(IPROC) * conv * dsiguu                             
         else                                                         
             write(*,*) "Error in matrix element"                     
             dsiguu=0d0                                               
            dlum = 0d0                                                
         endif                                                        
      END                                                             

*****************************************************************
* This file contains the MadGraph routines that calculate the   *
* matrix elements squared (color and spin sums/averages and     *
* bose factors not included) for generic decays:                * 
*                                                               *
*     SUB NAME                 PROCESSES                        * 
*                                                               *
*---------------------------------------------------------------*
*     emme_f3f          :  f    >  f'  w > f' k  k'~            *
*               examples:  t    >  b   vl  l+                   *
*                          t    >  b   u   d~                   *
*                          tau- > vt   l-  v~                   *
*                                                               *
*---------------------------------------------------------------*
*     emme_fx3f         :  f~   >  f'~ w > f'~ k k'~            *
*               examples:  t~   >  b~  l-  v~                   *
*                          t~   >  b~  d   u~                   *
*                          tau+ > vt~  vl  l+                   *
*                                                               *
*---------------------------------------------------------------*
*     emme_vff          :  v    >  f   f~                       *
*               examples:  z    >  e-  e+                       *
*                          w+   >  lv  l+                       *
*                                                               *
*---------------------------------------------------------------*
*     emme_hff          :  h    >  f  f~                        *
*               examples:  h    >  b  b~                        *
*                          h   >  mu  mu+                       *
*                                                               *
*---------------------------------------------------------------*
*     emme_hvv          :  h    >  v  v                         *
*               examples:  h    >  a  a                         *
*                          h   >   w- w+                        *
*                                                               *
*---------------------------------------------------------------*
*     emme_h4f          :  h    >  v  v >  f  f~  f' f~'        *
*               examples:  h    >  z  z >  e- e+  u  u~         *
*                                                               *
*---------------------------------------------------------------*
*     emme_ffs          :  f    >  f'  s                        *
*               examples:  tau- > vt  pi-                       *
*                                                               *
*---------------------------------------------------------------*
*     emme_fxfs         :  f~   > f~'  s                        *
*               examples: tau+  >vt~  pi+                       *
*                                                               *   
*---------------------------------------------------------------*
*     emme_ffv          :  f    > f'   v                        *
*               examples:  tau- > vt  rho-                      *
*                          t    > b    w+                       *
*                                                               *   
*---------------------------------------------------------------*
*     emme_fxfv         :  f~   > f~'  v                        *
*               examples:  tau+ > vt~ rho+                      *
*                          t~   > b~    w-                      *
*                                                               *
*****************************************************************



      SUBROUTINE emme_f3f(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1--->----*--->---2
C          *--->---3
C          *---<---4
C
C t    >  b  vl l+ 
C t    >  b  u  d~ 
C tau- > vt  l- vl~
c
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=5) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
      integer i
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------


      CALL IXXXXX(P(0,1   ),M1 ,NHEL(1   ),+1,W(1,1   ))       
      CALL OXXXXX(P(0,2   ),M2 ,NHEL(2   ),+1,W(1,2   ))       
      CALL OXXXXX(P(0,3   ),M3 ,NHEL(3   ),+1,W(1,3   ))        
      CALL IXXXXX(P(0,4   ),M4 ,NHEL(4   ),-1,W(1,4   ))        
c
      CALL JIOXXX(W(1,1   ),W(1,2   ),GXX ,MV  ,GV  ,W(1,5   ))    
      CALL IOVXXX(W(1,4   ),W(1,3   ),W(1,5   ),GXX ,AMP(1   ))            

      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_fx3f(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1---<----*---<---2
C          *--->---3
C          *---<---4
C
C t~   >  b~  l- vl~ 
C t~   >  b~  d  u~  
C tau+ > vt~  vl l+ 
c
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=5) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------

      CALL OXXXXX(P(0,1   ),M1 ,NHEL(1   ),-1,W(1,1   ))
      CALL IXXXXX(P(0,2   ),M2 ,NHEL(2   ),-1,W(1,2   ))
      CALL OXXXXX(P(0,3   ),M3 ,NHEL(3   ),+1,W(1,3   ))
      CALL IXXXXX(P(0,4   ),M4 ,NHEL(4   ),-1,W(1,4   ))
c
      CALL JIOXXX(W(1,2   ),W(1,1   ),GXX ,MV,GV ,W(1,5   ))
      CALL IOVXXX(W(1,4   ),W(1,3   ),W(1,5   ),GXX ,AMP(1   ))
      
      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_v(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1*********---<---2
C          *--->---3
C
c  v > f f~
c   
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------

      CALL VXXXXX(P(0,1)  ,M1   ,NHEL(1  ),-1,W(1,1)  )
      CALL IXXXXX(P(0,2)  ,M2   ,NHEL(2  ),-1,W(1,2)  )
      CALL OXXXXX(P(0,3)  ,M3   ,NHEL(3  ), 1,W(1,3)  )
c
      CALL IOVXXX(W(1,2)  ,W(1,3)  ,W(1,1)  ,GXX,AMP(1  ))
      
      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_vff(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1*********--->---2
C          *---<---3
C
c   v    >  f f~        
c
c   
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
      integer i
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------


      CALL VXXXXX(P(0,1)  ,M1   ,NHEL(1  ),-1,W(1,1)  )
      CALL OXXXXX(P(0,2)  ,M2   ,NHEL(2  ),+1,W(1,2)  )
      CALL IXXXXX(P(0,3)  ,M3   ,NHEL(3  ),-1,W(1,3)  )
c
      CALL IOVXXX(W(1,3)  ,W(1,2), W(1,1),GXX,AMP(1  ))
      

      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_hff(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1*********--->---2
C          *---<---3
C
C  h    >  f f~ 
c   
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------


      CALL SXXXXX(P(0,1),                -1,W(1,1)  )
      CALL OXXXXX(P(0,2),M2   ,NHEL(2  ),+1,W(1,2)  )
      CALL IXXXXX(P(0,3),M3   ,NHEL(3  ),-1,W(1,3)  )
c
      CALL IOSXXX(W(1,3)  ,W(1,2)  ,W(1,1)  ,GXX, AMP(1  ))
      
      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_hvv(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1--------********2
C          ********3
C
c h    >  v  v
c   
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------

      CALL SXXXXX(P(0,1),                -1,W(1,1)  )
      CALL VXXXXX(P(0,2),M2   ,NHEL(2  ),+1,W(1,2)  )
      CALL VXXXXX(P(0,3),M3   ,NHEL(3  ),+1,W(1,3)  )
c
      CALL VVSXXX(W(1,3)  ,W(1,2)  ,W(1,1)  ,GX , AMP(1  ))
      
      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_h4f(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1*********--->---2
C          *---<---3
C          *--->---4
C          *---<---5
C
C h    >  v  v >  f  f~  f' f~'         
C  
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=7) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'
      
      integer i
      real*8 dot
C                                                                          
C ----------
C BEGIN CODE
C ----------

      CALL SXXXXX(P(0,1   ),-1,W(1,1   ))
      CALL OXXXXX(P(0,2   ),M2,NHEL(2   ),+1,W(1,2   ))
      CALL IXXXXX(P(0,3   ),M3,NHEL(3   ),-1,W(1,3   ))
      CALL OXXXXX(P(0,4   ),M4,NHEL(4   ),+1,W(1,4   ))
      CALL IXXXXX(P(0,5   ),M5,NHEL(5   ),-1,W(1,5   ))
c
      CALL JIOXXX(W(1,3   ),W(1,2   ),GXX ,MV  ,GV ,W(1,6   ))
      CALL JIOXXX(W(1,5   ),W(1,4   ),GXX1,MV  ,GV ,W(1,7   ))
      CALL VVSXXX(W(1,6   ),W(1,7   ),W(1,1   ),GX ,AMP(1))
c
      EMMESQ = amp(1)*dconjg(amp(1))

      RETURN
      END



      SUBROUTINE emme_ffv(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1--->----*--->---2
C          ********3
C
C f > f'  v 
c
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'
C
C ----------
C BEGIN CODE
C ----------

C
      CALL IXXXXX(P(0,1   ),M1 ,NHEL(1   ),+1,W(1,1   ))       
      CALL OXXXXX(P(0,2   ),M2 ,NHEL(2   ),+1,W(1,2   ))       
      CALL VXXXXX(P(0,3   ),M3 ,NHEL(3  ) ,+1,W(1,3   ))
C
      CALL IOVXXX(W(1,1   ),W(1,2   ),W(1,3   ),GXX ,AMP(1   ))            
      
      
      EMMESQ = amp(1)*dconjg(amp(1))
      if(emmesq.lt.1d-15) emmesq=0d0
      return
      end


      SUBROUTINE emme_fxfv(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1---<----*---<---2
C          ********3
C
C f~ > f~'  v 
c
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------

      CALL OXXXXX(P(0,1   ),M1 ,NHEL(1   ),-1,W(1,1   ))       
      CALL IXXXXX(P(0,2   ),M2 ,NHEL(2   ),-1,W(1,2   ))       
      CALL VXXXXX(P(0,3   ),M3 ,NHEL(3  ) ,+1,W(1,3   ))
c
      CALL IOVXXX(W(1,2   ),W(1,1   ),W(1,3   ),GXX ,AMP(1   ))            
      
      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_ffs(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1--->----*--->---2
C          ********3
C
C f > f'  s 
c
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'

C ----------
C BEGIN CODE
C ----------

      CALL IXXXXX(P(0,1   ),M1 ,NHEL(1   ),+1,W(1,1   ))       
      CALL OXXXXX(P(0,2   ),M2 ,NHEL(2   ),+1,W(1,2   ))       
      CALL SXXXXX(P(0,3   ),+1,W(1,3   ))
c
      CALL IOSXXX(W(1,1   ),W(1,2   ),W(1,3   ),GXX ,AMP(1   ))            
      
      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end


      SUBROUTINE emme_fxfs(P,NHEL,EMMESQ)
C---------------------------------------------
C  Function for the processes
C
C 1---<----*---<---2
C          ********3
C
C f~ > f~' s 
c
c NO COLOR/SPIN SUMS OR AVERAGES ARE INCLUDED
c----------------------------------------------          
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5) 
      INTEGER    NWAVEFUNCS     
      PARAMETER (NWAVEFUNCS=3) 
C  
C ARGUMENTS 
C  
      REAL*8 EMMESQ
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(1)
      COMPLEX*16 W(6,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include 'decay.inc'


C ----------
C BEGIN CODE
C ----------

      CALL OXXXXX(P(0,1   ),M1 ,NHEL(1   ),-1,W(1,1   ))       
      CALL IXXXXX(P(0,2   ),M2 ,NHEL(2   ),-1,W(1,2   ))       
      CALL SXXXXX(P(0,3   ),+1,W(1,3   ))
c
      CALL IOSXXX(W(1,2   ),W(1,1   ),W(1,3   ),GXX ,AMP(1   ))            
      
      EMMESQ = amp(1)*dconjg(amp(1))

      return
      end



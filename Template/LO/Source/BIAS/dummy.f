C ************************************************************
C Source for the library implementing a dummt bias function 
C always returns one
C ************************************************************
      
      subroutine bias_wgt(p, bias_weight)
          implicit none
C
C Parameters
C
          include '../nexternal.inc'
C
C Arguments
C
          double precision p(0:3,nexternal)
          double precision bias_weight
C
C local variables
C
C
C Global variables
C
C --------------------
C BEGIN IMPLEMENTATION
C --------------------

          bias_weight = 1.0d0

      end subroutine bias_wgt

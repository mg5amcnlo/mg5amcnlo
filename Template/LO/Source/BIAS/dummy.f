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
C Mandatory common block to be defined in bias modules
C
          double precision stored_bias_weight
          logical impact_xsec
          data stored_bias_weight/1.0d0/
C         Not impacting the xsec since the bias is 1.0. Therefore
C         bias_wgt will not be written in the lhe event file.
          data impact_xsec/.True./
          common/bias/stored_bias_weight,impact_xsec
C --------------------
C BEGIN IMPLEMENTATION
C --------------------

          bias_weight = 1.0d0

      end subroutine bias_wgt

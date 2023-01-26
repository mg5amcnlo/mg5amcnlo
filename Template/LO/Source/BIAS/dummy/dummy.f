C ************************************************************
C Source for the library implementing a dummt bias function 
C always returns one
C ************************************************************
      
      subroutine bias_wgt(p, original_weight, bias_weight)
          implicit none
C
C Parameters
C
c          include '../../nexternal.inc'
C
C Arguments
C
          double precision p(*)
          double precision original_weight, bias_weight
C
C local variables
C
C
C Global variables
C
C Mandatory common block to be defined in bias modules
C
          double precision stored_bias_weight
          data stored_bias_weight/1.0d0/          
          logical impact_xsec, requires_full_event_info
C         Not impacting the xsec since the bias is 1.0. Therefore
C         bias_wgt will not be written in the lhe event file.
C         Setting it to .True. makes sure that it will not be written.
          data impact_xsec/.True./
C         Of course this module does not require the full event
C         information (color, resonances, helicities, etc..)
          data requires_full_event_info/.False./ 
          common/bias/stored_bias_weight,impact_xsec,
     &                requires_full_event_info

C --------------------
C BEGIN IMPLEMENTATION
C --------------------
c         new default call the dunction          
          call bias_wgt_custom(p, original_weight, bias_weight)
c          bias_weight = 1.0d0

      end subroutine bias_wgt

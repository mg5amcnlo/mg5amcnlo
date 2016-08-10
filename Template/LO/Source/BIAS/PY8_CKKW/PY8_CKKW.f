C ************************************************************
C Source for the library implementing a bias function that 
C implements the Sudakov weight in CKKW directly from Pythia8 
C ************************************************************
C
C The following lines are read by MG5aMC to set what are the 
C relevant parameters for this bias module.
C
C  parameters = {'arg1': 10.0,
C                'arg2': 20.0}
C

      subroutine bias_wgt(p, bias_weight)
          implicit none
C
C Parameters
C
          include '../../maxparticles.inc'          
          include '../../nexternal.inc'

C
C Arguments
C
          double precision p(0:3,nexternal)
          double precision bias_weight
C
C local variables
C
          integer i
c
c local variables defined in the run_card
c
          double precision arg1, arg2 
C
C Global variables
C
C
C Mandatory common block to be defined in bias modules
C
          double precision stored_bias_weight
          data stored_bias_weight/1.0d0/          
          logical impact_xsec, requires_full_event_info
C         We only want to bias distributions, but not impact the xsec. 
          data impact_xsec/.True./
C         Pythia8 will need the full information for the event
C          (color, resonances, helicities, etc..)
          data requires_full_event_info/.True./ 
          common/bias/stored_bias_weight,impact_xsec,
     &                requires_full_event_info
C
C Accessingt the details of the event
C
          include '../../run_config.inc'
          include '../../lhe_event_infos.inc'
C
C Access the value of the run parameters in run_card
C
          include '../../run.inc'
          include '../../cuts.inc'
C
C Read the definition of the bias parameter from the run_card    
C
          include '../bias.inc'

C --------------------
C BEGIN IMPLEMENTATION
C --------------------

          bias_weight = 1.0d0

          return

      end subroutine bias_wgt

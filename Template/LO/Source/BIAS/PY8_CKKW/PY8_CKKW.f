C ************************************************************
C Source for the library implementing a bias function that 
C implements the Sudakov weight in CKKW directly from Pythia8 
C ************************************************************
C
C The following lines are read by MG5aMC to set what are the 
C relevant parameters for this bias module.
C
C  parameters = {'BeamA': 0.0,
C                'BeamB': 0.0}
C

      subroutine bias_wgt(p, original_weight, bias_weight)
          implicit none
C
C Parameters
C
          include '../../maxparticles.inc'          
          include '../../nexternal.inc'
C
C Accessingt the details of the event
C
          include '../../run_config.inc'
          include '../../lhe_event_infos.inc'
C
C Arguments
C
          double precision p(0:3,nexternal)
          double precision original_weight, bias_weight
C
C local variables
C
          integer lpp_to_beam(-1:3)
          data lpp_to_beam/-2212,0,2212,2212,11/
c
c local variables defined in the run_card
c
c         Bias module arguments
          double precision BeamA, BeamB

c         truly local variables
          integer i,j, n_initial
          double precision OutputBiasWeight
          double precision Pythia8eCM
          integer Pythia8nParticles
          double precision Pythia8p(5,npart)
          integer Pythia8BeamA
          integer Pythia8BeamB 
          character*(maxEventLength) Pythia8EvtRecord
          integer Pythia8Helicities(npart)
          integer Pythia8ColorOne(npart)
          integer Pythia8ColorTwo(npart)          
          integer Pythia8ID(npart)
          integer Pythia8Status(npart)
          integer Pythia8MotherOne(npart)
          integer Pythia8MotherTwo(npart)          
          integer Pythia8SubprocessGroup
          integer Pythia8MurScale
          integer Pythia8AlphaQCD
          integer Pythia8AlphaQED

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

C        Let's initialize the PY8 variables describing the event
         Pythia8eCM             = sqrt(4d0*ebeam(1)*ebeam(2))
         Pythia8EvtRecord       = event_record
         Pythia8SubprocessGroup = ngroup
         Pythia8MurScale        = sscale
         Pythia8AlphaQCD        = aaqcd
         Pythia8AlphaQED        = aaqed
         Pythia8nParticles      = npart
         do i=1,npart
           Pythia8ID(i)         = jpart(1,i)
           Pythia8MotherOne(i)  = jpart(2,i)
           Pythia8MotherTwo(i)  = jpart(3,i)
           Pythia8ColorOne(i)   = jpart(4,i)
           Pythia8ColorTwo(i)   = jpart(5,i)           
           Pythia8Status(i)     = jpart(6,i)
           Pythia8Helicities(i) = jpart(7,i)           
           do j=1,4
             Pythia8p(j,i)=pb(mod(j,4),i)
           enddo
           Pythia8p(5,npart)=pb(4,i)
         enddo
         Pythia8BeamA = lpp_to_beam(lpp(1))
         Pythia8BeamB = lpp_to_beam(lpp(2))
         n_initial = 0
         do i=1,npart
           if (Pythia8Status(i).eq.-1) then
             n_initial = n_initial + 1
             if (lpp(n_initial).eq.0) then
               if (n_initial.eq.1) then
                 Pythia8BeamA = Pythia8ID(i)
               elseif (n_initial.eq.2) then
                 Pythia8BeamB = Pythia8ID(i)
                 exit
               endif
             endif
             if (n_initial.eq.2) then
               exit
             endif
           endif
         enddo
C        Make sure to enforce the user-choice of beam if specified.
         if (idnint(BeamA).ne.0) then
           Pythia8BeamA = idnint(BeamA)
         endif
         if (idnint(BeamB).ne.0) then
           Pythia8BeamB = idnint(BeamB) 
         endif

C        Call PY8 to derive the bias weight.
         call py8_bias_weight( Pythia8eCM,
     &                         Pythia8BeamA,
     &                         Pythia8BeamB,
     &                         Pythia8EvtRecord,
     &                         Pythia8p,
     &                         Pythia8nParticles,
     &                         Pythia8MurScale,
     &                         Pythia8AlphaQCD,
     &                         Pythia8AlphaQED,
     &                         Pythia8ID,
     &                         Pythia8MotherOne,
     &                         Pythia8MotherTwo,
     &                         Pythia8ColorOne,
     &                         Pythia8ColorTwo,
     &                         Pythia8Status,
     &                         Pythia8Helicities,
     &                         OutputBiasWeight    )
 
          bias_weight = OutputBiasWeight
            
          return

      end subroutine bias_wgt

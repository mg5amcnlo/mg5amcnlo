
! 1) Settings used in the main program.
Main:numberOfEvents = 20000    ! Number of events in the LHE file
Main:spareMode1 = 20000            ! Number of events to be showered
Main:spareWord1 = sum         ! Event weights are normalized to sum
                                      ! or average to the cross section
Main:timesAllowErrors = 2001  ! Number of allowed errors

! 2) Settings related to output in init(), next(), amd stat().
Init:showChangedParticleData = off    ! Shows all non-default particle settings
Next:numberShowEvent = 2     ! print event record n times
Stat:showProcessLevel = on            ! Process statistics
Stat:showErrors = on                  ! Error statistics
Check:epTolErr = 0.001                ! Momentum-conservation tolerance

! 3) Beam-parameter settings.
Beams:frameType = 4                ! LHE initialization
Beams:LHEF = events.lhe            ! Input LHE file

! 4) Switch on/off the key event-generation steps.
Beams:setProductionScalesFromLHEF = on ! read the radiator starting scales
Beams:setDipoleShowerStartingScalesFromLHEF = on ! retain separate dipole scales
PDF:pSet = LHAPDF6:NNPDF40_nlo_as_01180    ! PDF set used in showers and multiparton interactions
PartonLevel:MPI = off              ! Multiparton interactions
                                   ! ** USE on FOR REALISTIC SHOWER SIMULATION ** !!!
HadronLevel:Hadronize = off         ! Hadronisation
                                   ! ** USE on FOR REALISTIC SHOWER SIMULATION ** !!!

! 5) Final-state shower.
TimeShower:pTmaxMatch = 1          ! Use scalup (re-check)
TimeShower:pTmaxFudge = 1.         ! Factor changing the max scale
TimeShower:alphaSvalue = 0.118     ! Alpha_s(MZ) in final-state shower
TimeShower:alphaSorder = 1         ! Alpha_s running order in final-state shower
TimeShower:alphaEMorder = 0        ! Alpha_EM running order in final-state shower
TimeShower:dampenBeamRecoil = off  ! Dampens the effect of beam recoil
TimeShower:globalRecoil = on       ! All final-state particles recoil against the branching
TimeShower:nMaxGlobalRecoil = 1    ! Number of splittings with TimeShower:globalRecoil = on
TimeShower:globalRecoilMode = 2    ! Global recoil only for S events whose first emission is FSR
TimeShower:nMaxGlobalBranch = 1    ! Number of FSR splittings proposed with global recoil
TimeShower:nPartonsInBorn = -1     ! Number of Born QCD final-state partons (to treat H and S differently)
TimeShower:limitPTmaxGlobal = on   ! Limits pT < min(SCALUP,mDipole/2).
TimeShower:QEDshowerByQ = off     ! Prevent quarks from radiating photons
TimeShower:QEDshowerByL = off      ! Prevent leptons from radiating photons
TimeShower:QEDshowerByGamma = off  ! Prevent photons from branching
TimeShower:QEDshowerByOther = off  ! Include charged resonances, also stable tops
TimeShower:mMaxGamma = 10e0     ! Same conversion bound as MC subtraction
TimeShower:alphaSuseCMW = false    ! Use the CMW prescription in FSR
TimeShower:weightGluonToQuark = 1  ! Use normal Altarelli-Parisi kernels for g -> q qbar
TimeShower:recoilDeadCone = on     ! Gluon recoil weight included in the MC subtraction

! 6) Initial-state shower.
SpaceShower:pTmaxMatch = 1         ! Use scalup (re-check)
SpaceShower:pTmaxFudge = 1.        ! Factor changing the max scale
SpaceShower:alphaSvalue = 0.118    ! Alpha_s(MZ) in final-state shower
SpaceShower:alphaSorder = 1        ! Alpha_s running order in final-state shower
SpaceShower:alphaEMorder = 0       ! Alpha_EM running order in final-state shower
SpaceShower:QEDshowerByQ = off     ! Prevent quarks from radiating photons
SpaceShower:QEDshowerByL = off     ! Prevent leptons from radiating photons
SpaceShower:rapidityorder = off    ! Do not order branchings in rapidity
SpaceShower:alphaSuseCMW = false   ! Use the CMW prescription in ISR

! 7) Non-perturbative stuff.
BeamRemnants:primordialKT = off    ! off = no primordial kT
                                   ! ** USE on FOR REALISTIC SHOWER SIMULATION ** !!!

! 8) Particle features.
1:m0 = 0.33                    ! down mass
2:m0 = 0.33                    ! up mass
3:m0 = 0.5                    ! strange mass
4:m0 = 1.5                    ! charm mass
5:m0 = 4.8                    ! bottom mass
6:m0 = 173.0                      ! top mass
11:m0 = 0.000510999                   ! electron mass
13:m0 = 0.105658                  ! muon mass
15:m0 = 1.77682                 ! tauon mass
23:m0 = 91.188                     ! Z mass
24:m0 = 80.419002                     ! W mass
25:m0 = 125.0                   ! Higgs mass
6:mWidth = 1.4915                 ! top width
23:mWidth = 2.441404                ! Z width
24:mWidth = 2.0476                ! W width
25:mWidth = 0.00407              ! Higgs width

! 9) Multijet merging.
JetMatching:doFxFx = off          ! switch off FxFx


! 10) Matrix-element corrections.
SpaceShower:MEcorrections = off     ! no MECs for ISR
TimeShower:MEcorrections = on     ! MECs for FSR
TimeShower:MEextended = off     ! no extended MECs for FSR
TimeShower:MEafterFirst = off     ! no MECS MECs after first FSR
! 11) Decays.
111:mayDecay = false               ! stable pi0
25:maydecay = false                ! stable Higgs boson
 6:MAYDECAY = OFF


! Consume each first-emission LHE input once; stop on a failed shower.
Main:spareFlag1 = on

! Continue from the written endpoint and its remaining scale limits.
! Exhausted S events start below these one-step cutoffs.
Beams:setProductionScalesFromLHEF = on
Beams:setDipoleShowerStartingScalesFromLHEF = on
TimeShower:globalRecoil = off
TimeShower:pTmaxMatch = 1
SpaceShower:pTmaxMatch = 1
TimeShower:pTmaxFudge = 1.
SpaceShower:pTmaxFudge = 1.
TimeShower:limitPTmaxGlobal = on
TimeShower:pTmin = 5.00000000000000111e-01
SpaceShower:pTmin = 5.00000000000000111e-01
LesHouches:setQuarkMass = 0
LesHouches:setLeptonMass = 0
TimeShower:MEcorrections = off
SpaceShower:MEcorrections = off
TimeShower:MEafterFirst = off
SpaceShower:MEafterFirst = off

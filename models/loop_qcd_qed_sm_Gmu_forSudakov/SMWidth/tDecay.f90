MODULE tDecay
  USE ParamModule
  USE alfas_functions
CONTAINS
  FUNCTION SMtWidth(qcdord,qedord,finitemassin,lwidthin)
    USE Func_PSI
    IMPLICIT NONE
    ! (qcdord,qedord):
    ! (0,0): LO
    ! (1,0): NLO QCD
    ! (0,1): NLO QED
    ! (1,1): NLO QCD+QED
    INTEGER,INTENT(IN)::qcdord,qedord
    REAL(KIND(1d0))::SMtWidth
    REAL(KIND(1d0))::tWidth_LO,rr,CF
    LOGICAL,INTENT(IN),OPTIONAL::finitemassin,lwidthin
    LOGICAL::finitemass,lwidth
    REAL(KIND(1d0))::haha
    IF(.NOT.print_banner)THEN
       INCLUDE "banner.inc"
       print_banner=.TRUE.
    ENDIF
    IF(qcdord.LT.0.OR.qcdord.GT.1)THEN
       WRITE(*,*)"ERROR:Please specify the QCD corr. order to be 0 or 1"
       STOP
    ENDIF
    IF(qedord.LT.0.OR.qedord.GT.1)THEN
       WRITE(*,*)"ERROR:Please specify the QED corr. order to be 0 or 1"
       STOP
    ENDIF
    rr=Decay_MW/Decay_MT
    IF(rr.GT.1d0.OR.rr.LT.0d0)THEN
       WRITE(*,*)"ERROR:The mass of top or the W is wrong !"
       STOP
    ENDIF
    ! include finite b mass effect 
    IF(PRESENT(finitemassin))THEN
       finitemass=finitemassin
    ELSE
       finitemass=.FALSE. ! it should be turned off with loop_qcd_qed_sm correspondigly
    END IF
    ! include width effect
    IF(PRESENT(lwidthin))THEN
       lwidth=lwidthin
    ELSE
       lwidth=.FALSE. ! it should be turned off as well
    ENDIF
    ! t > W b, Eq.(6) in arXiv:1106.5483
    tWidth_LO=Decay_e**2/4d0/pi*Decay_MT**3/16d0/Decay_R_SW2&
         /Decay_MW**2*(1d0-rr**2)**2*(1d0+2d0*rr**2)
    IF(lwidth)THEN
       SMtWidth=SMtWidth_cms(0)
    ELSE
       SMtWidth=tWidth_LO
    ENDIF
    IF(finitemass)THEN
       SMtWidth=SMtWidth+SMtWidth_bmass(Finite_MB)
    ENDIF
    IF(qcdord.EQ.0.AND.qedord.EQ.0)RETURN
    IF(qcdord.EQ.1)THEN
       NLOOP=2
       Decay_aS=ALPHAS_MCFM(Decay_MT)
       Decay_gs=DSQRT(4d0*pi*Decay_aS)
       ! the difference between gs is higher order effect
       ! Eq.(11) in arXiv:1106.5483
       CF=4d0/3d0
       SMtWidth=SMtWidth-tWidth_LO*Decay_aS*CF/2d0/pi*&
            (2d0*pi**2/3d0-3d0/2d0-4d0/(3d0*(1d0-rr**2))&
            +1d0/(3d0*(1d0+2d0*rr**2))-2d0*DLOG(rr**2/(1d0-rr**2))&
            +2d0*DLOG(rr**2)*DLOG(1d0-rr**2)+(22d0-34d0*rr**2)/(9d0*(1d0-rr**2)**2)&
            *DLOG(rr**2)+3d0*DLOG(1d0-rr**2)/(1d0+2d0*rr**2)&
            -4d0*DLOG(rr**2)/(9d0*(1d0+2d0*rr**2))+4d0*DBLE(li2R(rr**2)))
    ENDIF
    IF(qedord.EQ.1)THEN
       SMtWidth=SMtWidth+SMtWidth_EW(Decay_Scheme)
    ENDIF
    !IF(lwidth)THEN
    !   ! subtract the double counting terms
    !   SMtWidth=SMtWidth+SMtWidth_cms_dc(qcdord,qedord)
    !ENDIF
    RETURN
  END FUNCTION SMtWidth
  
  FUNCTION SMtWidth_EW(iScheme)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::iScheme ! 1: alpha(MZ) 2: Gmu
    REAL(KIND(1d0))::SMtWidth_EW
    REAL(KIND(1d0)),EXTERNAL::Width_t2Wb_EW
    ! t > W b
    SMtWidth_EW=Width_t2Wb_EW(iScheme)
    RETURN
  END FUNCTION SMtWIDTH_EW

  FUNCTION SMtWidth_bmass(MB)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::MB
    REAL(KIND(1d0))::SMtWidth_bmass
    REAL(KIND(1d0)),EXTERNAL::Width_t2Wb_bmass
    ! include finite b mass effect at LO in alpha and alphas
    SMtWidth_bmass=Width_t2Wb_bmass(MB)
    RETURN
  END FUNCTION SMtWidth_bmass

  FUNCTION SMtWidth_cms(idummy)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::idummy
    REAL(KIND(1d0))::SMtWidth_cms
    REAL(KIND(1d0)),EXTERNAL::Width_t2Wb2udxb_cms
    ! calculate the width in Complex-mass Scheme at LO in alpha and alphas
    ! t > W b > (u d~, c s~, e+ ve, m+ vm, tt+ vt) b
    SMtWidth_cms=Width_t2Wb2udxb_cms(0)*3d0
    RETURN
  END FUNCTION SMtWidth_cms

  FUNCTION SMtWidth_cms_dc(qcdord,qedord)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::qcdord,qedord
    REAL(KIND(1d0))::SMtWidth_cms_dc
    REAL(KIND(1d0)),EXTERNAL::Width_t2Wb2udxb_dc_qcd,Width_t2Wb2udxb_dc_EW
    ! subtract the double-counting ones when there are qcd/qed corrections
    SMtWidth_cms_dc=0d0
    IF(qcdord.EQ.1)THEN
       ! t > W b > (u d~, c s~, e+ ve, m+ vm, tt+ vt) b
       SMtWidth_cms_dc=SMtWidth_cms_dc-Width_t2Wb2udxb_dc_qcd(0)*3d0
    ENDIF
    IF(qedord.EQ.1)THEN
       ! t > W b > (u d~, c s~, e+ ve, m+ vm, tt+ vt) b
       SMtWidth_cms_dc=SMtWidth_cms_dc-Width_t2Wb2udxb_dc_EW(Decay_scheme)*3d0
    ENDIF
    RETURN
  END FUNCTION SMtWidth_cms_dc
END MODULE TDecay

subroutine sudakov_wrapper(p)
implicit none
include 'nexternal.inc'
include "born_nhel.inc"
double precision p(0:3, nexternal-1)

integer i,j
INTEGER IHEL
INTEGER     NCOMB
PARAMETER ( NCOMB=  %(ncomb)d )
INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
double complex ans_summed

include 'orders.inc'
DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD
DOUBLE COMPLEX AMP_SPLIT_EWSUD_LO2(AMP_SPLIT_SIZE)
COMMON /TO_AMP_SPLIT_EWSUD_LO2/ AMP_SPLIT_EWSUD_LO2

DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
DOUBLE COMPLEX AMP_SPLIT_EWSUD_XXC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
common /to_amp_ewsud_PAR/amp_split_ewsud_PAR
DOUBLE COMPLEX AMP_SPLIT_EWSUD_QCD(AMP_SPLIT_SIZE)
common /to_amp_ewsud_QCD/amp_split_ewsud_QCD
DOUBLE COMPLEX AMP_SPLIT_BORN_ONEHEL(AMP_SPLIT_SIZE)
common /to_amp_born_onehel/amp_split_born_onehel
DOUBLE COMPLEX AMP_SPLIT_EWSUD_PARQCD(AMP_SPLIT_SIZE)
COMMON /TO_AMP_EWSUD_PARQCD/AMP_SPLIT_EWSUD_PARQCD

integer iflist(nexternal-1), pdglist(nexternal-1), pdglist_oth(nexternal-1)

double precision invariants(nexternal-1, nexternal-1)

double complex get_lsc_diag, get_lsc_nondiag
external get_lsc_diag, get_lsc_nondiag
double complex get_ssc_n_diag, get_ssc_n_nondiag_1, get_ssc_n_nondiag_2, get_ssc_c
external get_ssc_n_diag, get_ssc_n_nondiag_1, get_ssc_n_nondiag_2, get_ssc_c
double complex get_xxc_diag, get_xxc_nondiag
external get_xxc_diag, get_xxc_nondiag
double complex GET_QCD_LO2
external GET_QCD_LO2

LOGICAL CALCULATEDBORN
COMMON/CCALCULATEDBORN/CALCULATEDBORN

integer ewsud_helselect
common/to_ewsud_helselect/ewsud_helselect

double precision compensate_identical_factor
external compensate_identical_factor
double precision comp_idfac

logical good_hel(ncomb), firsttime_hel(ncomb)
data good_hel  /ncomb*.true./
data firsttime_hel  /ncomb*.true./
integer iamp
logical sud_filter_hel
COMMON /to_filter_hel/ sud_filter_hel
C stuff for the MC over helicities
LOGICAL SUD_MC_HEL
COMMON /TO_MC_HEL/ SUD_MC_HEL
integer igood_hel(ncomb), HEL(0:ncomb)
DATA IGOOD_HEL  /NCOMB * 0/
logical firsttime
data firsttime /.true./
integer hel_min, hel_max, hel_pick
double precision volh

include 'ewsudakov_haslo.inc'


AMP_SPLIT_EWSUD_LSC(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_SSC(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_XXC(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_PAR(:) = dcmplx(0d0,0d0)

AMP_SPLIT_EWSUD_QCD(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_PARQCD(:) = dcmplx(0d0,0d0)


%(iflist)s

call sdk_get_invariants(p, iflist, invariants)

if (firsttime.or..not.sud_mc_hel) then
 hel_min = 1
 hel_max = ncomb
 hel(:)=0
 volh=1d0
 firsttime=.false.
else
 call pickhelicitymc(p,igood_hel,hel,hel_pick,volh)
 hel_min = hel(hel_pick)
 hel_max = hel_min
endif

CALCULATEDBORN=.FALSE.

C Main loop over the helicities
DO IHEL=hel_min, hel_max
C if an helicity is selected, then filter it
if (ewsud_helselect.ne.0.and.ewsud_helselect.ne.ihel) cycle

if (.not.good_hel(ihel).and.sud_filter_hel) cycle

C call the born amplitude and store the result
call sborn_onehel(p,nhel(1,ihel),ihel,ans_summed)
amp_split_born_onehel(:) = amp_split_ewsud(:) 

if (firsttime_hel(ihel)) then
    good_hel(ihel) = .false.
    do iamp = 1, amp_split_size
      good_hel(ihel) = good_hel(ihel).or.amp_split_born_onehel(iamp).ne.0d0.or.amp_split_ewsud_lo2(iamp).ne.0d0
    enddo

    if (good_hel(ihel)) then
      hel(0) = hel(0)+1
      hel(hel(0)) = ihel
      igood_hel(hel(0)) = 1
    endif
    !write(*,*) 'EW SUDAKOV WRAPPER HEL FILTER', ihel, nhel(:,ihel), good_hel(ihel)
    firsttime_hel(ihel) = .false.
endif

C call the amplitudes with the goldstones for the longitudinally-polarised legs and the born for all-transverse
%(calls_to_me)s
enddo

AMP_SPLIT_EWSUD_LSC(:) = AMP_SPLIT_EWSUD_LSC(:)/volh
AMP_SPLIT_EWSUD_SSC(:) = AMP_SPLIT_EWSUD_SSC(:)/volh
AMP_SPLIT_EWSUD_XXC(:) = AMP_SPLIT_EWSUD_XXC(:)/volh
AMP_SPLIT_EWSUD_PAR(:) = AMP_SPLIT_EWSUD_PAR(:)/volh

AMP_SPLIT_EWSUD_QCD(:) = AMP_SPLIT_EWSUD_QCD(:)/VOLH
AMP_SPLIT_EWSUD_PARQCD(:) = AMP_SPLIT_EWSUD_PAR(:)

DO IAMP = 1, AMP_SPLIT_SIZE

  IF (HAS_LO1) THEN
    IF(IAMP.ne.LO1_POS) AMP_SPLIT_EWSUD_PAR(IAMP)=(0d0,0d0)
  ELSE
    AMP_SPLIT_EWSUD_PAR(IAMP)=(0d0,0d0)
  ENDIF

  IF (HAS_LO2) THEN
    IF(IAMP.ne.LO2_POS) AMP_SPLIT_EWSUD_PARQCD(IAMP)=(0d0,0d0)
  ELSE
    AMP_SPLIT_EWSUD_PARQCD(IAMP)=(0d0,0d0)
  ENDIF

ENDDO



CALCULATEDBORN=.FALSE.

return
end





integer function sdk_get_ncomb()
implicit none
INTEGER     NCOMB
PARAMETER ( NCOMB=  %(ncomb)d )

sdk_get_ncomb = ncomb

return
end


subroutine sdk_get_hels(ihel, hels)
implicit none
include 'nexternal.inc'
INTEGER IHEL, i
INTEGER HELS(nexternal-1)
INTEGER     NCOMB
PARAMETER ( NCOMB=  %(ncomb)d )
INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s

hels(:) = NHEL(:, ihel) 

return
end


double precision function compensate_identical_factor(igold_me)
C     provides the multiplicative factor for the result of the
C      sudakov MEs so that they have the same identical-particle
C      factor as the Born in the specific FKS configuration
implicit none
integer igold_me
C 
integer sdk_ident_goldstone(%(ngoldstone_me)d)
data sdk_ident_goldstone / %(sdk_ident_goldstone)s / 
C
integer nfksprocess
common/c_nfksprocess/nfksprocess
%(den_factor_lines)s
integer bornspincol
data bornspincol / %(bornspincol)d /

  compensate_identical_factor = dble(sdk_ident_goldstone(igold_me))/dble(iden_values(nfksprocess)/bornspincol)

return
end


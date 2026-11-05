! Phase-space generation for a free beam particle (momentum p1 along the
! beam axis) on a bound atomic electron p_B = (m_e, vec p_B), in the rest
! frame of the atom (= lab frame), cf. section 11.7.3 of the notes.
!
! Variables: s = (p1+p_B)^2, |p_B|, phi_B with
!   d^3p_B = |p_B|/(2|p1|) ds d|p_B| dphi_B ,
!   |p_B| in [ | |p1|-Q(s) |, |p1|+Q(s) ],   Q(s) = sqrt((E1+m_e)^2 - s),
!   cos(theta_B) = (s0 - s - |p_B|^2)/(2|p1||p_B|),  s0 = m1^2+m_e^2+2 E1 m_e,
! theta_B measured from the beam direction.
!
! Importance sampling (eatom_ps_mode=1): the wavefunction factor of |M|^2
! is ~ n(p)/p^2 with n(p) = sum_shells nume p^2 (gtilde^2+ftilde^2) the radial
! momentum density (Int n dp = Z). In (s,|p_B|) the measure times this factor
! is ~ n(p)/p, flat in s inside each |p_B| window. Hence
!   s     ~ L(s) = G(p+(s)) - G(p-(s)),  G(p) = Int_0^p n(q)/q dq ("electron
!           luminosity"), unless the channel provides a Breit-Wigner for s,
!   |p_B| ~ n(p)/p restricted to the window (inverse of G),
!   phi_B   flat.
! G is tabulated on a logarithmic grid (piecewise-linear interpolant, so the
! sampling density is piecewise constant and the weights are exact); the
! complementary cumulative is kept to stay accurate in the far tails.
! eatom_ps_mode=0: s and |p_B| flat; eatom_ps_mode=2: s flat and the map
! |p_B|/m_e = sqrt(u)/(1-u)**(1/4) proposed in the notes.

MODULE eatom_phase_space
  USE atomic_electron_wavefunctions, ONLY: eatom_radial_density, eatom_table_range
  USE eatom_frame
  IMPLICIT NONE
  PRIVATE
  PUBLIC::eatom_ps_init,eatom_ps_s_from_x,eatom_ps_pb_from_x,eatom_ps_kinematics
  PUBLIC::eatom_ps_boost,eatom_ps_flux,eatom_ps_window,eatom_ps_lumi,eatom_ps_g
  REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338327950288d0
  LOGICAL,SAVE::initialised=.FALSE.
  INTEGER,SAVE::ps_A,ps_Z,ps_iatom,ps_mode
  REAL(KIND(1d0)),SAVE::ps_E1,ps_m1,ps_me,ps_p1,ps_Etot,ps_s0,ps_pmax
  REAL(KIND(1d0)),SAVE::ps_slo,ps_shi
  ! G table: nodes pg(0:NG) (pg(0)=0), bin integrals gb(1:NG) of n(p)/p,
  ! cumulative from below gl(0:NG) and from above gu(0:NG)
  INTEGER,PARAMETER::NG=8000
  REAL(KIND(1d0)),SAVE::pg(0:NG),gb(NG),gl(0:NG),gu(0:NG)
  INTEGER,SAVE::kmed
  ! s table (mode 1): nodes sg(0:ns), bin probabilities sb(1:ns), cumulative cs(0:ns)
  INTEGER,PARAMETER::NSMAX=8004
  INTEGER,SAVE::ns
  REAL(KIND(1d0)),SAVE::sg(0:NSMAX),sb(NSMAX),cs(0:NSMAX)
CONTAINS

  SUBROUTINE eatom_ps_init(A,Z,iatom,E1,m1,me,pmax_in,sthr,mode,stot,slo,shi)
    ! A,Z: atom; iatom: leg of the bound electron (1 or 2); E1,m1: energy and
    ! mass of the free beam; me: electron mass; pmax_in: largest |p_B| to
    ! sample (<=0: kinematic limit); sthr: lowest s allowed by the final
    ! state; mode: see above. Returns stot (largest s, used by madevent to
    ! normalise its invariants) and the sampled s range [slo,shi].
    IMPLICIT NONE
    INTEGER,INTENT(IN)::A,Z,iatom,mode
    REAL(KIND(1d0)),INTENT(IN)::E1,m1,me,pmax_in,sthr
    REAL(KIND(1d0)),INTENT(OUT)::stot,slo,shi
    REAL(KIND(1d0))::ptmin,ptmax,plo,phi,pkin,fa,fm,fb,pa,pb,pm,q,qmax,ds
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::stmp
    INTEGER::k,nside,n,i
    IF(iatom.NE.1.AND.iatom.NE.2)THEN
       WRITE(*,*)"ERROR: eatom_ps_init: iatom must be 1 or 2, got",iatom
       STOP 1
    ENDIF
    ps_A=A
    ps_Z=Z
    ps_iatom=iatom
    ps_mode=mode
    ps_E1=E1
    ps_m1=m1
    ps_me=me
    ps_p1=DSQRT(MAX(E1**2-m1**2,0d0))
    ps_Etot=E1+me
    ps_s0=m1**2+me**2+2d0*E1*me
    IF(ps_p1.LE.0d0)THEN
       WRITE(*,*)"ERROR: eatom_ps_init: the beam must move (E1 > m1)",E1,m1
       STOP 1
    ENDIF
    pkin=ps_p1+DSQRT(MAX(ps_Etot**2-sthr,0d0))
    IF(pmax_in.GT.0d0)THEN
       ps_pmax=MIN(pmax_in,pkin)
    ELSE
       ps_pmax=pkin
    ENDIF
    ! s range: s(p,cos) = s0 - 2 p1 p cos - p^2 with p <= pmax
    ps_slo=MAX(sthr,ps_s0-2d0*ps_p1*ps_pmax-ps_pmax**2)
    IF(ps_pmax.GE.ps_p1)THEN
       ps_shi=ps_Etot**2
    ELSE
       ps_shi=ps_s0+2d0*ps_p1*ps_pmax-ps_pmax**2
    ENDIF
    IF(ps_slo.GE.ps_shi)THEN
       WRITE(*,*)"ERROR: eatom_ps_init: no phase space, s range",ps_slo,ps_shi
       STOP 1
    ENDIF
    stot=ps_shi
    slo=ps_slo
    shi=ps_shi
    ! --- G table on [0,pmax]: log grid from 1e-3 of the first table point
    CALL eatom_table_range(A,Z,ptmin,ptmax)
    plo=MIN(1d-3*ptmin,1d-3*ps_pmax)
    phi=ps_pmax
    pg(0)=0d0
    DO k=1,NG
       pg(k)=plo*(phi/plo)**(DBLE(k-1)/DBLE(NG-1))
    ENDDO
    pg(NG)=phi
    DO k=1,NG
       pa=pg(k-1)
       pb=pg(k)
       pm=0.5d0*(pa+pb)
       fa=0d0
       IF(pa.GT.0d0)fa=eatom_radial_density(A,Z,0,pa)/pa
       fm=eatom_radial_density(A,Z,0,pm)/pm
       fb=eatom_radial_density(A,Z,0,pb)/pb
       gb(k)=(pb-pa)*(fa+4d0*fm+fb)/6d0
    ENDDO
    gl(0)=0d0
    DO k=1,NG
       gl(k)=gl(k-1)+gb(k)
    ENDDO
    gu(NG)=0d0
    DO k=NG-1,0,-1
       gu(k)=gu(k+1)+gb(k+1)
    ENDDO
    kmed=NG
    DO k=0,NG
       IF(gl(k).GE.0.5d0*gl(NG))THEN
          kmed=k
          EXIT
       ENDIF
    ENDDO
    ! --- s table (luminosity), nodes from the |p_B| reached at cos=+-1
    ns=0
    IF(mode.EQ.1)THEN
       nside=(NSMAX-4)/2
       ALLOCATE(stmp(2*nside+4))
       n=0
       n=n+1
       stmp(n)=ps_slo
       n=n+1
       stmp(n)=ps_shi
       IF(ps_s0.GT.ps_slo.AND.ps_s0.LT.ps_shi)THEN
          n=n+1
          stmp(n)=ps_s0
       ENDIF
       ! below s0: s = s0 - 2 p1 q - q^2 (q = p- = Q - p1)
       qmax=ps_pmax
       DO i=1,nside
          q=plo*(qmax/plo)**(DBLE(i-1)/DBLE(nside-1))
          ds=-2d0*ps_p1*q-q**2
          IF(ps_s0+ds.GT.ps_slo.AND.ps_s0+ds.LT.ps_shi)THEN
             n=n+1
             stmp(n)=ps_s0+ds
          ENDIF
       ENDDO
       ! above s0: s = s0 + 2 p1 q - q^2 (q = p- = p1 - Q <= p1)
       qmax=MIN(ps_pmax,ps_p1)
       DO i=1,nside
          q=plo*(qmax/plo)**(DBLE(i-1)/DBLE(nside-1))
          ds=2d0*ps_p1*q-q**2
          IF(ps_s0+ds.GT.ps_slo.AND.ps_s0+ds.LT.ps_shi)THEN
             n=n+1
             stmp(n)=ps_s0+ds
          ENDIF
       ENDDO
       CALL sort_unique(n,stmp)
       ns=n-1
       sg(0:ns)=stmp(1:n)
       DEALLOCATE(stmp)
       cs(0)=0d0
       DO k=1,ns
          sb(k)=0.5d0*(eatom_ps_lumi(sg(k-1))+eatom_ps_lumi(sg(k)))*(sg(k)-sg(k-1))
          cs(k)=cs(k-1)+sb(k)
       ENDDO
       IF(cs(ns).LE.0d0)THEN
          WRITE(*,*)"ERROR: eatom_ps_init: zero electron luminosity in the s range"
          STOP 1
       ENDIF
    ENDIF
    eatom_me_frame=me
    eatom_me_set=.TRUE.
    initialised=.TRUE.
    WRITE(*,'(A,I4,A,I3,A,I2,A,I2)')' eatom phase space: A=',A,' Z=',Z,' iatom=',iatom,' mode=',mode
    WRITE(*,'(A,ES12.5,A,ES12.5,A,ES12.5)')'   |p1|=',ps_p1,' GeV  s0=',ps_s0,' GeV^2  pmax=',ps_pmax
    WRITE(*,'(A,ES14.7,A,ES14.7,A,ES12.5)')'   s range [',ps_slo,',',ps_shi,'] GeV^2   Int n/p dp=',gl(NG)
    RETURN
  END SUBROUTINE eatom_ps_init

  SUBROUTINE sort_unique(n,a)
    ! in-place ascending sort (heapsort) and removal of duplicates
    IMPLICIT NONE
    INTEGER,INTENT(INOUT)::n
    REAL(KIND(1d0)),DIMENSION(n),INTENT(INOUT)::a
    INTEGER::i,j,l,ir,m
    REAL(KIND(1d0))::ra
    IF(n.LT.2)RETURN
    l=n/2+1
    ir=n
    DO
       IF(l.GT.1)THEN
          l=l-1
          ra=a(l)
       ELSE
          ra=a(ir)
          a(ir)=a(1)
          ir=ir-1
          IF(ir.EQ.1)THEN
             a(1)=ra
             EXIT
          ENDIF
       ENDIF
       i=l
       j=l+l
       DO WHILE(j.LE.ir)
          IF(j.LT.ir)THEN
             IF(a(j).LT.a(j+1))j=j+1
          ENDIF
          IF(ra.LT.a(j))THEN
             a(i)=a(j)
             i=j
             j=j+j
          ELSE
             j=ir+1
          ENDIF
       ENDDO
       a(i)=ra
    ENDDO
    m=1
    DO i=2,n
       IF(a(i).GT.a(m))THEN
          m=m+1
          a(m)=a(i)
       ENDIF
    ENDDO
    n=m
    RETURN
  END SUBROUTINE sort_unique

  INTEGER FUNCTION pbin(p)
    ! bin k of the G table with pg(k-1) <= p <= pg(k)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p
    INTEGER::lo,hi,mid
    IF(p.LE.pg(1))THEN
       pbin=1
       RETURN
    ENDIF
    IF(p.GE.pg(NG))THEN
       pbin=NG
       RETURN
    ENDIF
    lo=1
    hi=NG
    DO WHILE(hi-lo.GT.1)
       mid=(lo+hi)/2
       IF(p.LT.pg(mid))THEN
          hi=mid
       ELSE
          lo=mid
       ENDIF
    ENDDO
    pbin=hi
    RETURN
  END FUNCTION pbin

  REAL(KIND(1d0)) FUNCTION glow(p)
    ! G(p) = Int_0^p n(q)/q dq (piecewise-linear interpolant)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p
    INTEGER::k
    IF(p.LE.0d0)THEN
       glow=0d0
       RETURN
    ENDIF
    IF(p.GE.pg(NG))THEN
       glow=gl(NG)
       RETURN
    ENDIF
    k=pbin(p)
    glow=gl(k-1)+gb(k)*(p-pg(k-1))/(pg(k)-pg(k-1))
    RETURN
  END FUNCTION glow

  REAL(KIND(1d0)) FUNCTION gup(p)
    ! Int_p^pmax n(q)/q dq (same interpolant, accurate in the upper tail)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p
    INTEGER::k
    IF(p.GE.pg(NG))THEN
       gup=0d0
       RETURN
    ENDIF
    IF(p.LE.0d0)THEN
       gup=gu(0)
       RETURN
    ENDIF
    k=pbin(p)
    gup=gu(k)+gb(k)*(pg(k)-p)/(pg(k)-pg(k-1))
    RETURN
  END FUNCTION gup

  REAL(KIND(1d0)) FUNCTION eatom_ps_g(pm,pp)
    ! Int_pm^pp n(q)/q dq, split at the median node for accuracy
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::pm,pp
    REAL(KIND(1d0))::pmed
    pmed=pg(kmed)
    eatom_ps_g=0d0
    IF(pp.LE.pm)RETURN
    IF(pm.LT.pmed)eatom_ps_g=eatom_ps_g+glow(MIN(pp,pmed))-glow(pm)
    IF(pp.GT.pmed)eatom_ps_g=eatom_ps_g+gup(MAX(pm,pmed))-gup(pp)
    eatom_ps_g=MAX(eatom_ps_g,0d0)
    RETURN
  END FUNCTION eatom_ps_g

  SUBROUTINE eatom_ps_window(s,pm,pp)
    ! |p_B| window for given s (both 0 if s is not reachable)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::s
    REAL(KIND(1d0)),INTENT(OUT)::pm,pp
    REAL(KIND(1d0))::q2,q
    q2=ps_Etot**2-s
    IF(q2.LT.0d0)THEN
       pm=0d0
       pp=0d0
       RETURN
    ENDIF
    q=DSQRT(q2)
    ! p- = |Q - p1| = |s0 - s|/(Q + p1), without cancellation
    pm=ABS(ps_s0-s)/(q+ps_p1)
    pp=MIN(ps_p1+q,ps_pmax)
    IF(pm.GE.pp)THEN
       pm=0d0
       pp=0d0
    ENDIF
    RETURN
  END SUBROUTINE eatom_ps_window

  REAL(KIND(1d0)) FUNCTION eatom_ps_lumi(s)
    ! electron luminosity L(s) = Int_window n(p)/p dp
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::s
    REAL(KIND(1d0))::pm,pp
    CALL eatom_ps_window(s,pm,pp)
    eatom_ps_lumi=eatom_ps_g(pm,pp)
    RETURN
  END FUNCTION eatom_ps_lumi

  SUBROUTINE eatom_ps_s_from_x(x,s,jac)
    ! s from a uniform x in [0,1] (no Breit-Wigner): luminosity (mode 1)
    ! or flat (modes 0,2) in [slo,shi]; jac = ds/dx
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::x
    REAL(KIND(1d0)),INTENT(OUT)::s,jac
    REAL(KIND(1d0))::target,f
    INTEGER::lo,hi,mid,k
    IF(ps_mode.NE.1)THEN
       s=ps_slo+x*(ps_shi-ps_slo)
       jac=ps_shi-ps_slo
       RETURN
    ENDIF
    target=MIN(MAX(x,0d0),1d0)*cs(ns)
    lo=0
    hi=ns
    DO WHILE(hi-lo.GT.1)
       mid=(lo+hi)/2
       IF(target.LT.cs(mid))THEN
          hi=mid
       ELSE
          lo=mid
       ENDIF
    ENDDO
    k=hi
    ! skip empty bins
    DO WHILE(sb(k).LE.0d0.AND.k.LT.ns)
       k=k+1
    ENDDO
    IF(sb(k).LE.0d0)THEN
       s=ps_shi
       jac=0d0
       RETURN
    ENDIF
    f=MIN(MAX((target-cs(k-1))/sb(k),0d0),1d0)
    s=sg(k-1)+f*(sg(k)-sg(k-1))
    jac=(sg(k)-sg(k-1))*cs(ns)/sb(k)
    RETURN
  END SUBROUTINE eatom_ps_s_from_x

  SUBROUTINE eatom_ps_pb_from_x(s,x,pb,jac)
    ! |p_B| in the window of s from a uniform x in [0,1]; jac = d|p_B|/dx
    ! (jac=0 if the window is empty)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::s,x
    REAL(KIND(1d0)),INTENT(OUT)::pb,jac
    REAL(KIND(1d0))::pm,pp,w,w1,t,um,up,u,xx
    INTEGER::k
    CALL eatom_ps_window(s,pm,pp)
    pb=0d0
    jac=0d0
    IF(pp.LE.pm)RETURN
    xx=MIN(MAX(x,0d0),1d0)
    IF(ps_mode.EQ.0)THEN
       pb=pm+xx*(pp-pm)
       jac=pp-pm
       RETURN
    ELSEIF(ps_mode.EQ.2)THEN
       um=u_of_p(pm)
       up=u_of_p(pp)
       u=um+xx*(up-um)
       IF(u.GE.1d0.OR.u.LE.0d0)RETURN
       pb=ps_me*DSQRT(u)/(1d0-u)**0.25d0
       pb=MIN(MAX(pb,pm),pp)
       jac=(up-um)*ps_me*(2d0-u)/(4d0*DSQRT(u)*(1d0-u)**1.25d0)
       RETURN
    ENDIF
    ! mode 1: inverse of the tabulated G restricted to [pm,pp]
    w=eatom_ps_g(pm,pp)
    IF(w.LE.0d0)THEN
       pb=pm+xx*(pp-pm)
       jac=pp-pm
       RETURN
    ENDIF
    IF(pm.LT.pg(kmed))THEN
       w1=eatom_ps_g(pm,MIN(pp,pg(kmed)))
    ELSE
       w1=0d0
    ENDIF
    t=xx*w
    IF(t.LE.w1.AND.w1.GT.0d0)THEN
       ! lower part: glow(pb) = glow(pm) + t
       CALL invert_low(glow(pm)+t,pb,k)
    ELSE
       ! upper part: gup(pb) = gup(max(pm,pmed)) - (t-w1)
       CALL invert_up(gup(MAX(pm,pg(kmed)))-(t-w1),pb,k)
    ENDIF
    pb=MIN(MAX(pb,pm),pp)
    k=pbin(pb)
    IF(gb(k).LE.0d0)THEN
       jac=0d0
       RETURN
    ENDIF
    jac=w*(pg(k)-pg(k-1))/gb(k)
    RETURN
  END SUBROUTINE eatom_ps_pb_from_x

  SUBROUTINE invert_low(g,p,k)
    ! p with glow(p) = g
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::g
    REAL(KIND(1d0)),INTENT(OUT)::p
    INTEGER,INTENT(OUT)::k
    INTEGER::lo,hi,mid
    lo=0
    hi=NG
    DO WHILE(hi-lo.GT.1)
       mid=(lo+hi)/2
       IF(g.LT.gl(mid))THEN
          hi=mid
       ELSE
          lo=mid
       ENDIF
    ENDDO
    k=hi
    IF(gb(k).GT.0d0)THEN
       p=pg(k-1)+(pg(k)-pg(k-1))*MIN(MAX((g-gl(k-1))/gb(k),0d0),1d0)
    ELSE
       p=pg(k-1)
    ENDIF
    RETURN
  END SUBROUTINE invert_low

  SUBROUTINE invert_up(g,p,k)
    ! p with gup(p) = g (gu decreasing)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::g
    REAL(KIND(1d0)),INTENT(OUT)::p
    INTEGER,INTENT(OUT)::k
    INTEGER::lo,hi,mid
    lo=0
    hi=NG
    DO WHILE(hi-lo.GT.1)
       mid=(lo+hi)/2
       IF(g.GT.gu(mid))THEN
          hi=mid
       ELSE
          lo=mid
       ENDIF
    ENDDO
    k=hi
    IF(gb(k).GT.0d0)THEN
       p=pg(k)-(pg(k)-pg(k-1))*MIN(MAX((g-gu(k))/gb(k),0d0),1d0)
    ELSE
       p=pg(k)
    ENDIF
    RETURN
  END SUBROUTINE invert_up

  REAL(KIND(1d0)) FUNCTION u_of_p(p)
    ! inverse of p/m_e = sqrt(u)/(1-u)^(1/4) (notes, section 11.7.3)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p
    REAL(KIND(1d0))::r2
    r2=(p/ps_me)**2
    ! u = r2/2 (sqrt(4+r2^2) - r2) = 2 r2/(sqrt(4+r2^2)+r2)
    u_of_p=2d0*r2/(DSQRT(4d0+r2**2)+r2)
    RETURN
  END FUNCTION u_of_p

  SUBROUTINE eatom_ps_kinematics(s,pb,phi,plab,pcm,beta,ok)
    ! Atom-rest-frame momenta plab(0:3,1:2) of the two incoming legs (beam
    ! along +z if iatom=2, along -z if iatom=1; p_B=(m_e, vec p_B)), the same
    ! momenta in the rest frame of P=p1+p_B, pcm(0:3,1:2), and the velocity
    ! beta of that frame (pcm -> plab is a boost by +beta).
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::s,pb,phi
    REAL(KIND(1d0)),DIMENSION(0:3,2),INTENT(OUT)::plab,pcm
    REAL(KIND(1d0)),DIMENSION(3),INTENT(OUT)::beta
    LOGICAL,INTENT(OUT)::ok
    REAL(KIND(1d0))::ct,st,dir,ptot(0:3),pB2,rs,e1s,ebs,kst,n(3),nn,tmp(0:3)
    INTEGER::ib,ia
    ok=.FALSE.
    IF(pb.LE.0d0)RETURN
    ct=(ps_s0-s-pb**2)/(2d0*ps_p1*pb)
    IF(ABS(ct).GT.1d0+1d-10)RETURN
    ct=MIN(MAX(ct,-1d0),1d0)
    st=DSQRT(MAX(1d0-ct**2,0d0))
    ia=ps_iatom
    ib=3-ia
    dir=1d0
    IF(ia.EQ.1)dir=-1d0
    plab(0,ib)=ps_E1
    plab(1,ib)=0d0
    plab(2,ib)=0d0
    plab(3,ib)=dir*ps_p1
    plab(0,ia)=ps_me
    plab(1,ia)=pb*st*COS(phi)
    plab(2,ia)=pb*st*SIN(phi)
    plab(3,ia)=dir*pb*ct
    ptot(0:3)=plab(0:3,1)+plab(0:3,2)
    beta(1:3)=ptot(1:3)/ptot(0)
    ! CM momenta with exact energies and momentum, direction from the boost
    pB2=ps_me**2-pb**2
    rs=DSQRT(s)
    e1s=(s+ps_m1**2-pB2)/(2d0*rs)
    ebs=(s+pB2-ps_m1**2)/(2d0*rs)
    kst=DSQRT(MAX(e1s**2-ps_m1**2,0d0))
    CALL eatom_ps_boost(plab(0:3,ib),-beta,tmp)
    nn=DSQRT(tmp(1)**2+tmp(2)**2+tmp(3)**2)
    IF(nn.LE.0d0)RETURN
    n(1:3)=tmp(1:3)/nn
    pcm(0,ib)=e1s
    pcm(1:3,ib)=kst*n(1:3)
    pcm(0,ia)=ebs
    pcm(1:3,ia)=-kst*n(1:3)
    ok=.TRUE.
    RETURN
  END SUBROUTINE eatom_ps_kinematics

  SUBROUTINE eatom_ps_boost(p,beta,pout)
    ! pure Lorentz boost of p by velocity beta (pout = Lambda(beta) p)
    IMPLICIT NONE
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::p
    REAL(KIND(1d0)),DIMENSION(3),INTENT(IN)::beta
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(OUT)::pout
    REAL(KIND(1d0))::b2,g,bp,gm1
    REAL(KIND(1d0)),DIMENSION(0:3)::pin
    pin=p
    b2=beta(1)**2+beta(2)**2+beta(3)**2
    IF(b2.LE.0d0)THEN
       pout=pin
       RETURN
    ENDIF
    g=1d0/DSQRT(1d0-b2)
    gm1=b2*g**2/(g+1d0)
    bp=beta(1)*pin(1)+beta(2)*pin(2)+beta(3)*pin(3)
    pout(0)=g*(pin(0)+bp)
    pout(1:3)=pin(1:3)+(gm1*bp/b2+g*pin(0))*beta(1:3)
    RETURN
  END SUBROUTINE eatom_ps_boost

  REAL(KIND(1d0)) FUNCTION eatom_ps_flux(pb)
    ! 1/(4 m_e |p1|) * 1/(2 pi)^3 * |p_B|/(2|p1|): flux times the Jacobian of
    ! d^3p_B/(2pi)^3 -> ds d|p_B| dphi_B (the phi_B range is in the caller)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::pb
    eatom_ps_flux=1d0/(4d0*ps_me*ps_p1)/(2d0*pi)**3*pb/(2d0*ps_p1)
    RETURN
  END FUNCTION eatom_ps_flux

END MODULE eatom_phase_space

! ------------------------------------------------------------------------
! External wrappers for genps.f (fixed form, no USE)
! ------------------------------------------------------------------------
SUBROUTINE EATOM_PS_SETUP(A,Z,iatom,E1,m1,me,pmax,sthr,mode,stot,slo,shi)
  USE eatom_phase_space
  IMPLICIT NONE
  INTEGER,INTENT(IN)::A,Z,iatom,mode
  REAL(KIND(1d0)),INTENT(IN)::E1,m1,me,pmax,sthr
  REAL(KIND(1d0)),INTENT(OUT)::stot,slo,shi
  CALL eatom_ps_init(A,Z,iatom,E1,m1,me,pmax,sthr,mode,stot,slo,shi)
END SUBROUTINE EATOM_PS_SETUP

SUBROUTINE EATOM_PS_S(x,s,jac)
  USE eatom_phase_space
  IMPLICIT NONE
  REAL(KIND(1d0)),INTENT(IN)::x
  REAL(KIND(1d0)),INTENT(OUT)::s,jac
  CALL eatom_ps_s_from_x(x,s,jac)
END SUBROUTINE EATOM_PS_S

SUBROUTINE EATOM_PS_PB(s,x,pb,jac)
  USE eatom_phase_space
  IMPLICIT NONE
  REAL(KIND(1d0)),INTENT(IN)::s,x
  REAL(KIND(1d0)),INTENT(OUT)::pb,jac
  CALL eatom_ps_pb_from_x(s,x,pb,jac)
END SUBROUTINE EATOM_PS_PB

SUBROUTINE EATOM_PS_KIN(s,pb,phi,plab,pcm,beta,ok)
  USE eatom_phase_space
  IMPLICIT NONE
  REAL(KIND(1d0)),INTENT(IN)::s,pb,phi
  REAL(KIND(1d0)),DIMENSION(0:3,2),INTENT(OUT)::plab,pcm
  REAL(KIND(1d0)),DIMENSION(3),INTENT(OUT)::beta
  LOGICAL,INTENT(OUT)::ok
  CALL eatom_ps_kinematics(s,pb,phi,plab,pcm,beta,ok)
END SUBROUTINE EATOM_PS_KIN

SUBROUTINE EATOM_PS_BOOSTV(p,beta,pout)
  USE eatom_phase_space
  IMPLICIT NONE
  REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::p
  REAL(KIND(1d0)),DIMENSION(3),INTENT(IN)::beta
  REAL(KIND(1d0)),DIMENSION(0:3),INTENT(OUT)::pout
  CALL eatom_ps_boost(p,beta,pout)
END SUBROUTINE EATOM_PS_BOOSTV

REAL(KIND(1d0)) FUNCTION EATOM_PS_FLUXJ(pb)
  USE eatom_phase_space
  IMPLICIT NONE
  REAL(KIND(1d0)),INTENT(IN)::pb
  EATOM_PS_FLUXJ=eatom_ps_flux(pb)
END FUNCTION EATOM_PS_FLUXJ

REAL(KIND(1d0)) FUNCTION EATOM_PS_LUMINOSITY(s)
  USE eatom_phase_space
  IMPLICIT NONE
  REAL(KIND(1d0)),INTENT(IN)::s
  EATOM_PS_LUMINOSITY=eatom_ps_lumi(s)
END FUNCTION EATOM_PS_LUMINOSITY

SUBROUTINE EATOM_PS_MG_SETUP(A,Z,iatom,nexternal,pmass,ebeam,lpp,frame_id,pmax,mode,stot,slo,shi)
  ! madevent set-up: checks the run settings and initialises the sampler.
  ! The free beam is leg 3-iatom with energy ebeam(3-iatom); the atom is at
  ! rest (its ebeam is ignored).
  USE eatom_phase_space
  IMPLICIT NONE
  INTEGER,INTENT(IN)::A,Z,iatom,nexternal,frame_id,mode
  INTEGER,DIMENSION(2),INTENT(IN)::lpp
  REAL(KIND(1d0)),DIMENSION(nexternal),INTENT(IN)::pmass
  REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::ebeam
  REAL(KIND(1d0)),INTENT(IN)::pmax
  REAL(KIND(1d0)),INTENT(OUT)::stot,slo,shi
  INTEGER::ib
  REAL(KIND(1d0))::sthr
  IF(lpp(1).NE.0.OR.lpp(2).NE.0)THEN
     WRITE(*,*)"ERROR: a bound atomic electron requires lpp1 = lpp2 = 0"
     STOP 1
  ENDIF
  IF(frame_id.NE.6)THEN
     WRITE(*,*)"ERROR: the bound-electron matrix element is only defined in the", &
          " rest frame of the atom: me_frame must be [1,2]"
     STOP 1
  ENDIF
  IF(mode.LT.0.OR.mode.GT.2)THEN
     WRITE(*,*)"ERROR: eatom_ps_mode must be 0, 1 or 2, got",mode
     STOP 1
  ENDIF
  ib=3-iatom
  IF(nexternal.EQ.3)THEN
     sthr=pmass(3)**2
  ELSE
     sthr=SUM(pmass(3:nexternal))**2
  ENDIF
  CALL eatom_ps_init(A,Z,iatom,ebeam(ib),pmass(ib),pmass(iatom),pmax,sthr,mode,stot,slo,shi)
END SUBROUTINE EATOM_PS_MG_SETUP

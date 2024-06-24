module kinematics_module
  ! Need to call fill_kinematics_module before anything else !
  use process_module
  implicit none
  integer,public :: ileg,fksfather
  double precision,public :: xm12,xm22,xtk,xuk,xq1q,xq2q,w1,w2,yi,yj,x, &
       xij,betad,betas,kn,knbar,kn0
  double precision,dimension(0:3),private :: xp1,xp2,xk1,xk2,xk3,pp_rec
  double precision,private :: jmass
  double precision,private,parameter :: tiny=1d-5

  public :: get_qMC, fill_kinematics_module,dot,sumdot,pt,deltaR,delta_phi,delta_y
  private

contains
  !TODO: modify qMC to be the shower variable???
  double precision function get_qMC(xi_i_fks,y_ij_fks)
    ! This is the (relative) pT of the splitting. For some showers this is
    ! equal to the shower variable, but not for all. This is what is used for
    ! the damping.
    implicit none
    double precision :: xi_i_fks,y_ij_fks
    if(ileg.eq.1)then
       get_qMC=qMC_ileg1(xi_i_fks,y_ij_fks)
    elseif(ileg.eq.2)then
       get_qMC=qMC_ileg2(xi_i_fks,y_ij_fks)
    elseif(ileg.eq.3)then
       get_qMC=qMC_ileg3(xi_i_fks,y_ij_fks)
    elseif(ileg.eq.4)then
       get_qMC=qMC_ileg4(xi_i_fks,y_ij_fks)
    endif
    if(get_qMC.lt.0d0)then
       write(*,*) 'Error in get_qMC: qMC=',get_qMC
       stop 1
    endif
  end function get_qMC
  
  double precision function qMC_ileg1(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks
    if(shower_mc_mod.eq.'HERWIG6'  .or. &
         shower_mc_mod.eq.'HERWIGPP') qMC_ileg1=xi_i_fks/2d0*sqrt(shat_n1*(1-y_ij_fks**2))
    if(shower_mc_mod.eq.'PYTHIA6Q') qMC_ileg1=sqrt(-xtk)
    if(shower_mc_mod.eq.'PYTHIA6PT'.or. &
         shower_mc_mod.eq.'PYTHIA8') qMC_ileg1=sqrt(-xtk*xi_i_fks)
  end function qMC_ileg1

  double precision function qMC_ileg2(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks
    if(shower_mc_mod.eq.'HERWIG6'  .or. &
         shower_mc_mod.eq.'HERWIGPP') qMC_ileg2=xi_i_fks/2d0*sqrt(shat_n1*(1-y_ij_fks**2))
    if(shower_mc_mod.eq.'PYTHIA6Q') qMC_ileg2=sqrt(-xuk)
    if(shower_mc_mod.eq.'PYTHIA6PT'.or. &
         shower_mc_mod.eq.'PYTHIA8') qMC_ileg2=sqrt(-xuk*xi_i_fks)
  end function qMC_ileg2

  double precision function qMC_ileg3(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks,zeta1,qMCarg,z
    if(shower_mc_mod.eq.'HERWIG6'.or. &
         shower_mc_mod.eq.'HERWIGPP')then
       zeta1=get_zeta(shat_n1,w1,w2,xm12,xm22)
       qMCarg=zeta1*((1-zeta1)*w1-zeta1*xm12)
       if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny) qMCarg=0d0
       if(qMCarg.lt.-tiny) then
          write(*,*) 'Error 1 in qMC_ileg3: negtive sqrt'
          write(*,*) qMCarg
          stop 1
       endif
       qMC_ileg3=sqrt(qMCarg)
    elseif(shower_mc_mod.eq.'PYTHIA6Q')then
       qMC_ileg3=sqrt(w1+xm12)
    elseif(shower_mc_mod.eq.'PYTHIA6PT')then
       write(*,*)'PYTHIA6PT not available for FSR'
       stop
    elseif(shower_mc_mod.eq.'PYTHIA8')then
       z=1d0-shat_n1*xi_i_fks*(xm12+w1)/w1/(shat_n1+w1+xm12-xm22)
       qMC_ileg3=sqrt(z*(1-z)*w1)
    endif
  end function qMC_ileg3

  double precision function qMC_ileg4(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks,zeta2,qMCarg,z
    if(shower_mc_mod.eq.'HERWIG6'.or.shower_mc_mod.eq.'HERWIGPP')then
       zeta2=get_zeta(shat_n1,w2,w1,xm22,xm12)
       qMCarg=zeta2*(1d0-zeta2)*w2
       if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny) qMCarg=0d0
       if(qMCarg.lt.-tiny)then
          write(*,*)'Error 1 in qMC_ileg4: negtive sqrt'
          write(*,*)qMCarg
          stop 1
       endif
       qMC_ileg4=sqrt(qMCarg)
    elseif(shower_mc_mod.eq.'PYTHIA6Q')then
       qMC_ileg4=sqrt(w2)
    elseif(shower_mc_mod.eq.'PYTHIA6PT')then
       write(*,*)'PYTHIA6PT not available for FSR'
       stop
    elseif(shower_mc_mod.eq.'PYTHIA8')then
       z=1d0-shat_n1*xi_i_fks/(shat_n1+w2-xm12)
       qMC_ileg4=sqrt(z*(1-z)*w2)
    endif
  end function qMC_ileg4

  subroutine fill_kinematics_module(pp,i_fks,j_fks,xi_i_fks,y_ij_fks,mass)
    ! takes an n+1-body phase-space point, and fills invariants relevant for
    ! computation of shower subtraction terms
    implicit none
    double precision,dimension(0:3,next_n1) :: pp
    double precision :: xi_i_fks,y_ij_fks,mass
    integer :: i_fks,j_fks
    double precision :: veckn_ev,veckbarn_ev,xp0jfks
    common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
    
    fksfather=min(i_fks,j_fks)
    
    jmass=mass ! this is the mass of j_fks

    xm12=0d0
    xm22=0d0
    xq1q=0d0
    xq2q=0d0
    kn=veckn_ev
    knbar=veckbarn_ev
    kn0=xp0jfks

    ! Determine ileg
    call fill_ileg()

    ! fill the momenta for the recoilers and emitters and emitted.
    call get_momenta_emitter_recoiler(pp,i_fks,j_fks)

    ! Determine the Mandelstam invariants needed in the MC functions in terms
    ! of FKS variables: the argument of MC functions are (p+k)^2, NOT 2 p.k
    !
    ! Definitions of invariants in terms of momenta
    !
    ! xm12 =     xk1 . xk1
    ! xm22 =     xk2 . xk2
    ! xtk  = - 2 xp1 . xk3
    ! xuk  = - 2 xp2 . xk3
    ! xq1q = - 2 xp1 . xk1 + xm12
    ! xq2q = - 2 xp2 . xk2 + xm22
    ! w1   = + 2 xk1 . xk3        = - xq1q + xq2q - xtk
    ! w2   = + 2 xk2 . xk3        = - xq2q + xq1q - xuk
    ! xq1c = - 2 xp1 . xk2        = - s - xtk - xq1q + xm12
    ! xq2c = - 2 xp2 . xk1        = - s - xuk - xq2q + xm22
    !
    ! Parametrisation of invariants in terms of FKS variables
    !
    ! ileg = 1
    ! xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
    ! xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
    ! xk3  =  B * ( 1 , 0 , sqrt(1-yi**2) , yi )
    ! xk1  =  irrelevant
    ! xk2  =  irrelevant
    ! yi = y_ij_fks
    ! x = 1 - xi_i_fks
    ! B = sqrt(s)/2*(1-x)
    !
    ! ileg = 2
    ! xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
    ! xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
    ! xk3  =  B * ( 1 , 0 , sqrt(1-yi**2) , -yi )
    ! xk1  =  irrelevant
    ! xk2  =  irrelevant
    ! yi = y_ij_fks
    ! x = 1 - xi_i_fks
    ! B = sqrt(s)/2*(1-x)
    !
    ! ileg = 3
    ! xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
    ! xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
    ! xk1  =  ( sqrt(veckn_ev**2+xm12) , 0 , 0 , veckn_ev )
    ! xk2  =  xp1 + xp2 - xk1 - xk3
    ! xk3  =  B * ( 1 , 0 , sqrt(1-yj**2) , yj )
    ! yj = y_ij_fks
    ! yi = irrelevant
    ! x = 1 - xi_i_fks
    ! veckn_ev is such that xk2**2 = xm22
    ! B = sqrt(s)/2*(1-x)
    ! azimuth = irrelevant (hence set = 0)
    !
    ! ileg = 4
    ! xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
    ! xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
    ! xk1  =  xp1 + xp2 - xk2 - xk3
    ! xk2  =  A * ( 1 , 0 , 0 , 1 )
    ! xk3  =  B * ( 1 , 0 , sqrt(1-yj**2) , yj )
    ! yj = y_ij_fks
    ! yi = irrelevant
    ! x = 1 - xi_i_fks
    ! A = (s*x-xm12)/(sqrt(s)*(2-(1-x)*(1-yj)))
    ! B = sqrt(s)/2*(1-x)
    ! azimuth = irrelevant (hence set = 0)

    if(ileg.eq.1)then
       call fill_invariants_ileg1(xi_i_fks,y_ij_fks)
       call check_invariants_ileg12
    elseif(ileg.eq.2)then
       call fill_invariants_ileg2(xi_i_fks,y_ij_fks)
       call check_invariants_ileg12
    elseif(ileg.eq.3)then
       call fill_invariants_ileg3(xi_i_fks,y_ij_fks)
       call check_invariants_ileg3
    elseif(ileg.eq.4)then
       call fill_invariants_ileg4(xi_i_fks,y_ij_fks)
       call check_invariants_ileg4
    else
       write(*,*)'Error 4 in fill_kinematics_module: assigned wrong ileg'
       stop
    endif
    x=1d0-xi_i_fks
    xij=2d0*(1d0-xm12/shat_n1-(1d0-x))/(2d0-(1d0-x)*(1d0-yj))
    betad=sqrt((1d0-(xm12-xm22)/shat_n1)**2-(4d0*xm22/shat_n1))
    betas=1d0+(xm12-xm22)/shat_n1
  end subroutine fill_kinematics_module
  
  double precision function deltaR(p1,p2)
    implicit none
    double precision,dimension(0:3) :: p1,p2
    double precision :: eta,delta_phi
    deltaR = sqrt((delta_phi(p1,p2))**2+(delta_y(p1,p2))**2)
  end function deltaR

  double precision function delta_phi(p1, p2)
    implicit none
    double precision,dimension(0:3) :: p1,p2
    double precision :: denom, temp
    double precision,parameter :: tiny=1d-8
    denom = sqrt(p1(1)**2 + p1(2)**2) * sqrt(p2(1)**2 + p2(2)**2)
    temp = max(-(1d0-tiny), (p1(1)*p2(1) + p1(2)*p2(2)) / denom)
    temp = min( (1d0-tiny), temp)
    delta_phi = acos(temp)
  end function delta_phi

  double precision  function delta_y(p1,p2)
    implicit none
    double precision,dimension(0:3) :: p1,p2
    delta_y =.5d0*dlog((p1(0)+p1(3))/(p1(0)-p1(3)))- &
             .5d0*dlog((p2(0)+p2(3))/(p2(0)-p2(3)))
  end function delta_y
      
  double precision function pt(p)
    implicit none
    double precision,dimension(0:3) :: p
    pt = dsqrt(p(1)**2+p(2)**2)
  end function pt

  double precision function sumdot(p1,p2,sign)
    implicit  none
    double precision,dimension(0:3) :: p1,p2
    double precision :: sign
    sumdot=dot(p1+sign*p2,p1+sign*p2)
  end function sumdot
  
  double precision function dot(p1,p2)
    implicit none
    double precision,dimension(0:3) :: p1,p2
    dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
  end function dot

  subroutine fill_invariants_ileg1(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks
    xtk=-shat_n1*xi_i_fks*(1-y_ij_fks)/2d0
    xuk=-shat_n1*xi_i_fks*(1+y_ij_fks)/2d0
    yj=0d0
    yi=y_ij_fks
  end subroutine fill_invariants_ileg1

  subroutine fill_invariants_ileg2(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks
    xtk=-shat_n1*xi_i_fks*(1+y_ij_fks)/2d0
    xuk=-shat_n1*xi_i_fks*(1-y_ij_fks)/2d0
    yj=0d0
    yi=y_ij_fks
  end subroutine fill_invariants_ileg2

  subroutine fill_invariants_ileg3(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks
    xm12=jmass**2
    xm22=dot(pp_rec,pp_rec)
    xtk=-2d0*dot(xp1,xk3)
    xuk=-2d0*dot(xp2,xk3)
    xq1q=-2d0*dot(xp1,xk1)+xm12
    xq2q=-2d0*dot(xp2,xk2)+xm22
    w1=-xq1q+xq2q-xtk
    w2=-xq2q+xq1q-xuk
    yj=y_ij_fks
    yi=0d0
  end subroutine fill_invariants_ileg3

  subroutine fill_invariants_ileg4(xi_i_fks,y_ij_fks)
    implicit none
    double precision :: xi_i_fks,y_ij_fks,xij
    xm12=dot(pp_rec,pp_rec)
    xm22=0d0
    xtk=-2d0*dot(xp1,xk3)
    xuk=-2d0*dot(xp2,xk3)
    xij=2d0*(1d0-xm12/shat_n1-xi_i_fks)/(2d0-xi_i_fks*(1d0-y_ij_fks))
    w2=shat_n1*xi_i_fks*xij*(1d0-y_ij_fks)/2d0
    xq2q=-shat_n1*xij*(2d0-dot(xp1,xk2)*4d0/(shat_n1*xij))/2d0
    xq1q=xuk+xq2q+w2
    w1=-xq1q+xq2q-xtk
    yj=y_ij_fks
    yi=0d0
  end subroutine fill_invariants_ileg4


  double precision function get_zeta(xs,xw1,xw2,xxm12,xxm22)
    implicit none
    double precision :: xs,xw1,xw2,xxm12,xxm22
    double precision :: eps,beta
    eps=1-(xxm12-xxm22)/(xs-xw1)
    beta=sqrt(eps**2-4*xs*xxm22/(xs-xw1)**2)
    get_zeta=( (2*xs-(xs-xw1)*eps)*xw2+(xs-xw1)*((xw1+xw2)*beta-eps*xw1) )/ &
         ( (xs-xw1)*beta*(2*xs-(xs-xw1)*eps+(xs-xw1)*beta) )
  end function get_zeta

  subroutine fill_ileg()
    implicit none
    ! ileg = 1 ==> emission from left     incoming parton
    ! ileg = 2 ==> emission from right    incoming parton
    ! ileg = 3 ==> emission from massive  outgoing parton
    ! ileg = 4 ==> emission from massless outgoing parton
    ! Instead of jmass, one should use pmass(fksfather), but the
    ! kernels where pmass(fksfather) != jmass are non-singular
    if(fksfather.le.2 .and. fksfather.gt.0)then
       ileg=fksfather
    elseif(jmass.ne.0d0)then
       ileg=3
    elseif(jmass.eq.0d0)then
       ileg=4
    else
       write(*,*)'Error 1 in get_ileg: unknown ileg'
       write(*,*)ileg,fksfather,jmass
       stop
    endif
    if(ileg.gt.2 .and. shower_mc_mod.eq.'PYTHIA6PT')then
       write (*,*) 'FSR not allowed when matching PY6PT'
       stop 1
    endif
  end subroutine fill_ileg
  
  
  subroutine get_momenta_emitter_recoiler(pp,i_fks,j_fks)
    implicit none
    double precision,dimension(0:3,next_n1) :: pp
    integer :: i_fks,j_fks
    ! Determine and assign momenta:
    ! xp1 = incoming left parton  (emitter (recoiler) if ileg = 1 (2))
    ! xp2 = incoming right parton (emitter (recoiler) if ileg = 2 (1))
    ! xk1 = outgoing parton       (emitter (recoiler) if ileg = 3 (4))
    ! xk2 = outgoing parton       (emitter (recoiler) if ileg = 4 (3))
    ! xk3 = extra parton          (FKS parton)
    ! (xk1 and xk2 are never used for ISR)
    xp1(0:3)=pp(0:3,1)
    xp2(0:3)=pp(0:3,2)
    xk3(0:3)=pp(0:3,i_fks)
    if(ileg.gt.2)pp_rec(0:3)=pp(0:3,1)+pp(0:3,2)-pp(0:3,i_fks)-pp(0:3,j_fks)
    if(ileg.eq.3)then
       xk1(0:3)=pp(0:3,j_fks)
       xk2(0:3)=pp_rec(0:3)
    elseif(ileg.eq.4)then
       xk1(0:3)=pp_rec(0:3)
       xk2(0:3)=pp(0:3,j_fks)
    endif
  end subroutine get_momenta_emitter_recoiler

  
  subroutine check_invariants_ileg12
    implicit none
    integer,parameter :: max_imprecision=10
    integer,save,dimension(7) :: imprecision=0
    if((abs(xtk+2*dot(xp1,xk3))/shat_n1.ge.tiny).or. &
         (abs(xuk+2*dot(xp2,xk3))/shat_n1.ge.tiny))then
       write(*,*)'Warning: imprecision 1 in check_invariants_ileg12'
       write(*,*)abs(xtk+2*dot(xp1,xk3))/shat_n1, &
            abs(xuk+2*dot(xp2,xk3))/shat_n1
       imprecision(1)=imprecision(1)+1
       if (imprecision(1).ge.max_imprecision) then
          write (*,*) 'Error: ',max_imprecision &
               ,' imprecisions. Stopping...'
          stop
       endif
    endif
  end subroutine check_invariants_ileg12

  subroutine check_invariants_ileg3
    implicit none
    integer,parameter :: max_imprecision=10
    integer,save,dimension(7) :: imprecision=0
    if(sqrt(w1+xm12).ge.sqrt(shat_n1)-sqrt(xm22))then
       write(*,*)'Warning: imprecision 2 in check_invariants_ileg3'
       write(*,*)sqrt(w1),sqrt(shat_n1),xm22
       imprecision(2)=imprecision(2)+1
       if (imprecision(2).ge.max_imprecision) then
          write (*,*) 'Error: ',max_imprecision &
               ,' imprecisions. Stopping...'
          stop
       endif
    endif
    if(((abs(w1-2*dot(xk1,xk3))/shat_n1.ge.tiny)).or. &
         ((abs(w2-2*dot(xk2,xk3))/shat_n1.ge.tiny)))then
       write(*,*)'Warning: imprecision 3 in check_invariants_ileg3'
       write(*,*)abs(w1-2*dot(xk1,xk3))/shat_n1, &
            abs(w2-2*dot(xk2,xk3))/shat_n1
       imprecision(3)=imprecision(3)+1
       if (imprecision(3).ge.max_imprecision) then
          write (*,*) 'Error: ',max_imprecision &
               ,' imprecisions. Stopping...'
          stop
       endif
    endif
    if(xm12.eq.0d0)then
       write(*,*)'Warning 4 in check_invariants_ileg3'
       imprecision(4)=imprecision(4)+1
       if (imprecision(4).ge.max_imprecision) then
          write (*,*) 'Error: ',max_imprecision &
               ,' warnings. Stopping...'
          stop
       endif
    endif
  end subroutine check_invariants_ileg3

  subroutine check_invariants_ileg4
    implicit none
    integer,parameter :: max_imprecision=10
    integer,save,dimension(7) :: imprecision=0
    if(sqrt(w2).ge.sqrt(shat_n1)-sqrt(xm12))then
       write(*,*)'Warning: imprecision 5 in check_invariants_ileg4'
       write(*,*)sqrt(w2),sqrt(shat_n1),xm12
       imprecision(5)=imprecision(5)+1
       if (imprecision(5).ge.max_imprecision) then
          write (*,*) 'Error: ',max_imprecision &
               ,' imprecisions. Stopping...'
          stop
       endif
    endif
    if(((abs(w2-2*dot(xk2,xk3))/shat_n1.ge.tiny)).or. &
         ((abs(xq2q+2*dot(xp2,xk2))/shat_n1.ge.tiny)).or. &
         ((abs(xq1q+2*dot(xp1,xk1)-xm12)/shat_n1.ge.tiny)))then
       write(*,*)'Warning: imprecision 6 in check_invariants_ileg4'
       write(*,*)abs(w2-2*dot(xk2,xk3))/shat_n1, &
            abs(xq2q+2*dot(xp2,xk2))/shat_n1, &
            abs(xq1q+2*dot(xp1,xk1)-xm12)/shat_n1
       imprecision(6)=imprecision(6)+1
       if (imprecision(6).ge.max_imprecision) then
          write (*,*) 'Error: ',max_imprecision &
               ,' imprecisions. Stopping...'
          stop
       endif
    endif
    if(xm22.ne.0d0)then
       write(*,*)'Warning 7 in check_invariants_ileg4'
       imprecision(7)=imprecision(7)+1
       if (imprecision(7).ge.max_imprecision) then
          write (*,*) 'Error: ',max_imprecision &
               ,' warnings. Stopping...'
          stop
       endif
    endif
  end subroutine check_invariants_ileg4


end module kinematics_module

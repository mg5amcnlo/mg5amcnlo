module kinematics_module
  implicit none
  integer,public :: ileg
  double precision,public :: xm12,xm22,xtk,xuk,xq1q,xq2q,qMC,w1,w2
  double precision,dimension(0:3),private :: xp1,xp2,xk1,xk2,xk3,pp_rec
  double precision,private :: sh,jmass
  double precision,private,parameter :: tiny=1d-5
contains
  subroutine fill_kinematics_module(pp,i_fks,j_fks,xi_i_fks,y_ij_fks,mass)
    ! takes an n+1-body phase-space point, and fills invariants relevant for
    ! computation of shower subtraction terms
    implicit none
    include "nexternal.inc"
    include "run.inc"
    double precision,dimension(0:3,nexternal) :: pp
    double precision :: xi_i_fks,y_ij_fks
    integer :: i_fks,j_fks,fks_father

    jmass=mass ! this is the mass of j_fks
    sh=2d0*dot(p(0:3,1),p(0:3,2)) ! s-hat

    xm12=0d0
    xm22=0d0
    xq1q=0d0
    xq2q=0d0
    qMC=-1d0

    ! Determine ileg
    call fill_ileg(i_fks,j_fks)

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
    elseif(ileg.eq.2)then
       call fill_invariants_ileg2(xi_i_fks,y_ij_fks)
    elseif(ileg.eq.3)then
       call fill_invariants_ileg3(xi_i_fks,y_ij_fks)
    elseif(ileg.eq.4)then
       call fill_invariants_ileg4(xi_i_fks,y_ij_fks)
    else
       write(*,*)'Error 4 in fill_kinematics_module: assigned wrong ileg'
       stop
    endif
  end subroutine fill_kinematics_module

  double precision function dot(p1,p2)
    implicit none
    double precision,dimension(0:3) :: p1,p2
    dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
  end function dot

  subroutine fill_invariants_ileg1(xi_i_fks,y_ij_fks)
    implicit none
    include "run.inc"
    double precision :: xi_i_fks,y_ij_fks
    xtk=-sh*xi_i_fks*(1-y_ij_fks)/2d0
    xuk=-sh*xi_i_fks*(1+y_ij_fks)/2d0
    if(shower_mc.eq.'HERWIG6'  .or. &
         shower_mc.eq.'HERWIGPP') qMC=xi_i_fks/2d0*sqrt(sh*(1-y_ij_fks**2))
    if(shower_mc.eq.'PYTHIA6Q')qMC=sqrt(-xtk)
    if(shower_mc.eq.'PYTHIA6PT'.or. &
         shower_mc.eq.'PYTHIA8')qMC=sqrt(-xtk*xi_i_fks)
  end subroutine fill_invariants_ileg1

  subroutine fill_invariants_ileg2(xi_i_fks,y_ij_fks)
    implicit none
    include "run.inc"
    double precision :: xi_i_fks,y_ij_fks
    xtk=-sh*xi_i_fks*(1+y_ij_fks)/2d0
    xuk=-sh*xi_i_fks*(1-y_ij_fks)/2d0
    if(shower_mc.eq.'HERWIG6'  .or. &
         shower_mc.eq.'HERWIGPP') qMC=xi_i_fks/2d0*sqrt(sh*(1-y_ij_fks**2))
    if(shower_mc.eq.'PYTHIA6Q') qMC=sqrt(-xuk)
    if(shower_mc.eq.'PYTHIA6PT'.or. &
         shower_mc.eq.'PYTHIA8') qMC=sqrt(-xuk*xi_i_fks)
  end subroutine fill_invariants_ileg2

  subroutine fill_invariants_ileg3(xi_i_fks,y_ij_fks)
    implicit none
    include "run.inc"
    double precision :: xi_i_fks,y_ij_fks,zeta1,qMCarg,z
    xm12=jmass**2
    xm22=dot(pp_rec,pp_rec)
    xtk=-2d0*dot(xp1,xk3)
    xuk=-2d0*dot(xp2,xk3)
    xq1q=-2d0*dot(xp1,xk1)+xm12
    xq2q=-2d0*dot(xp2,xk2)+xm22
    w1=-xq1q+xq2q-xtk
    w2=-xq2q+xq1q-xuk
    if(shower_mc.eq.'HERWIG6'.or. &
         shower_mc.eq.'HERWIGPP')then
       zeta1=get_zeta(sh,w1,w2,xm12,xm22)
       qMCarg=zeta1*((1-zeta1)*w1-zeta1*xm12)
       if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny) qMCarg=0d0
       if(qMCarg.lt.-tiny) then
          write(*,*) 'Error 1 in fill_invariants_ileg3: negtive sqrt'
          write(*,*) qMCarg
          stop 1
       endif
       qMC=sqrt(qMCarg)
    elseif(shower_mc.eq.'PYTHIA6Q')then
       qMC=sqrt(w1+xm12)
    elseif(shower_mc.eq.'PYTHIA6PT')then
       write(*,*)'PYTHIA6PT not available for FSR'
       stop
    elseif(shower_mc.eq.'PYTHIA8')then
       z=1d0-sh*xi_i_fks*(xm12+w1)/w1/(sh+w1+xm12-xm22)
       qMC=sqrt(z*(1-z)*w1)
    endif
  end subroutine fill_invariants_ileg3

  subroutine fill_invariants_ileg4(xi_i_fks,y_ij_fks)
    implicit none
    include "run.inc"
    double precision :: xi_i_fks,y_ij_fks,zeta2,qMCarg,z
    xm12=dot(pp_rec,pp_rec)
    xm22=0d0
    xtk=-2d0*dot(xp1,xk3)
    xuk=-2d0*dot(xp2,xk3)
    xij=2d0*(1d0-xm12/sh-xi_i_fks)/(2d0-xi_i_fks*(1d0-y_ij_fks))
    w2=sh*xi_i_fks*xij*(1d0-y_ij_fks)/2d0
    xq2q=-sh*xij*(2d0-dot(xp1,xk2)*4d0/(sh*xij))/2d0
    xq1q=xuk+xq2q+w2
    w1=-xq1q+xq2q-xtk
    if(shower_mc.eq.'HERWIG6'.or.shower_mc.eq.'HERWIGPP')then
       zeta2=get_zeta(sh,w2,w1,xm22,xm12)
       qMCarg=zeta2*(1d0-zeta2)*w2
       if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny) qMCarg=0d0
       if(qMCarg.lt.-tiny)then
          write(*,*)'Error 1 in fill_invariants_ileg4: negtive sqrt'
          write(*,*)qMCarg
          stop 1
       endif
       qMC=sqrt(qMCarg)
    elseif(shower_mc.eq.'PYTHIA6Q')then
       qMC=sqrt(w2)
    elseif(shower_mc.eq.'PYTHIA6PT')then
       write(*,*)'PYTHIA6PT not available for FSR'
       stop
    elseif(shower_mc.eq.'PYTHIA8')then
       z=1d0-sh*xi_i_fks/(sh+w2-xm12)
       qMC=sqrt(z*(1-z)*w2)
    endif
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

  subroutine fill_ileg(i_fks,j_fks)
    implicit none
    integer :: i_fks,j_fks,fksfather
    ! ileg = 1 ==> emission from left     incoming parton
    ! ileg = 2 ==> emission from right    incoming parton
    ! ileg = 3 ==> emission from massive  outgoing parton
    ! ileg = 4 ==> emission from massless outgoing parton
    ! Instead of jmass, one should use pmass(fksfather), but the
    ! kernels where pmass(fksfather) != jmass are non-singular
    fksfather=min(i_fks,j_fks)
    if(fksfather.le.2)then
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
  end subroutine fill_ileg
  
  
  subroutine get_momenta_emitter_recoiler(pp,i_fks,j_fks)
    implicit none
    include "nexternal.inc"
    double precision,dimension(0:3,nexternal) :: pp
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

  
  subroutine check_invariants
    implicit none
    integer,parameter :: max_imprecision=10
    integer,save,dimension(7) :: imprecision=0
    if(ileg.eq.1 .or. ileg.eq.2)then
       if((abs(xtk+2*dot(xp1,xk3))/sh.ge.tiny).or. &
            (abs(xuk+2*dot(xp2,xk3))/sh.ge.tiny))then
          write(*,*)'Warning: imprecision 1 in check_invariants'
          write(*,*)abs(xtk+2*dot(xp1,xk3))/sh, &
               abs(xuk+2*dot(xp2,xk3))/sh
          imprecision(1)=imprecision(1)+1
          if (imprecision(1).ge.max_imprecision) then
             write (*,*) 'Error: ',max_imprecision &
                  ,' imprecisions. Stopping...'
             stop
          endif
       endif
    elseif(ileg.eq.3)then
       if(sqrt(w1+xm12).ge.sqrt(sh)-sqrt(xm22))then
          write(*,*)'Warning: imprecision 2 in check_invariants'
          write(*,*)sqrt(w1),sqrt(sh),xm22
          imprecision(2)=imprecision(2)+1
          if (imprecision(2).ge.max_imprecision) then
             write (*,*) 'Error: ',max_imprecision &
                  ,' imprecisions. Stopping...'
             stop
          endif
       endif
       if(((abs(w1-2*dot(xk1,xk3))/sh.ge.tiny)).or. &
            ((abs(w2-2*dot(xk2,xk3))/sh.ge.tiny)))then
          write(*,*)'Warning: imprecision 3 in check_invariants'
          write(*,*)abs(w1-2*dot(xk1,xk3))/sh, &
               abs(w2-2*dot(xk2,xk3))/sh
          imprecision(3)=imprecision(3)+1
          if (imprecision(3).ge.max_imprecision) then
             write (*,*) 'Error: ',max_imprecision &
                  ,' imprecisions. Stopping...'
             stop
          endif
       endif
       if(xm12.eq.0d0)then
          write(*,*)'Warning 4 in check_invariants'
          imprecision(4)=imprecision(4)+1
          if (imprecision(4).ge.max_imprecision) then
             write (*,*) 'Error: ',max_imprecision &
                  ,' warnings. Stopping...'
             stop
          endif
       endif
    elseif(ileg.eq.4)then
       if(sqrt(w2).ge.sqrt(sh)-sqrt(xm12))then
          write(*,*)'Warning: imprecision 5 in check_invariants'
          write(*,*)sqrt(w2),sqrt(sh),xm12
          imprecision(5)=imprecision(5)+1
          if (imprecision(5).ge.max_imprecision) then
             write (*,*) 'Error: ',max_imprecision &
                  ,' imprecisions. Stopping...'
             stop
          endif
       endif
       if(((abs(w2-2*dot(xk2,xk3))/sh.ge.tiny)).or. &
            ((abs(xq2q+2*dot(xp2,xk2))/sh.ge.tiny)).or. &
            ((abs(xq1q+2*dot(xp1,xk1)-xm12)/sh.ge.tiny)))then
          write(*,*)'Warning: imprecision 6 in check_invariants'
          write(*,*)abs(w2-2*dot(xk2,xk3))/sh, &
               abs(xq2q+2*dot(xp2,xk2))/sh, &
               abs(xq1q+2*dot(xp1,xk1)-xm12)/sh
          imprecision(6)=imprecision(6)+1
          if (imprecision(6).ge.max_imprecision) then
             write (*,*) 'Error: ',max_imprecision &
                  ,' imprecisions. Stopping...'
             stop
          endif
       endif
       if(xm22.ne.0d0)then
          write(*,*)'Warning 7 in check_invariants'
          imprecision(7)=imprecision(7)+1
          if (imprecision(7).ge.max_imprecision) then
             write (*,*) 'Error: ',max_imprecision &
                  ,' warnings. Stopping...'
             stop
          endif
       endif
    endif
  end subroutine check_invariants

  
end module kinematics_module
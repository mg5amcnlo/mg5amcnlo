c The function fks_Sij returns the FKS function S_{i,j} with
c   i == ii_fks   and   j=jj_fks
c Note that ii_fks and jj_fks need not coincide with the indices of the
c FKS parton (i_fks) and of its sister (j_fks), which are passed to
c this function with a common block. Note also that the argument
c   xi_i_fks
c is the rescaled energy of the FKS parton, and thus does not coincide,
c in general, with the rescaled energy of parton ii_fks. Analogously,
c   y_ij_fks
c is the cosine of the angle between the FKS parton and its sister,
c and thus does not coincide, in general, with the cosine of the angle
c between partons ii_fks and jj_fks.
c
c The function fks_Sij has smooth numerical limits for 
c   E(i_fks)-->0   and  theta(i_fks,j_fks)-->0
c if and only if ii_fks=i_fks and if (ii_fks,jj_fks)=(i_fks,j_fks)
c respectively. In all other cases, the program is protected against
c a division by zero by the parameter vtiny in get_dkl_Sij, but this implies
c discontinuities as the limits are approached. A side effect is that,
c close to these limits, the sum of all S functions may not be equal to
c one. This must be irrelevant in practical applications, since within
c one given directory this function should only be called with
c arguments (ii_fks,jj_fks)=(i_fks,j_fks). An exception is the directory
c where the finite contribution is computed, where potential problems
c can be avoided with a suitable choice of f()
c
c 
      Double precision function fks_Sij(p,ii_fks,jj_fks,
     #                                  xi_i_fks,y_ij_fks)
      implicit none

      include "nexternal.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include "fks_powers.inc"
      include "coupl.inc"

      real*8 p(0:3,nexternal)
      integer ii_fks,jj_fks
      real*8 xi_i_fks,y_ij_fks

      double precision shattmp,dot

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      logical firsttime
      real*8 hfact,h_damp,z
      real*8 phat_ii(0:3),E_ii_resc,xnum,xden,inverseSij
      integer i,j,k,l,kk,ll,ihdamp,isorsc,ijskip(nexternal,nexternal)
      integer ione,itwo
      parameter (ione=1)
      parameter (itwo=2)
      save ijskip

      logical setsijzero

      double precision tiny,zero,one
      parameter (tiny=1.d-6)
      parameter (zero=0d0)
      parameter (one=1d0)

      double precision pmass(nexternal)
      include "pmass.inc"

      data firsttime/.true./

      if(p(0,1).le.0.d0)then
c Unphysical kinematics: set S function equal to zero
        fks_Sij=0.d0
        return
      endif

      if(particle_type(jj_fks).eq.8.and.particle_type(ii_fks).ne.8.and.
     &     jj_fks.gt.nincoming)then
        write(*,*)'Error #0 in fks_Sij',ii_fks,jj_fks,
     #    particle_type(ii_fks),particle_type(jj_fks)
        stop
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(p(0,1),p(0,2))
      else
         shattmp=p(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in fks_Sij: inconsistent shat #1'
        write(*,*)shattmp,shat
        stop
      endif

      setsijzero=.false.
      E_ii_resc=-one

      if (xi_i_fks.gt.tiny.or.ii_fks.ne.i_fks)then
         do i=0,3
            phat_ii(i)=p(i,ii_fks)
         enddo
         E_ii_resc=one
      elseif (xi_i_fks.le.tiny.and.ii_fks.eq.i_fks)then
         isorsc=0
         if( 1d0-y_ij_fks.lt.tiny.and.jj_fks.eq.j_fks.and.
     #       pmass(j_fks).eq.0.d0 )isorsc=2
         if(p_i_fks_cnt(0,isorsc).lt.0.d0)then
           if(xi_i_fks.eq.0.d0)then
             write(*,*)'Error #7 in fks_Sij',isorsc,xi_i_fks,y_ij_fks
             stop
           endif
           if(p(0,ii_fks).ne.0.d0)then
             write(*,*)'WARNING in fks_Sij: no cnt momenta',
     #         isorsc,xi_i_fks,y_ij_fks
             do i=0,3
               phat_ii(i)=p(i,ii_fks)
             enddo
             E_ii_resc=one
           else
             write(*,*)'Error #8 in fks_Sij',isorsc,xi_i_fks,y_ij_fks
             stop
           endif
         else
           do i=0,3
              phat_ii(i)=p_i_fks_cnt(i,isorsc)
           enddo
           E_ii_resc=xi_i_fks
         endif
      else
         write(*,*)'fks_Sij: do not know what to do',
     #     ii_fks,i_fks,xi_i_fks
      endif
      
      if (firsttime) then
c         firsttime=.false.
         do k = 1,nexternal
           do l = 1,nexternal
             ijskip(k,l) = 0
           enddo
         enddo
         do i=1,nexternal
          if (fks_j_from_i(i,0).ne.0) then
           do j=1,fks_j_from_i(i,0)
            kk = i
            ll = fks_j_from_i(i,j)
            if (nincoming.ne.2 .and. ll.le.nincoming) cycle
            if     ( ijskip(kk,ll).eq.0 .and. ijskip(ll,kk).eq.0 ) then
               ijskip(kk,ll) = 1
            elseif ( ijskip(kk,ll).eq.0 .and. ijskip(ll,kk).eq.1 ) then
               ijskip(kk,ll) = 2
               if(particle_type(kk).ne.8.or.particle_type(ll).ne.8)then
                 write(*,*)'Error #1 in fks_Sij',kk,ll,
     #             particle_type(kk),particle_type(ll)
                 do k=1,nexternal
                    write (*,*) k,(ijskip(k,l),l=1,nexternal)
                 enddo
                 stop
               endif
            else
               write (*,*)'Error #2 in fks_Sij',kk,ll
               stop
            endif
           enddo
          endif
         enddo
      endif

      inverseSij=0d0
      ihdamp=0
      hfact=1.d0
      
      do i=1,nexternal
       if (fks_j_from_i(i,0).ne.0) then
        do j=1,fks_j_from_i(i,0)
         kk = i
         ll = fks_j_from_i(i,j)
         if (nincoming.ne.2 .and. ll.le.nincoming) cycle
         if(ijskip(kk,ll).ne.1)goto 222
         if(particle_type(ll).eq.8.and.particle_type(kk).ne.8.and.
     #      ll.gt.nincoming)then
           write(*,*)'Error #3 in fks_Sij',kk,ll,
     #       particle_type(kk),particle_type(ll)
           stop
         endif
         if( particle_type(ll).ne.8.and.particle_type(kk).ne.8 .and.
     #       pmass(ll).ne.zero.or.pmass(kk).ne.zero )then
           write(*,*)'Error #4 in fks_Sij',kk,ll,
     #       particle_type(kk),particle_type(ll),
     #       pmass(kk),pmass(ll)
           stop
         endif
         if ( (kk.eq.ii_fks .and. ll.eq.jj_fks) .or.
     #        (ll.eq.ii_fks .and. kk.eq.jj_fks) )then
            inverseSij=inverseSij+1d0
            if( particle_type(ll).eq.8 .and.
     #          particle_type(kk).eq.8 .and.
     #          jj_fks.gt.nincoming )then
              z=p(0,ii_fks)/(p(0,ii_fks)+p(0,jj_fks))
              hfact=hfact*h_damp(z)
              ihdamp=ihdamp+1
            endif
         elseif ( kk.eq.ii_fks .and. ll.ne.jj_fks )then
            call get_dkl_Sij(phat_ii,p(0,jj_fks),
     &                p(0,ione),p(0,itwo),one,
     &                ii_fks,particle_type(ii_fks),
     &                jj_fks,particle_type(jj_fks),
     &                ii_fks,jj_fks,ione,
     &                xnum,setsijzero)
            call get_dkl_Sij(phat_ii,p(0,ll),
     &                p(0,ione),p(0,itwo),one,
     &                ii_fks,particle_type(ii_fks),
     &                ll,particle_type(ll),
     &                ii_fks,jj_fks,ione,
     &                xden,setsijzero)
            if(setsijzero)then
              fks_Sij=0d0
              return
            endif
            inverseSij=inverseSij+xnum/xden
         elseif ( ll.eq.ii_fks .and. kk.ne.jj_fks )then
            call get_dkl_Sij(phat_ii,p(0,jj_fks),
     &                p(0,ione),p(0,itwo),one,
     &                ii_fks,particle_type(ii_fks),
     &                jj_fks,particle_type(jj_fks),
     &                ii_fks,jj_fks,ione,
     &                xnum,setsijzero)
            call get_dkl_Sij(phat_ii,p(0,kk),
     &                p(0,ione),p(0,itwo),one,
     &                ii_fks,particle_type(ii_fks),
     &                kk,particle_type(kk),
     &                ii_fks,jj_fks,ione,
     &                xden,setsijzero)
            if(setsijzero)then
              fks_Sij=0d0
              return
            endif
            inverseSij=inverseSij+xnum/xden
         else
            call get_dkl_Sij(phat_ii,p(0,jj_fks),
     &                p(0,ione),p(0,itwo),E_ii_resc,
     &                ii_fks,particle_type(ii_fks),
     &                jj_fks,particle_type(jj_fks),
     &                ii_fks,jj_fks,itwo,
     &                xnum,setsijzero)
            call get_dkl_Sij(p(0,kk),p(0,ll),
     &                p(0,ione),p(0,itwo),one,
     &                kk,particle_type(kk),
     &                ll,particle_type(ll),
     &                ii_fks,jj_fks,itwo,
     &                xden,setsijzero)
            if(setsijzero)then
              fks_Sij=0d0
              return
            endif
            inverseSij=inverseSij+xnum/xden
         endif
 222     continue
        enddo
       endif
      enddo

      if(ihdamp.ne.0.and.ihdamp.ne.1)then
        write(*,*)'Error #5 in fks_Sij',ihdamp
        stop
      endif

      fks_Sij=1d0/inverseSij * hfact

      if(fks_Sij.lt.0.d0.or.fks_Sij.gt.1.d0)then
        write(*,*)'Error #6 in fks_Sij',fks_Sij
        stop
      endif

      return
      end


      subroutine get_dkl_Sij(p1,p2,ka,kb,E1resc,
     #                       i1,itype1,i2,itype2,
     #                       ii_fks,jj_fks,ioneortwo,
     #                       dkl_Sij,setsijzero)
      implicit none
      real*8 p1(0:3),p2(0:3),ka(0:3),kb(0:3),E1resc,dkl_Sij
      integer i1,itype1,i2,itype2,ii_fks,jj_fks,ioneortwo
      logical setsijzero

      include "nexternal.inc"
      include "fks_powers.inc"
      include "coupl.inc"

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      real*8 energy,E1,E2,beta,beta1,beta2,angle,costhfks,vtiny
      parameter (vtiny=1.d-8)

      double precision get_cms_energy

      double precision pmass(nexternal),zero
      parameter (zero=0d0)
      include "pmass.inc"

      if(E1resc.lt.0.d0)then
        write(*,*)'Error #0 in dkl_Sij',E1resc,i1,i2
      endif

      setsijzero=.false.
      dkl_Sij=0.d0

      energy=1.d0
      E1=-1.d0
      E2=-1.d0
      if(ioneortwo.eq.2)then
        if(itype1.eq.8.or.(itype1.ne.8.and.useenergy))then
          E1=get_cms_energy(p1,ka,kb)
          energy=energy*E1*E1resc/(sqrtshat/2d0)
          setsijzero=setsijzero.or.
     #               (E1*E1resc).lt.(vtiny*sqrtshat/2d0)
        elseif(itype1.ne.8.and.(.not.useenergy))then
          energy=energy
        else
          write(*,*)'Error #1 in dkl_Sij',itype1,useenergy
          stop
        endif
      elseif(ioneortwo.ne.1)then
        write(*,*)'Error in dkl_Sij: unknown option',ioneortwo
        stop
      endif
      if(setsijzero)return

      if(itype2.eq.8.or.(itype2.ne.8.and.useenergy))then
        E2=get_cms_energy(p2,ka,kb)
        energy=energy*E2/(sqrtshat/2.d0)
        setsijzero=setsijzero.or.
     #             E2.lt.(vtiny*sqrtshat/2d0)
      elseif(itype2.ne.8.and.(.not.useenergy))then
        energy=energy
      else
        write(*,*)'Error #2 in dkl_Sij',itype2,useenergy
        stop
      endif
      if(setsijzero)return

      if(energy.gt.0.d0)then
        energy=energy**fks_a
      elseif(energy.lt.0.d0)then
        write(*,*)'Error #3 in dkl_Sij',energy
        stop
      endif

      beta=1.d0
      call get_cms_costh_fks(p1,p2,ka,kb,E1,E2,pmass(i1),pmass(i2),
     #                       beta1,beta2,costhfks)
      if(itype1.ne.8.and.pmass(i1).ne.zero .and. usebeta)
     #  beta=beta*beta1
      if(itype2.ne.8.and.pmass(i2).ne.zero .and. usebeta)
     #  beta=beta*beta2
      angle=1-beta*costhfks
      setsijzero=setsijzero.or.angle.lt.vtiny
      if(angle.gt.0.d0)then
        angle=angle**fks_b
      elseif(angle.lt.0.d0)then
        write(*,*)'Error #4 in dkl_Sij',angle
        stop
      endif

      dkl_Sij = energy*angle

      if(dkl_Sij.eq.0.d0.and.(.not.setsijzero))then
        write(*,*)'Error #5 in dkl_Sij'
        stop
      endif

      return
      end



      double precision function get_cms_energy(p,ka,kb)
c Given the momentum p in the \tilde{k}_1+\tilde{k}_2 c.m. frame, returns
c the energy component of p in the k_1+k_2 c.m. frame. Here,
c ka=\tilde{k}_1, ,kb=\tilde{k}_2
      implicit none
      real*8 p(0:3),ka(0:3),kb(0:3)
      double precision dot,xden,xnum,tmp
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      external dot
c
      if(ybst_til_tocm.eq.0.d0)then
        tmp=p(0)
      else
        xden=dot(p,ka)+dot(p,kb)
        xnum=2*dot(ka,kb)
        if(abs(xnum/shat-1.d0).gt.1.d-6)then
          write(*,*)'Inconsistency in get_cms_energy'
          stop
        endif
        tmp=xden/sqrt(xnum)
      endif
      get_cms_energy=tmp
      return
      end



      subroutine get_cms_costh_fks(p1,p2,ka,kb,E1,E2,xm1,xm2,
     #                             beta1,beta2,costhfks)
c Given the momenta p1 and p2 in the \tilde{k}_1+\tilde{k}_2 c.m. frame, 
c returns the velocities beta1 and beta2, and the 3-angle between p1 and p2
c in the k_1+k_2 c.m. frame. Here, ka=\tilde{k}_1, ,kb=\tilde{k}_2
      implicit none
      real*8 p1(0:3),p2(0:3),ka(0:3),kb(0:3)
      real*8 E1,E2,xm1,xm2,beta1,beta2,costhfks
      double precision tmp,costh_fks,get_cms_energy,dot
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      real*8 tiny
      parameter (tiny=1.d-6)
      external dot
c
      if(ybst_til_tocm.eq.0.d0)then
        tmp=costh_fks(p1,p2)
        beta1=sqrt(1-(xm1/p1(0))**2)
        beta2=sqrt(1-(xm2/p2(0))**2)
      else
        if(E1.lt.0.d0)E1=get_cms_energy(p1,ka,kb)
        beta1=sqrt(1-(xm1/E1)**2)
        if(E2.lt.0.d0)E2=get_cms_energy(p2,ka,kb)
        beta2=sqrt(1-(xm2/E2)**2)
        tmp=(1-dot(p1,p2)/(E1*E2))/(beta1*beta2)
        if((abs(tmp)-1.d0).gt.tiny)then
          write(*,*)'Warning in get_cms_costh_fks',tmp
          tmp=sign(1.d0,tmp)
        elseif( (abs(tmp)-1.d0).le.tiny .and.
     #          (abs(tmp)-1.d0).ge.0.d0 )then
          tmp=sign(1.d0,tmp)
        endif
      endif
c
      costhfks=tmp
c
      return
      end



      double precision function costh_fks(p1,p2)
      implicit none

      real*8 p1(0:3),p2(0:3),length1,length2
      real*8 tiny
      parameter (tiny=1.d-6)

      length1=dsqrt(p1(1)**2+p1(2)**2+p1(3)**2)
      length2=dsqrt(p2(1)**2+p2(2)**2+p2(3)**2)
      
      if (length1.ne.0 .and. length2.ne.0)then
         costh_fks=(p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3))/length1/length2
         if((abs(costh_fks)-1.d0).gt.tiny)then
           write(*,*)'Error in costh_fks',costh_fks
           stop
         elseif( (abs(costh_fks)-1.d0).le.tiny .and.
     #           (abs(costh_fks)-1.d0).ge.0.d0 )then
           costh_fks=sign(1.d0,costh_fks)
         endif
      else
         costh_fks=1d0
      endif

      return
      end



      double precision function f_damp(x)
      implicit none
      double precision x,y

      include "fks_powers.inc"

      if (x.lt.0d0 .or. x.gt.1d0)then
         write (*,*) 'ERROR in f_damp',x
         stop
      endif

      if(x.le.one_f_damp)then
         f_damp=1.d0
      else
         y=(x-one_f_damp)/(1.d0-one_f_damp)
         if ( y.lt.0d0 .or. y.gt.1d0 )then
            write (*,*) 'ERROR in f_damp',x,y,one_f_damp
            stop
         endif
         f_damp=(1-y)**(2d0*a_f_damp)/
     &      ( (1-y)**(2d0*a_f_damp)+c_f_damp*y**(2d0*a_f_damp) )
      endif

      return
      end



      double precision function h_damp(x)
      implicit none
      double precision x,y

      include "fks_powers.inc"
      
      if (x.lt.0d0 .or. x.gt.1d0)then
         write (*,*) 'ERROR in h_damp',x
         stop
      endif

      if(x.le.one_h_damp)then
         h_damp=1.d0
      elseif(x.ge.1-one_h_damp)then
         h_damp=0.d0
      else
         y=(x-one_h_damp)/(1.d0-2*one_h_damp)
         if ( y.lt.0d0 .or. y.gt.1d0 )then
            write (*,*) 'ERROR in h_damp',x,y,one_h_damp
            stop
         endif
         h_damp=(1-y)**(2d0*a_h_damp)/
     &      ( (1-y)**(2d0*a_h_damp)+y**(2d0*a_h_damp) )
      endif

      return
      end


      Double precision function fks_Hij(p,ii_fks,jj_fks)
      implicit none

      include "nexternal.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type

      real*8 p(0:3,nexternal),z
      integer ii_fks,jj_fks
      double precision shattmp,dot,h_damp
      external h_damp
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      if(particle_type(jj_fks).ne.8.or.particle_type(ii_fks).ne.8.or.
     &     jj_fks.le.nincoming)then
         fks_Hij=1d0
         return
      endif

      if(p(0,1).le.0.d0)then
c Unphysical kinematics: set H function equal to zero
        fks_Hij=0.d0
        return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(p(0,1),p(0,2))
      else
         shattmp=p(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in fks_Hij: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif
      
      z=p(0,ii_fks)/(p(0,ii_fks)+p(0,jj_fks))
      fks_Hij=h_damp(z)

      return
      end



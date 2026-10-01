      module fks_born_sampling
c Born phase-space trees and beam sampling share one sampling chart.
c Keep intermediate samplers private; the phase-space orchestration
c uses only the tree setup, beam sampler and Born-momentum generator.
      use fks_phase_space_helpers, only: lambda,gentcms,yminmax
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      private
      public generate_momenta_born,initialize_born_chart,
     $     generate_tau_y_wrapper

c The initialized channel supplies this private Born sampling chart.
c It is independent of the masses/momenta of each sampled point.
      integer :: born_tree(2,-max_branch:-1)=0
      integer :: ns_channel=0,nt_channel=0,ionebody=0,nbranch=0
      logical :: one_body=.false.

      contains

      subroutine generate_momenta_born(x,shat_born,sqrtshat_born,totmass,
     $          m,s,
     $          qmass,qwidth,m_born,xpswgt0,xjac0)
      use fks_phase_space_data, only: p_born,p_born_l,p_born_ev
      ! generate the momenta for the reduced born system
      implicit none

      double precision x(99), shat_born, sqrtshat_born, totmass
      double precision S(-max_branch:max_particles),M(-max_branch:max_particles)
      double precision xpswgt0, xjac0
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),
     &                 m_born(nexternal-1)


      logical pass
      double precision pb(0:3,-max_branch:nexternal-1),p_born_CHECK(0:3,nexternal-1)
C
c Conflicting BW stuff
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),
     &     cBW_width(-1:1,-nexternal:-1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level

      integer i,j

      pass = .true.

c Generate the momenta for the initial state of the Born system
      if(nincoming.eq.2) then
        call mom2cx(sqrtshat_born,m(1),m(2),1d0,0d0,pb(0,1),pb(0,2))
      else
         pb(0,1)=sqrtshat_born
         do i=1,2
            pb(i,1)=0d0
         enddo
      endif
      s(-nbranch)  = shat_born
      m(-nbranch)  = sqrtshat_born
      pb(0,-nbranch)= m(-nbranch)
      pb(1,-nbranch)= 0d0
      pb(2,-nbranch)= 0d0
      pb(3,-nbranch)= 0d0
c
c Generate Born-level momenta
c
c Start by generating all the invariant masses of the s-channels
      call generate_inv_mass_sch(ns_channel,born_tree,m,sqrtshat_born
     $     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)

      if (.not.pass) then
         xjac0=-139
         return
      endif
c If only s-channels, also set the p1+p2 s-channel
      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
         s(-nbranch+1)=s(-nbranch)
         m(-nbranch+1)=m(-nbranch)       !Basic s-channel has s_hat
         pb(0,-nbranch+1) = m(-nbranch+1)!and 0 momentum
         pb(1,-nbranch+1) = 0d0
         pb(2,-nbranch+1) = 0d0
         pb(3,-nbranch+1) = 0d0
      endif
c
c     Next do the T-channel branchings
c
      if (nt_channel.ne.0) then
         call generate_t_channel_branchings(ns_channel,nbranch,born_tree
     $        ,m,s,x,pb,xjac0,xpswgt0,pass)
        if (.not.pass) then
           xjac0=-140
           return
        endif
      endif
c
c     Now generate momentum for all intermediate and final states
c     being careful to calculate from more massive to less massive states
c     so the last states done are the final particle states.
c
      call fill_born_momenta(nbranch,nt_channel,one_body,ionebody
     &     ,x,born_tree,m,s,pb,xjac0,xpswgt0,pass)
      if (.not.pass) then
         xjac0=-141
         return
      endif
c
c  Now I have the Born momenta
c
      do i=1,nexternal-1
         do j=0,3
            p_born_l(j,i)=pb(j,i)
            p_born_CHECK(j,i)=pb(j,i)
         enddo
         m_born(i)=m(i)
      enddo
      call phspncheck_born(sqrtshat_born,m_born,p_born_CHECK,pass)
      if (.not.pass) then
         xjac0=-142
         return
      endif

      p_born=p_born_l
      p_born_ev=p_born_l

      return
      end subroutine generate_momenta_born


      subroutine initialize_born_chart(itree)
      implicit none
c Set up the channel once; point generation owns its mass work arrays.
      integer,intent(in) :: itree(2,-max_branch:-1)

      born_tree(:,:) = itree(:,:)
      ionebody=0

      nbranch = nexternal-3 ! nexternal is for n+1-body, while itree uses n-body

c Determine number of s- and t-channel branches, at this point it
c includes the s-channel p1+p2
      ns_channel=1
      do while(itree(1,-ns_channel).ne.1 .and.
     &        itree(1,-ns_channel).ne.2 .and. ns_channel.lt.nbranch)
        ns_channel=ns_channel+1
      enddo
      ns_channel=ns_channel - 1
      nt_channel=nbranch-ns_channel-1
c If no t-channles, ns_channels is one less, because we want to exclude
c the s-channel p1+p2
      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
        ns_channel=ns_channel-1
      endif
c Set one_body to true if it's a 2->1 process at the Born (i.e. 2->2 for the n+1-body)
      if((nexternal-nincoming).eq.2)then
        one_body=.true.
        ionebody=nexternal-1
        ns_channel=0
        nt_channel=0
      elseif((nexternal-nincoming).gt.2)then
        one_body=.false.
      else
        write(*,*)'Error #1 in genps_fks.f',nexternal,nincoming
        stop
      endif

      return
      end subroutine initialize_born_chart


      subroutine generate_inv_mass_sch(ns_channel,itree,m,sqrtshat_born
     $     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)
      implicit none
      integer ns_channel
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision sqrtshat_born,totmass,xjac0
      integer itree(2,-max_branch:-1)
      integer i,j,ii,order(-nexternal:0)
      double precision smin,smax,totalmass
      logical pass
      integer cBW(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),cBW_width(-1:1,
     $     -nexternal:-1)
      double precision s_mass(-nexternal:nexternal)
      common/to_phase_space_s_channel/s_mass
      pass=.true.
      totalmass=totmass
      do ii = -1,-ns_channel,-1
c Randomize the order with which to generate the s-channel masses:
         call sChan_order(ns_channel,order)
         i=order(ii)
c     Generate invariant masses for all s-channel branchings of the Born
         smin = (m(itree(1,i))+m(itree(2,i)))**2
         smax = (sqrtshat_born-totalmass+sqrt(smin))**2
         if(smax.lt.smin.or.smax.lt.0.d0.or.smin.lt.0.d0)then
            write(*,*)'Error #13 in genps_fks.f'
            write(*,*)smin,smax,i
            stop
         endif
         call generate_si(i,smin,smax,s,cBW,cBW_width,cBW_mass,qmass
     &        ,qwidth,x,xjac0,s_mass)
c If numerical inaccuracy, quit loop
         if (xjac0 .lt. 0d0) then
            if ((xjac0.gt.-400d0 .or. xjac0.le.-500d0) .and.
     $           xjac0.ne.0d0)then
               write (*,*) 'WARNING #31 in genps_fks.f',i,s(i),smin,smax
     $              ,xjac0
            endif
            xjac0 = -6
            pass=.false.
            return
         endif
         if (s(i) .lt. smin) then
            write (*,*) 'WARNING #32 in genps_fks.f',i,s(i),smin,smax,x(
     $           -i)
            xjac0=-5
            pass=.false.
            return
         endif
c
c     fill masses, update totalmass
c
         m(i) = sqrt(s(i))
         totalmass=totalmass+m(i)-
     &        m(itree(1,i))-m(itree(2,i))
         if ( totalmass.gt.sqrtshat_born )then
            write (*,*) 'WARNING #33 in genps_fks.f',i,totalmass
     $           ,sqrtshat_born,s(i)
            xjac0 = -4
            pass=.false.
            return
         endif
      enddo
      return
      end subroutine generate_inv_mass_sch


      subroutine generate_si(i,smin,smax,s,cBW,cBW_width,cBW_mass,qmass
     $     ,qwidth,x,xjac0,s_mass)
      implicit none
      integer i
      double precision smin,smax,s(-max_branch:max_particles),qwidth(
     &     -nexternal:0),qmass(-nexternal:0),cBW_width(-1:1,-nexternal:
     &     -1),cBW_mass(-1:1,-nexternal:-1),xjac0,x(99),s_mass(
     &     -nexternal:nexternal)
      integer cBW(-nexternal:-1)
c Choose the appropriate s given our constraints smin,smax
      if(qwidth(i).ne.0.d0 .and. cBW(i).ne.2)then
c Breit Wigner
         if (cBW(i).eq.1 .and.
     &        cBW_width(1,i).gt.0d0 .and. cBW_width(-1,i).gt.0d0) then
c     conflicting BW on both sides
            call trans_x(6,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (cBW(i).eq.1.and.cBW_width(1,i).gt.0d0) then
c     conflicting BW with alternative mass larger
            call trans_x(5,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (cBW(i).eq.1.and.cBW_width(-1,i).gt.0d0) then
c     conflicting BW with alternative mass smaller
            call trans_x(4,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         else
c     normal BW
            call trans_x(3,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         endif
      else
c not a Breit Wigner
         if (smin.eq.0d0 .and. s_mass(i).eq.0d0) then
c     no lower limit on invariant mass from cuts or final state masses:
c     use flat distribution
            call trans_x(1,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (smin.ge.s_mass(i) .and. smin.gt.0d0) then
c     A lower limit on smin, which is larger than lower limit from cuts
c     or masses. Use 1/x importance sampling
            call trans_x(7,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (smin.lt.s_mass(i) .and. s_mass(i).gt.0d0) then
c     Use flat grid between smin and s_mass(i), and 1/x^nsamp above
c     s_mass(i)
            call trans_x(2,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         else
            write (*,*) "ERROR in genps_fks.f:"/
     $           /" cannot set s-channel without BW",i,smin,s_mass(i)
            stop 1
         endif
      endif
      return
      end subroutine generate_si


      subroutine generate_t_channel_branchings(ns_channel,nbranch,itree
     $     ,m,s,x,pb,xjac0,xpswgt0,pass)
c First we need to determine the energy of the remaining particles this
c is essentially in place of the cos(theta) degree of freedom we have
c with the s channel decay sequence
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xjac0,xpswgt0
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision pb(0:3,-max_branch:nexternal-1)
      integer itree(2,-max_branch:-1)
      integer ns_channel,nbranch
      logical pass
c
      double precision totalmass,smin,smax,s1,ma2,mbq,m12,mnq,tmin,tmax
     &     ,t,tmax_temp,phi,dum,dum3(-1:1),s_m,tm,tiny
      parameter (tiny=1d-8)
      integer i,ibranch,idim
      double precision dot
      external dot
      double precision s_mass(-nexternal:nexternal)
      common/to_phase_space_s_channel/s_mass
c
      pass=.true.
      totalmass=0d0
      s_m=0d0
      do ibranch = -ns_channel-1,-nbranch,-1
         totalmass=totalmass+m(itree(2,ibranch))
         s_m=s_m+sqrt(s_mass(itree(2,ibranch)))
      enddo
      m(-ns_channel-1) = dsqrt(S(-nbranch))
c
c Choose invariant masses of the pseudoparticles obtained by taking together
c all final-state particles or pseudoparticles found from the current
c t-channel propagator down to the initial-state particle found at the end
c of the t-channel line.
      do ibranch = -ns_channel-1,-nbranch+2,-1
         totalmass=totalmass-m(itree(2,ibranch))
         smin = totalmass**2
         smax = (m(ibranch) - m(itree(2,ibranch)))**2
         if (smin .gt. smax) then
            xjac0=-3d0
            pass=.false.
            return
         endif
         idim=(nbranch-1+(-ibranch)*2)
         s_m=s_m-sqrt(s_mass(itree(2,ibranch)))
         if (abs(smin-s_m**2).lt.tiny) then
            call trans_x(1,idim,x(idim),smin,smax,s_m**2,dum
     $           ,dum,dum3(-1),dum3(-1),xjac0,s1)
         else
            call trans_x(1,idim,x(idim),smin,smax,s_m**2,dum
     $           ,dum,dum3(-1),dum3(-1),xjac0,s1)
         endif
         if (xjac0.le.0d0) then
            if ((xjac0.gt.-400d0 .or. xjac0.le.-500d0) .and.
     $           xjac0.ne.0d0)then
               write (*,*) 'WARNING #31a in genps_fks.f',ibranch,s1
     $              ,smin,smax,s_m**2,xjac0
            endif
            xjac0 = -6
            pass=.false.
            return
         endif
         m(ibranch-1)=sqrt(s1)
         if (m(ibranch-1)**2.lt.smin.or.m(ibranch-1)**2.gt.smax
     &        .or.m(ibranch-1).ne.m(ibranch-1)) then
            xjac0=-1d0
            pass=.false.
            return
         endif
      enddo
c
c Set m(-nbranch) equal to the mass of the particle or pseudoparticle P
c attached to the vertex (P,t,p2), with t being the last t-channel propagator
c in the t-channel line, and p2 the incoming particle opposite to that from
c which the t-channel line starts
      m(-nbranch) = m(itree(2,-nbranch))
c
c     Now perform the t-channel decay sequence. Most of this comes from:
c     Particle Kinematics Chapter 6 section 3 page 166
c
c     From here, on we can just pretend this is a 2->2 scattering with
c     Pa                    + Pb     -> P1          + P2
c     p(0,itree(ibranch,1)) + p(0,2) -> p(0,ibranch)+ p(0,itree(ibranch,2))
c     M(ibranch) is the total mass available (Pa+Pb)^2
c     M(ibranch-1) is the mass of P2  (all the remaining particles)
c
      do ibranch=-ns_channel-1,-nbranch+1,-1
         s1  = m(ibranch)**2    !Total mass available
         ma2 = m(2)**2
         mbq = dot(pb(0,itree(1,ibranch)),pb(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mnq = m(ibranch-1)**2
         call yminmax(s1,t,m12,ma2,mbq,mnq,tmin,tmax)
         call trans_x(1,-ibranch,x(-ibranch),-tmax,-tmin,s_mass(ibranch)
     $        ,dum,dum,dum3(-1),dum3(-1),xjac0,tm)
         if (xjac0.le.0d0) then
            if ((xjac0.gt.-400d0 .or. xjac0.le.-500d0) .and.
     $           xjac0.ne.0d0)then
               write (*,*) 'WARNING #31b in genps_fks.f',ibranch,tm
     $              ,-tmax,-tmin,xjac0
            endif
            xjac0 = -6
            pass=.false.
            return
         endif
         t=-tm
         if (t .lt. tmin .or. t .gt. tmax) then
            write (*,*) "WARNING #35 in genps_fks.f",t,tmin,tmax
            xjac0=-3d0
            pass=.false.
            return
         endif
         phi = 2d0*pi*x(nbranch+(-ibranch-1)*2)
         xjac0 = xjac0*2d0*pi
c Finally generate the momentum. The call is of the form
c pa+pb -> p1+ p2; t=(pa-p1)**2;   pr = pa-p1
c gentcms(pa,pb,t,phi,m1,m2,p1,pr)
         call gentcms(pb(0,itree(1,ibranch)),pb(0,2),t,phi,
     &        m(itree(2,ibranch)),m(ibranch-1),pb(0,itree(2,ibranch)),
     &        pb(0,ibranch),xjac0)
c
         if (xjac0 .lt. 0d0) then
            write(*,*) 'Failed gentcms',ibranch,xjac0
            pass=.false.
            return
         endif
         xpswgt0 = xpswgt0/(4d0*dsqrt(lambda(s1,ma2,mbq)))
      enddo
c We need to get the momentum of the last external particle.  This
c should just be the sum of p(0,2) and the remaining momentum from our
c last t channel 2->2
      do i=0,3
         pb(i,itree(2,-nbranch)) = pb(i,-nbranch+1)+pb(i,2)
      enddo
      return
      end subroutine generate_t_channel_branchings


      subroutine fill_born_momenta(nbranch,nt_channel,one_body,ionebody
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      integer nbranch,nt_channel,ionebody
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision pb(0:3,-max_branch:nexternal-1)
      integer itree(2,-max_branch:-1)
      double precision xjac0,xpswgt0
      logical pass,one_body
c
      double precision one
      parameter (one=1d0)
      double precision costh,phi,xa2,xb2
      integer i,ix
      double precision dot
      external dot
      double precision vtiny
      parameter (vtiny=1d-12)
c
      pass=.true.
      do i = -nbranch+nt_channel+(nincoming-1),-1
         ix = nbranch+(-i-1)*2+(2-nincoming)
         if (nt_channel .eq. 0) ix=ix-1
         costh= 2d0*x(ix)-1d0
         phi  = 2d0*pi*x(ix+1)
         xjac0 = xjac0 * 4d0*pi
         xa2 = m(itree(1,i))*m(itree(1,i))/s(i)
         xb2 = m(itree(2,i))*m(itree(2,i))/s(i)
         if (m(itree(1,i))+m(itree(2,i)) .ge. m(i)) then
            xjac0=-8
            pass=.false.
            return
         endif
         xpswgt0 = xpswgt0*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI)
         call mom2cx(m(i),m(itree(1,i)),m(itree(2,i)),costh,phi,
     &        pb(0,itree(1,i)),pb(0,itree(2,i)))
c If there is an extremely large boost needed here, skip the phase-space point
c because of numerical stabilities.
         if (dsqrt(abs(dot(pb(0,i),pb(0,i))))/pb(0,i)
     &        .lt.vtiny) then
            xjac0=-81
            pass=.false.
            return
         else
            call boostm(pb(0,itree(1,i)),pb(0,i),m(i),pb(0,itree(1,i)))
            call boostm(pb(0,itree(2,i)),pb(0,i),m(i),pb(0,itree(2,i)))
         endif
      enddo
c
c
c Special phase-space fix for the one_body
      if (one_body) then
c Factor due to the delta function in dphi_1
         xpswgt0=pi/m(ionebody)
c Kajantie's normalization of phase space (compensated below in flux)
         xpswgt0=xpswgt0/(2*pi)
         do i=0,3
            pb(i,3) = pb(i,1)+pb(i,2)
         enddo
      endif
      return
      end subroutine fill_born_momenta


      subroutine trans_x(itype,idim,x,smin,smax,s_mass,qmass,qwidth
     $     ,cBW_mass,cBW_width,jac,s)
c Given the input random number 'x', returns the corresponding value of
c the invariant mass squared 's'.
c
c     itype=1: flat transformation
c     itype=2: flat between 0 and s_mass/stot, 1/x above
c     itype=3: Breit-Wigner
c     itype=4: Conflicting BW, with alternative mass smaller
c     itype=5: Conflicting BW, with alternative mass larger
c     itype=6: Conflicting BW on both sides
c
      implicit none
      integer itype,idim
      double precision x,smin,smax,s_mass,qmass,qwidth,cBW_mass(-1:1)
     $     ,cBW_width(-1:1),jac,s
      double precision fract,A,B,C,bs(-1:1),maxi,mini
      integer j
c
      if (itype.eq.1) then
c     flat transformation:
         A=smax-smin
         B=smin
         s=A*x+B
         jac=jac*A
      elseif (itype.eq.2) then
         fract=0.25d0
         if (s_mass.eq.0d0) then
            write (*,*) 's_mass is zero',itype,idim
         endif
         if (x.lt.fract) then
c     flat transformation:
            if (s_mass.lt.smin) then
               jac=-421d0
               return
            endif
            maxi=min(s_mass,smax)
            A=(maxi-smin)/fract
            B=smin
            s=A*x+B
            jac=jac*A
         else
c     S=A/(B-x) transformation:
            if (s_mass.ge.smax) then
               jac=-422d0
               return
            endif
            mini=max(s_mass,smin)
            A=mini*smax*(1d0-fract)/(smax-mini)
            B=(smax-fract*mini)/(smax-mini)
            s=A/(B-x)
            jac=jac*s**2/A
         endif
      elseif(itype.eq.3) then
c     Normal Breit-Wigner, i.e.
c        \int_smin^smax ds g(s)/((s-qmass^2)^2-qmass^2*qwidth^2) =
c        \int_0^1 dx g(s(x))
         A=atan((qmass-smin/qmass)/qwidth)
         B=atan((qmass-smax/qmass)/qwidth)
         s=qmass*(qmass-qwidth*tan(A-(A-B)*x))
         jac=jac*qmass*qwidth*(A-B)/(cos(A-(A-B)*x))**2
      elseif(itype.eq.4) then
c     Conflicting BW, with alternative mass smaller than current
c     mass. That is, we need to throw also many events at smaller masses
c     than the peak of the current BW. Split 'x' at 'bs(-1)', using a
c     flat distribution below the split, and a BW above the split.
         fract=0.3d0
         bs(-1)=(cBW_mass(-1)-qmass)/
     &        (qwidth+cBW_width(-1)) ! bs(-1) is negative here
         bs(-1)=qmass+bs(-1)*qwidth
         bs(-1)=bs(-1)**2
         if (x.lt.fract) then
            if(smin.gt.bs(-1)) then
               jac=-441d0
               return
            endif
            maxi=min(bs(-1),smax)
            A=(maxi-smin)/fract
            B=smin
            s=A*x+B
            jac=jac*A
         else
            if(smax.lt.bs(-1)) then
               jac=-442d0
               return
            endif
            mini=max(bs(-1),smin)
            A=atan((qmass-mini/qmass)/qwidth)
            B=atan((qmass-smax/qmass)/qwidth)
            C=((1d0-x)*A+(x-fract)*B)/(1d0-fract)
            s=qmass*(qmass-qwidth*tan(C))
            jac=jac*qmass*qwidth*(A-B)/((cos(C))**2*(1d0-fract))
         endif
      elseif(itype.eq.5) then
c     Conflicting BW, with alternative mass larger than current
c     mass. That is, we need to throw also many events at larger masses
c     than the peak of the current BW. Split 'x' at 'bs(1)' and the
c     alternative mass. Use a BW below bs(1), a flat distribution
c     between bs(1) and the alternative mass, and a 1/x above the
c     alternative mass.
         fract=0.35d0
         bs(1)=(cBW_mass(1)-qmass)/
     &        (qwidth+cBW_width(1))
         bs(1)=qmass+bs(1)*qwidth
         bs(1)=bs(1)**2
         if (x.lt.fract) then
            if(smin.gt.bs(1)) then
               jac=-451d0
               return
            endif
            maxi=min(bs(1),smax)
            A=atan((qmass-smin/qmass)/qwidth)
            B=atan((qmass-maxi/qmass)/qwidth)
            C=((B-A)*x+fract*A)/fract
            s=qmass*(qmass-qwidth*tan(C))
            jac=jac*qmass*qwidth*(A-B)/((cos(C))**2*fract)
         elseif (x.lt.1d0-fract) then
            if(smin.gt.cBW_mass(1)**2 .or. smax.lt.bs(1)) then
               jac=-452d0
               return
            endif
            maxi=min(cBW_mass(1)**2,smax)
            mini=max(bs(1),smin)
            A=(maxi-mini)/(1d0-2d0*fract)
            B=((1d0-fract)*mini-fract*maxi)/(1d0-2d0*fract)
            s=A*x+B
            jac=jac*A
         else
            if(smax.le.cBW_mass(1)**2) then
               jac=-453d0
               return
            endif
            mini=max(cBW_mass(1)**2,smin)
            A=mini*smax*fract/(smax-mini)
            B=(smax-(1d0-fract)*mini)/(smax-mini)
            s=A/(B-x)
            jac=jac*s**2/A
         endif
      elseif(itype.eq.6) then
         fract=0.3d0
c     Conflicting BW on both sides. Use flat below bs(-1); BW between
c     bs(-1) and bs(1); flat between bs(1) and alternative mass; and 1/x
c     above alternative mass.
         do j=-1,1,2
            bs(j)=(cBW_mass(j)-qmass)/
     &           (qwidth+cBW_width(j))
            bs(j)=qmass+bs(j)*qwidth
            bs(j)=bs(j)**2
         enddo
         if (x.lt.fract) then
            if(smin.gt.bs(-1)) then
               jac=-461d0
               return
            endif
            maxi=min(bs(-1),smax)
            A=(maxi-smin)/fract
            B=smin
            s=A*x+B
            jac=jac*A
         elseif(x.lt.1d0-fract) then
            if(smin.gt.bs(1) .or. smax.lt.bs(-1)) then
               jac=-462d0
               return
            endif
            maxi=min(bs(1),smax)
            mini=max(bs(-1),smin)
            A=atan((qmass-mini/qmass)/qwidth)
            B=atan((qmass-maxi/qmass)/qwidth)
            C=((1d0-fract-x)*A+(x-fract)*B)/(1d0-2d0*fract)
            s=qmass*(qmass-qwidth*tan(C))
            jac=-jac*qmass*qwidth*(B-A)/((cos(C))**2*(1d0-2d0*fract))
         elseif(x.lt.1d0-fract/2d0) then
            if(smin.gt.cBW_mass(1)**2 .or. smax.lt.bs(1)) then
               jac=-463d0
               return
            endif
            maxi=min(cBW_mass(1)**2,smax)
            mini=max(bs(1),smin)
            A=2d0*(maxi-mini)/fract
            B=2d0*maxi-mini-2d0*(maxi-mini)/fract
            s=A*x+B
            jac=jac*A
         else
            if(smax.le.cBW_mass(1)**2) then
               jac=-464d0
               return
            endif
            mini=max(cBW_mass(1)**2,smin)
            A=mini*smax*fract/(2d0*(smax-mini))
            B=(smax-(1d0-fract/2d0)*mini)/(smax-mini)
            s=A/(B-x)
            jac=jac*s**2/A
         endif
      elseif (itype.eq.7) then
c     S=A/(B-x) transformation:
         if (smin.le.0d0) then
            jac=-471d0
            return
         endif
         A=smin*smax/(smax-smin)
         B=smax/(smax-smin)
         s=A/(B-x)
         jac=jac*s**2/A
      endif
      return
      end subroutine trans_x


      subroutine generate_tau_y_wrapper(
     $     qmass,qwidth,totmass,stot,rndx,tau_born,ycm_born,ycmhat,xjac)
      use fks_phase_space_data,only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg
      ! generates tau and y, calling the functions that correpsond to the
      ! case at hand
      implicit none

      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      double precision totmass, stot
      double precision rndx(2)
      double precision tau_born, ycm_born, ycmhat, xjac
C
      include 'run.inc'
      integer ndim_dummy
      double precision fksmass
c Conflicting BW stuff
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),
     &     cBW_width(-1:1,-nexternal:-1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      logical softtest,colltest
      common/sctests/softtest,colltest


      ndim_dummy=-1 ! this is actually not used anymore

      if (abs(lpp(1)).ne.abs(lpp(2))) then
          write(*,*) 'Different beams not implemented', lpp
          stop 1
      endif
      if (abs(lpp(1)).ge.1 .and. abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         if (abs(lpp(1)).ne.4.and.abs(lpp(1)).ne.3) then ! this is for pp collisions
c x(ndim-1) -> tau_cnt(0); x(ndim) -> ycm_cnt(0)
C rndx(1) -> tau; rndx(2) -> ycm
           if (one_body) then
c tau is fixed by the mass of the final state particle
              call compute_tau_one_body(totmass,stot,tau_born,xjac)
           else
               if(nt_channel.eq.0 .and. qwidth(-ns_channel-1).ne.0.d0 .and.
     $           cBW(-ns_channel-1).ne.2)then
c Generate tau according to a Breit-Wiger function
                 call generate_tau_BW(stot,ndim_dummy,rndx(1),qmass(
     $              -ns_channel-1),qwidth(-ns_channel-1),cBW(-ns_channel
     $              -1),cBW_mass(-1, -ns_channel-1),cBW_width(-1,
     $              -ns_channel-1),tau_born,xjac)
               else
c     not a Breit Wigner
                 call generate_tau(stot,ndim_dummy,rndx(1),tau_born,xjac)
               endif
           endif

c Generate the rapditity of the Born system
           call generate_y(tau_born,rndx(2),ycm_born,ycmhat,xjac)

        else                    ! this is for dressed ee collisions
           call generate_ee_tau_y(rndx(1), rndx(2), one_body, totmass,
     $        stot, nt_channel, qmass(-ns_channel-1),qwidth(-ns_channel-1),
     $        cBW(-ns_channel-1),cBW_mass(-1, -ns_channel-1),
     $        cBW_width(-1,-ns_channel-1),
     $        tau_born, ycm_born, ycmhat, xjac)
           ! for non-physical configurations, xjac=-1000
           if (xjac.eq.-1000d0) return
         endif
      elseif (abs(lpp(1)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x1 not implemented in one_tree'
         stop
      elseif (abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x2 not implemented in one_tree'
         stop
      else
c No PDFs (also use fixed energy when performing tests)
         call compute_tau_y_epem(j_fks,one_body,totmass,stot,
     &        tau_born,ycm_born,ycmhat)
c FI limit tests need nonzero beam momentum available for recoil. Keep
c their Born energy fixed, with xbar=1/2 instead of the usual xbar=1.
c A one-body Born fixes tau from its physical mass.
         if((softtest.or.colltest).and.initial_recoil_leg.gt.0)then
            if(one_body)then
               tau_born=totmass**2/stot
            else
               tau_born=0.25d0
            endif
         endif
         if (j_fks.le.nincoming .and. .not.(softtest.or.colltest)) then
            write (*,*) 'Process has incoming j_fks, but fixed shat: '/
     &           /'not allowed for processes generated at NLO.'
            stop 1
         endif
      endif

      return
      end subroutine generate_tau_y_wrapper


      subroutine compute_tau_one_body(totmass,stot,tau,jac)
      implicit none
      double precision totmass,stot,tau,jac,roH
      roH=totmass**2/stot
      tau=roH
c Jacobian due to delta() of tau_born
      jac=jac*2*totmass/stot
      return
      end subroutine compute_tau_one_body


      subroutine generate_tau_BW(stot,idim,x,mass,width,cBW,BWmass
     $     ,BWwidth,tau,jac)
      use fks_phase_space_data, only: tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
      implicit none
      integer cBW,idim
      double precision stot,x,tau,jac,mass,width,BWmass(-1:1),BWwidth(
     $     -1:1),s_mass,s
      double precision smax,smin
      if (cBW.eq.1 .and. width.gt.0d0 .and. BWwidth(1).gt.0d0) then
         smin=tau_Born_lower_bound*stot
         smax=stot
         s_mass=smin
         call trans_x(5,idim,x,smin,smax,s_mass,mass,width,BWmass(
     $        -1),BWwidth(-1),jac,s)
         tau=s/stot
         jac=jac/stot
      else
         smin=tau_Born_lower_bound*stot
         smax=stot
         s_mass=smin
         call trans_x(3,idim,x,smin,smax,s_mass,mass,width,BWmass(
     $        -1),BWwidth(-1),jac,s)
         tau=s/stot
         jac=jac/stot
      endif
      return
      end subroutine generate_tau_BW


      subroutine generate_tau(stot,idim,x,tau,jac)
      use fks_phase_space_data, only: tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
      use mc_native_context, only: native_mapping
      implicit none
      integer idim
      double precision x,tau,jac,smin,smax,s_mass,s,tiny,dum,dum3(-1:1)
     $     ,stot
      parameter (tiny=1d-8)
! A flat auxiliary map is finite also when the physical threshold is zero.
      if(native_mapping)then
         tau=x
         return
      endif
      smin=tau_born_lower_bound*stot
      smax=stot
      s_mass=tau_lower_bound_resonance*stot
      if (s_mass.gt.smin*(1d0+tiny)) then
         call trans_x(2,idim,x,smin,smax,s_mass,dum,dum
     $        ,dum3,dum3,jac,s)
      elseif(abs(s_mass-smin).lt.tiny*smin) then
         call trans_x(7,idim,x,smin,smax,s_mass,dum,dum
     $        ,dum3,dum3,jac,s)
      else
         write (*,*) 'ERROR #39 in genps_fks.f',s_mass,smin,smax
         jac=-1d0
      endif
      tau=s/stot
      jac=jac/stot
      return
      end subroutine generate_tau


      subroutine generate_y(tau,x,ycm,ycmhat,jac)
      implicit none
      double precision tau,x,ycm,jac
      double precision ylim,ycmhat
      ylim=-0.5d0*log(tau)
      ycmhat=2*x-1
      ycm=ylim*ycmhat
      jac=jac*ylim*2
      return
      end subroutine generate_y


      subroutine compute_tau_y_epem(j_fks,one_body,fksmass,
     &                              stot,tau,ycm,ycmhat)
      implicit none
      integer j_fks
      logical one_body
      double precision fksmass,stot,tau,ycm,ycmhat
      if(j_fks.le.nincoming)then
c This should never happen in normal integration: when no PDFs, j_fks
c cannot be initial state (but needed for testing). If tau set to one,
c integration range in xi_i_fks will be zero, so lower it artificially
c when too large
         if(one_body)then
            tau=fksmass**2/stot
         else
            tau=max((0.85d0)**2,fksmass**2/stot)
         endif
         ycm=0.d0
      else
c For e+e- collisions, set tau to one and y to zero
         tau=1.d0
         ycm=0.d0
      endif
      ycmhat=0.d0
      return
      end subroutine compute_tau_y_epem


      subroutine get_tau_y_from_x12(x1, x2, omx1, omx2, tau, ycm, ycmhat, jac)
      use fks_phase_space_data, only: tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
      implicit none
      double precision x1, x2, omx1, omx2, tau, ycm, ycmhat, jac
      double precision ylim
      double precision tolerance
      parameter (tolerance=1e-3)
      double precision y_settozero
      parameter (y_settozero=1e-12)
      double precision lx1, lx2

      tau = x1*x2

      ! ycm=-log(tau)/2 ;  ylim = log(x1/x2)/2
      if (1d0-x1.gt.tolerance) then
        lx1 = dlog(x1)
      else
        lx1 = -omx1-omx1**2/2d0-omx1**3/3d0-omx1**4/4d0-omx1**5/5d0
      endif
      ylim = -0.5d0*lx1
      ycm = 0.5d0*lx1

      if (1d0-x2.gt.tolerance) then
        lx2 = dlog(x2)
      else
        lx2 = -omx2-omx2**2/2d0-omx2**3/3d0-omx2**4/4d0-omx2**5/5d0
      endif
      ylim = ylim-0.5d0*lx2
      ycm = ycm-0.5d0*lx2

      ycmhat = ycm / ylim

      ! this is to prevent numerical inaccuracies
      ! when botn x->1
      if (ylim.lt.y_settozero) then
        ylim = 0d0
        ycm = 0d0
        ycmhat = 1d0
      endif

      if (abs(ycmhat).gt.1d0) then
        if (abs(ycmhat).gt.1d0 + tolerance) then
          write(*,*) 'ERROR YCMHAT', ycmhat, x1, x2
          stop 1
        else
          ycmhat = sign(1d0, ycmhat)
        endif
      endif

      if (tau.lt.tau_born_lower_bound) then
        write(*,*) 'get_tau_y_from_x12: Warning, unphysical tau',
     $  tau, tau_born_lower_bound
        jac = -1000d0
      endif

      return
      end subroutine get_tau_y_from_x12


      subroutine generate_x_ee(rnd, xmin, x, omx, jac)
      implicit none
      ! generates the momentum fraction with importance
      !  sampling suitable for ee collisions
      ! rnd is generated uniformly in [0,1],
      ! x is generated according to (1 -rnd)^-expo, starting
      ! from xmin
      ! jac is the corresponding jacobian
      ! omx is 1-x, stored to improve numerical accuracy
      double precision rnd, x, omx, jac, xmin
      double precision expo
      double precision get_ee_expo
      double precision tolerance
      parameter (tolerance=1.d-5)

      expo = get_ee_expo()

      x = 1d0 - rnd ** (1d0/(1d0-expo))
      omx = rnd ** (1d0/(1d0-expo))
      if (x.ge.1d0) then
        if (x.lt.1d0+tolerance) then
          x=1d0
        else
          write(*,*) 'ERROR in generate_x_ee', rnd, x
          stop 1
        endif
      endif
      jac = 1d0/(1d0-expo)
      ! then rescale it between xmin and 1
      x = x * (1d0 - xmin) + xmin
      omx = omx * (1d0 - xmin)
      jac = jac * (1d0 - xmin)**(1d0-expo)

      return
      end subroutine generate_x_ee


      subroutine generate_ee_tau_y(rnd1_in, rnd2_in, one_body, totmass,
     $     stot, nt_channel, qmass, qwidth, cBW, cBW_mass, cBW_width,
     $     tau_born, ycm_born, ycmhat, xjac0)
      use fks_phase_space_data, only: tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
      implicit none
      double precision rnd1_in, rnd2_in
      double precision rnd1, rnd2, totmass, stot, qmass, qwidth
      double precision cBW_mass(-1:1), cBW_width(-1:1)
      integer nt_channel, cBW
      logical one_body
      double precision tau_born, ycm_born, ycmhat, xjac0

      logical bw_exists, generate_with_bw
      common /to_ee_generatebw/ generate_with_bw
      double precision frac_bw
      parameter (frac_bw=0.5d0)
      integer idim_dum
C dressed lepton stuff
      double precision x1_ee, x2_ee, jac_ee

      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee


      double precision get_ee_expo
      double precision tau_m, tau_w
      double precision omtau_born

      ! these common blocks are never used
      ! we leave them here for the moment
      ! as e.g. one may want to plot random numbers, etc.
      double precision r1, r2, x1bk, x2bk
      common /to_random_numbers/r1,r2, x1bk, x2bk
      logical use_evpr
      common /to_use_evpr/use_evpr

      ! copy the random numbers, as they may be rescaled
      ! (avoids side effects)
      rnd1=rnd1_in
      rnd2=rnd2_in

      ! these lines store the random numbers in the common
      ! block (may be removed)
      r1=rnd1
      r2=rnd2

      ! define the analogous of tau for mass and width
      tau_m = qmass**2/stot
      tau_w = qwidth**2/stot

      bw_exists = nt_channel.eq.0.and.qwidth.ne.0.d0.and.cBW.ne.2
     $ .and.tau_m.lt.1d0.and.totmass.lt.qmass

      generate_with_bw=.false.

      ! if there are BWs, decide whether to generate flat or
      ! to use the BW-specific generation (half and half, or
      ! as determined by frac_bw)
      if (bw_exists) then
        generate_with_bw = rnd1.lt.frac_bw
        if (generate_with_bw) then
            rnd1 = rnd1 / frac_bw
            xjac0 = xjac0 / frac_bw
        else
            rnd1 = (rnd1 - frac_bw) / (1d0 - frac_bw)
            xjac0 = xjac0 / (1d0 - frac_bw)
        endif
      endif

      if (one_body) then
        write(*,*) 'one body with ee collisions not implemented'
        stop 1
      endif

      ! if tau is generated accodring to a BW, then force the momentum
      ! mapping with event projection
      use_evpr = generate_with_bw

      if(generate_with_bw) then
        ! here we treat the case of resonances

        ! first generate tau with the dedicated function
        idim_dum = 1000 ! this is never used in practice
        call generate_tau_BW(stot,idim_dum,rnd1,qmass,qwidth,cBW,cBW_mass,
     $       cBW_width,tau_born,xjac0)
        ! multiply the jacobian by a multichannel factor
        xjac0 = xjac0 * (1d0/((tau_born-tau_m)**2 + tau_m*tau_w)) /
     $       ( 1d0/((tau_born-tau_m)**2 + tau_m*tau_w) + (1d0-tau_born)**(1d0-2*get_ee_expo()))

        ! then pick either x1 or x2 and generate it the usual way;
        ! Note that:
        ! - setting xmin=sqrt(tau_born) ensures that the largest
        !    bjorken x is being generated.
        ! -  there is a jacobian for x1 x2 -> tau x1(2)
        ! -  we must include the factor 1/(1-x)^get_ee_expo,
        !    (x is the bjorken x which is not generated)
        !    because the compute_eepdf function assumes that
        !    this is the case in general
        if (rnd2.lt.0.5d0) then
          call generate_x_ee(rnd2*2d0, dsqrt(tau_born), x1_ee, omx_ee(1), jac_ee)
          x2_ee = tau_born / x1_ee
          omx_ee(2) = 1d0 - x2_ee
          xjac0 = xjac0 / x1_ee * 2d0 * jac_ee / (1d0-x2_ee)**get_ee_expo()
        else
          call generate_x_ee(1d0-2d0*(rnd2-0.5d0), dsqrt(tau_born), x2_ee, omx_ee(2), jac_ee)
          x1_ee = tau_born / x2_ee
          omx_ee(1) = 1d0 - x1_ee
          xjac0 = xjac0 / x2_ee * 2d0  * jac_ee / (1d0-x1_ee)**get_ee_expo()
        endif
      else
        ! standard (without resonances) generation:
        ! for dressed ee collisions the generation is different
        ! wrt the pp case. In the pp case, tau and y_cm are generated,
        ! while in the ee case x1 and x2 are generated first.

        call generate_x_ee(rnd1, tau_born_lower_bound,
     $      x1_ee, omx_ee(1), jac_ee)
        xjac0 = xjac0 * jac_ee
        call generate_x_ee(rnd2, tau_born_lower_bound/x1_ee,
     $      x2_ee, omx_ee(2), jac_ee)
        xjac0 = xjac0 * jac_ee

        tau_born = x1_ee * x2_ee
        ! better numerical accuracy
        omtau_born = omx_ee(1) + omx_ee(2) - omx_ee(1)*omx_ee(2)
        ! multiply the jacobian by a multichannel factor if the
        ! generation with resonances is also possible
        if (bw_exists) xjac0 = xjac0 * (omtau_born)**(1d0-2*get_ee_expo()) /
     $       ( 1d0/((tau_born-tau_m)**2 + tau_m*tau_w) + (omtau_born)**(1d0-2*get_ee_expo()))
      endif

      ! Check here if the bjorken x's are physical (may not be so
      ! because of instabilities
      if (x1_ee.gt.1d0.or.x2_ee.gt.1d0) then
        write(*,*) 'generate_ee_tau_y: Warning, unphysical x:',
     $   x1_ee, x2_ee, generate_with_bw
        xjac0 = -1000d0
        return
      endif

      ! now we are done. We must call the following function
      ! in order to (re-)generate tau and ycm
      ! from x1 and x2. It also (re-)checks that tau_born
      ! is pysical, and otherwise sets xjac0=-1000
      call get_tau_y_from_x12(x1_ee, x2_ee, omx_ee(1), omx_ee(2), tau_born, ycm_born, ycmhat, xjac0)

      x1bk=x1_ee
      x2bk=x2_ee

      return
      end subroutine generate_ee_tau_y

      end module fks_born_sampling

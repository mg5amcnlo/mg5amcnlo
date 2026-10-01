c Born phase-space trees, invariant-mass sampling and branchings.
c External routine interfaces and COMMON layouts are shared with
c genps_fks.f; keep coordinate and counterevent conventions consistent.

      subroutine generate_momenta_born(x,shat_born,sqrtshat_born,totmass,
     $          m,s,
     $          qmass,qwidth,m_born,xpswgt0,xjac0)
      ! generate the momenta for the reduced born system
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'

      double precision x(99), shat_born, sqrtshat_born, totmass
      double precision S(-max_branch:max_particles),M(-max_branch:max_particles)
      double precision xpswgt0, xjac0
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),
     &                 m_born(nexternal-1)

      integer itree(2,-max_branch:-1)
      integer ns_channel, nt_channel, ionebody, nbranch
      logical one_body
      common/born_trees/itree,ns_channel,nt_channel,ionebody,nbranch,one_body

      logical pass
      double precision pb(0:3,-max_branch:nexternal-1),p_born_CHECK(0:3,nexternal-1)
C
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision p_born_l(0:3,nexternal-1)
      common/pborn_l/p_born_l
      double precision p_born_ev(0:3,nexternal-1)
      common/pborn_ev/p_born_ev
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
      call generate_inv_mass_sch(ns_channel,itree,m,sqrtshat_born
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
         call generate_t_channel_branchings(ns_channel,nbranch,itree
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
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
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
      end


      subroutine fill_genmom_born_commons(itree,m)
      implicit none
      include 'genps.inc'
c arguments
      integer itree(2,-max_branch:-1)
      double precision M(-max_branch:max_particles)

      include 'nexternal.inc'

      integer itree_c(2,-max_branch:-1)
      integer ns_channel, nt_channel, ionebody, nbranch
      logical one_body
      common/born_trees/itree_c,ns_channel,nt_channel,ionebody,nbranch,one_body

      itree_c(:,:) = itree(:,:)

      nbranch = nexternal-3 ! nexternal is for n+1-body, while itree uses n-body

c Determine number of s- and t-channel branches, at this point it
c includes the s-channel p1+p2
      ns_channel=1
      do while(itree(1,-ns_channel).ne.1 .and.
     &        itree(1,-ns_channel).ne.2 .and. ns_channel.lt.nbranch)
        m(-ns_channel)=0d0
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
      end


      subroutine generate_inv_mass_sch(ns_channel,itree,m,sqrtshat_born
     $     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
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
      end


      subroutine generate_si(i,smin,smax,s,cBW,cBW_width,cBW_mass,qmass
     $     ,qwidth,x,xjac0,s_mass)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
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
      end


      subroutine generate_t_channel_branchings(ns_channel,nbranch,itree
     $     ,m,s,x,pb,xjac0,xpswgt0,pass)
c First we need to determine the energy of the remaining particles this
c is essentially in place of the cos(theta) degree of freedom we have
c with the s channel decay sequence
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
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
      double precision lambda,dot
      external lambda,dot
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
      end


      subroutine fill_born_momenta(nbranch,nt_channel,one_body,ionebody
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
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
      double precision lambda,dot
      external lambda,dot
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
      end


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
      end

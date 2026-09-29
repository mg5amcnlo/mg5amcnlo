c Main program for the generated leptonic process's limit-test executable.
c Replace its main program with this one, retaining the helper routines
c from test_soft_col_limits.f. Supply two graph selections, both equal 1.
c Probe the extra ISR limits with underlying Born flavours not included
c in u b > d b e+ ve. The real matrix element has a 1/pt**2 pole and
c Sij=1 in both cases, so these corners must be removed by the cuts.
      program probe_uncovered_limits
      use mint_module
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal),plotp(0:4,nexternal),
     $     beta(3),br(0:3),gm,b2,bp,pt,energy,mrest,ej,wgt,sij,
     $     xi,yy,mj,fks_Sij,xf,yf,previous,scaled,closure(0:3)
      integer istatus(nexternal),fks_j_from_i(nexternal,0:nexternal),
     $     particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision ylab,ycm,sqrtshat,shat
      common/parton_cms_stuff/ylab,ycm,sqrtshat,shat
      common/cxiyfix/xf,yf
      integer nstep,n,loop,i,spect,bmin,bmax,hard1,hard2
      logical dummy_cuts,accepted
      external fks_Sij,dummy_cuts
      if(nexternal.ne.7.or.nincoming.ne.2)stop 1
      xf=-2d0
      yf=-2d0
      call init_test_limits(2,nstep)
      do loop=5,6
         call init_new_loop(loop,bmin,bmax,mj)
         iconfig=1
         call init_iconfig_loop(2)
         if(loop.eq.5)then
            spect=5
            hard1=3
            hard2=7
         else
            spect=3
            hard1=5
            hard2=7
         endif
         previous=0d0
         do n=-1,5
            pt=10d0**(-0.5d0*n)
            if(n.eq.-1)pt=40d0
            p=0d0
            p(:,1)=(/500d0,0d0,0d0,500d0/)
            p(:,2)=(/500d0,0d0,0d0,-500d0/)
            p(:,spect)=(/100d0,pt,0d0,sqrt(1d4-pt**2)/)
            if(loop.eq.6)p(3,spect)=-p(3,spect)
            energy=900d0
            beta=-p(1:3,spect)/energy
            b2=sum(beta**2)
            gm=1d0/sqrt(1d0-b2)
            mrest=energy/gm
            ej=(mrest-80d0)/2d0
            p(:,hard1)=(/ej,ej,0d0,0d0/)
            p(:,hard2)=(/ej,-ej,0d0,0d0/)
            p(:,4)=(/40d0,0d0,-40d0,0d0/)
            p(:,6)=(/40d0,0d0,40d0,0d0/)
            do i=3,nexternal
               if(i.eq.spect)cycle
               br=p(:,i)
               bp=sum(beta*br(1:3))
               p(0,i)=gm*(br(0)+bp)
               p(1:3,i)=br(1:3)+((gm-1d0)*bp/b2+
     $              gm*br(0))*beta
            enddo
            closure=sum(p(:,1:2),dim=2)-sum(p(:,3:nexternal),dim=2)
            if(maxval(abs(closure)).gt.1d-9)stop 2
            shat=1d6
            sqrtshat=1d3
            ylab=0d0
            ycm=0d0
            call smatrix_real(p,wgt)
            xi=2d0*p(0,7)/sqrtshat
            yy=p(3,7)/p(0,7)
            if(loop.eq.6)yy=-yy
            sij=fks_Sij(p,7,loop-4,xi,yy)
            plotp=0d0
            plotp(0:3,:)=p
            istatus=1
            istatus(1:2)=-1
            accepted=dummy_cuts(plotp,istatus,pdg_type)
            write(*,'(a,2i3,1x,l1,4es22.12)')
     $           'CORNER ',loop,spect,accepted,pt,wgt,wgt*pt**2,sij
            if(accepted.neqv.(n.eq.-1))stop 3
            if(abs(sij-1d0).gt.1d-12)stop 4
            if(.not.(wgt.gt.0d0.and.wgt.lt.1d30))stop 5
            scaled=wgt*pt**2
            if(n.ge.4)then
               if(abs(scaled/previous-1d0).gt.1d-3)stop 6
            endif
            previous=scaled
         enddo
      enddo
      write(*,*)'PASS validation cuts reject both extra ISR poles'
      end

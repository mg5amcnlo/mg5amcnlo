      subroutine grandmother_fks(iconfig,nbranch,ns_channel,nt_channel,
     #                           i_fks,j_fks,searchforgranny,
     #                           fksmother,fksgrandmother,fksaunt,
     #                           is_beta_cms,is_granny_sch,topdown)
c
c Given iconfig, nbranch, ns_channel, nt_channel, i_fks, j_fks and
c searchforgranny (if .false., finds the mother and exits)
c this routine returns
c
c  fksmother: the mother of i_fks and j_fks
c  fksgrandmother: the mother of fksmother and fksaunt
c  fksaunt: the sister of fksmother
c  is_beta_cms: true if the 1->2 (for s-channel) or 2->2 (for t-channel)
c    scattering which results into the mother is the partonic c.m., 
c    false otherwise
c  is_granny_sch: true if grandmother is an s-channel, false otherwise
c  topdown: if true the fixed vector (p_a) in the iterative construction of
c    t-channel scattering is p_2; if false, p_a=p_1
c
c For initial-state singularities, fksgrandmother and fksaunt are
c undefined (set to zero), and fksmother=1 or 2. For final-state
c singularities, fksmother<0; fksgrandmother<0 in all cases except when the 
c mother is attached to parton 1 and 2, in which case fksgrandmother=1 or 2.
c fksaunt can have either sign, being positive when it is a single particle, 
c and negative when it is a set of particles
c
      implicit none
      integer iconfig,nbranch,ns_channel,nt_channel
      integer i_fks,j_fks
      integer fksmother, fksgrandmother,fksaunt
      logical searchforgranny,is_beta_cms,is_granny_sch,topdown

      integer i,itmp

      include "genps.inc"
      include "nexternal.inc"
      integer            mapconfig(0:lmaxconfigs), this_config
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)

      include "configs.inc"

      topdown=.false.

c This function need be called before the redefinition of ns_channel,
c which takes place if no t-channels are present
      if((ns_channel+nt_channel).ne.(nbranch-1))then
        write(*,*)'Error #1 in grandmother_fks',
     #    nbranch,ns_channel,nt_channel
        stop
      endif
      if( i_fks.lt.3.or.i_fks.gt.nexternal .or.
     #    j_fks.lt.1.or.j_fks.gt.nexternal )then
        write(*,*)'Error #2 in grandmother_fks',i_fks,j_fks
        stop
      endif
c
      fksmother=0
      fksgrandmother=0
      fksaunt=0
c Initial-state singularity
      if(j_fks.eq.1)then
        fksmother=1
        if( iforest(1,-ns_channel-1,iconfig).ne.1 .or.
     #      iforest(2,-ns_channel-1,iconfig).ne.i_fks .or.
     #      nt_channel.eq.0 )then
c This diagram has no singularities associated with the (i_fks,j_fks) pair
           fksmother=0
           return
        endif
c Initial-state singularity
      elseif(j_fks.eq.2)then
        fksmother=2
        topdown=.true.
        if(((iforest(1,-nbranch,iconfig).gt.0 .or.
     &       iforest(2,-nbranch,iconfig).ne.i_fks).and.
     &      (iforest(1,-ns_channel-1,iconfig).ne.2 .or. !This is needed, because of inverted t-channel
     &       iforest(2,-ns_channel-1,iconfig).ne.i_fks)) .or.
     &      nt_channel.eq.0 )then
c This diagram has no singularities associated with the (i_fks,j_fks) pair
           fksmother=0
           return
        endif
c Final-state singularity
      elseif(j_fks.ge.3)then
c Mother must be an s-channel
        do i=-1,-ns_channel,-1
          if( (iforest(1,i,iconfig).eq.i_fks.and.
     #         iforest(2,i,iconfig).eq.j_fks) .or.
     #        (iforest(2,i,iconfig).eq.i_fks.and.
     #         iforest(1,i,iconfig).eq.j_fks) )then
            fksmother=i
          endif
        enddo
        if(fksmother.eq.0)then
c No mother found: this diagram has no singularities associated with
c the (i_fks,j_fks) pair
          return
        elseif(fksmother.gt.0)then
          write(*,*)'Error #5 in grandmother_fks',fksmother
          stop
        endif
        if(.not.searchforgranny)return
c Look for grandmother and aunt; s-channels first
        do i=-1,-ns_channel,-1
          if(iforest(1,i,iconfig).eq.fksmother)then
            fksgrandmother=i
            fksaunt=iforest(2,i,iconfig)
          elseif(iforest(2,i,iconfig).eq.fksmother)then
            fksgrandmother=i
            fksaunt=iforest(1,i,iconfig)
          endif
        enddo
c s-channel grandmother found
        if(fksgrandmother.ne.0.and.fksaunt.ne.0)then
          if(fksgrandmother.gt.0)then
            write(*,*)'Error #6 in grandmother_fks',
     #        fksgrandmother,fksaunt
            stop
          endif
          is_granny_sch=.true.
          is_beta_cms=(fksgrandmother.eq.-ns_channel)
c s-channel grandmother not found; search t-channels
        else
          is_granny_sch=.false.
          do i=-ns_channel-1,-nbranch,-1
            if(iforest(2,i,iconfig).eq.fksmother)then
              fksgrandmother=iforest(1,i,iconfig)
              fksaunt=i
            endif
          enddo
          if(fksgrandmother.eq.0.or.fksaunt.eq.0)then
            write(*,*)'Error #7 in grandmother_fks',
     #        fksgrandmother,fksaunt
            stop
          endif
          if(fksgrandmother.le.0.or.fksaunt.eq.0)then
            write(*,*)'Error #8 in grandmother_fks',
     #        fksgrandmother,fksaunt
            stop
          endif
          if(fksaunt.eq.-nbranch)fksaunt=2
c Grandmother must be closer than aunt to one of the incoming partons
          if( fksaunt.eq.2.or. (fksgrandmother.ne.1.and.
     #          (-fksgrandmother-ns_channel).gt.int(nt_channel/2)))then
            itmp=fksgrandmother
            fksgrandmother=fksaunt
            fksaunt=itmp
            topdown=.false.
          else
            topdown=.true.
          endif
          is_beta_cms=( (fksgrandmother.eq.1) .or.
     #                  (fksgrandmother.eq.2) )
        endif
      endif
      return
      end


      subroutine rotate_invar(pin,pout,cth,sth,cphi,sphi)
c Given the four momentum pin, returns the four momentum pout (in the same
c Lorentz frame) by performing a three-rotation of an angle theta 
c (cos(theta)=cth) around the y axis, followed by a three-rotation of an
c angle phi (cos(phi)=cphi) along the z axis. The components of pin
c and pout are given along these axes
      implicit none
      real*8 cth,sth,cphi,sphi,pin(0:3),pout(0:3)
      real*8 q1,q2,q3
c
      q1=pin(1)
      q2=pin(2)
      q3=pin(3)
      pout(1)=q1*cphi*cth-q2*sphi+q3*cphi*sth
      pout(2)=q1*sphi*cth+q2*cphi+q3*sphi*sth
      pout(3)=-q1*sth+q3*cth 
      pout(0)=pin(0)
      return
      end


      subroutine trp_rotate_invar(pin,pout,cth,sth,cphi,sphi)
c This subroutine performs a rotation in the three-space using a rotation
c matrix that is the transpose of that used in rotate_invar(). Thus, if
c called with the *same* angles, trp_rotate_invar() acting on the output
c of rotate_invar() will return the input of rotate_invar()
      implicit none
      real*8 cth,sth,cphi,sphi,pin(0:3),pout(0:3)
      real*8 q1,q2,q3
c
      q1=pin(1)
      q2=pin(2)
      q3=pin(3)
      pout(1)=q1*cphi*cth+q2*sphi*cth-q3*sth
      pout(2)=-q1*sphi+q2*cphi 
      pout(3)=q1*cphi*sth+q2*sphi*sth+q3*cth
      pout(0)=pin(0)
      return
      end


      subroutine getaziangles(p,cphi,sphi)
      implicit none
      real*8 p(0:3),cphi,sphi
      real*8 xlength,cth,sth
      double precision rho
      external rho
c
      xlength=rho(p)
      if(xlength.ne.0.d0)then
        cth=p(3)/xlength
        sth=sqrt(1-cth**2)
        if(sth.ne.0.d0)then
          cphi=p(1)/(xlength*sth)
          sphi=p(2)/(xlength*sth)
        else
          cphi=1.d0
          sphi=0.d0
        endif
      else
        cphi=1.d0
        sphi=0.d0
      endif
      return
      end


      subroutine phspncheck(npart,ecm,xmass,xmom)
c Checks four-momentum conservation.
c WARNING: works only in the partonic c.m. frame
      implicit none
      integer npart,maxmom
      include "genps.inc"
      include "nexternal.inc"
      real*8 ecm,xmass(-max_branch:max_particles),
     # xmom(0:3,-max_branch:max_particles)
      real*8 tiny,xm,xlen4,den,xsum(0:3),xsuma(0:3),xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      integer jflag,i,j,jj
c
      jflag=0
      do i=0,3
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=3,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(i.eq.0)xsum(i)=xsum(i)-ecm
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=0,3)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=0,3)
        stop
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(ptmp(0).ge.1.d0)then
          den=ptmp(0)
        else
          den=1.d0
        endif
        if(abs(xm-xmass(j))/den.gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          stop
        endif
      enddo
      return
      end


      subroutine phspncheck_born(ecm,xmass,xmom)
c Identical to phspncheck
      implicit none
      include 'nexternal.inc'
      real*8 ecm,xmass(nexternal-1),xmom(0:3,nexternal-1)
      real*8 tiny,xm,xlen4,xsum(0:3),xsuma(0:3),xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      integer jflag,npart,i,j,jj
c
      jflag=0
      npart=nexternal-1
      do i=0,3
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=3,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(i.eq.0)xsum(i)=xsum(i)-ecm
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=0,3)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=0,3)
        stop
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(abs(xm-xmass(j))/ptmp(0).gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          stop
        endif
      enddo
      return
      end


      subroutine phspncheck_nocms(npart,ecm,xmass,xmom)
c Checks four-momentum conservation. Derived from phspncheck;
c works in any frame
      implicit none
      integer npart,maxmom
      include "genps.inc"
      include "nexternal.inc"
      real*8 ecm,xmass(-max_branch:max_particles),
     # xmom(0:3,-max_branch:max_particles)
      real*8 tiny,vtiny,xm,xlen4,den,ecmtmp,xsum(0:3),xsuma(0:3),
     # xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-6)
      integer jflag,i,j,jj
      double precision dot
      external dot
c
      jflag=0
      do i=0,3
        xsum(i)=-xmom(i,1)-xmom(i,2)
        xsuma(i)=abs(xmom(i,1))+abs(xmom(i,2))
        do j=3,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved [nocms]'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=0,3)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=0,3)
        stop
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(ptmp(0).ge.1.d0)then
          den=ptmp(0)
        else
          den=1.d0
        endif
        if(abs(xm-xmass(j))/den.gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation [nocms]'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          stop
        endif
      enddo
c
      ecmtmp=sqrt(2d0*dot(xmom(0,1),xmom(0,2)))
      if(abs(ecm-ecmtmp).gt.vtiny)then
        write(*,*)'Inconsistent shat [nocms]'
        write(*,*)'ecm given=   ',ecm
        write(*,*)'ecm computed=',ecmtmp
        write(*,'(4(d14.8,1x))') (xmom(jj,1),jj=0,3)
        write(*,'(4(d14.8,1x))') (xmom(jj,2),jj=0,3)
        stop
      endif

      return
      end


      function xlen4(v)
      implicit none
      real*8 xlen4,tmp,v(0:3)
c
      tmp=v(0)**2-v(1)**2-v(2)**2-v(3)**2
      xlen4=sign(1.d0,tmp)*sqrt(abs(tmp))
      return
      end


      subroutine get_emother_range(int_nlo,i_fks,j_fks,nbranch,
     #                       xi_i_fks,y_ij_fks,s,m,
     #                       xi_mother_min_evpc,xi_mother_max_evpc,
     #                       xi_mother_min_ev,xi_mother_max_ev,
     #                       xi_mother_min_cnt,xi_mother_max_cnt,
     #                       got_emother_range)
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      logical int_nlo,got_emother_range,test
      integer i_fks,j_fks,nbranch,maxcnt,i
      double precision xi_i_fks,y_ij_fks
      double precision xi_mother_min_evpc,xi_mother_max_evpc
      double precision xi_mother_min_ev,xi_mother_max_ev
      double precision xi_mother_min_cnt(-2:4),xi_mother_max_cnt(-2:4)
      double precision M(-max_branch:max_particles)
      double precision S(-max_branch:0)
      double precision zero,one
      parameter (zero=0.d0)
      parameter (one=1.d0)
c
      xi_mother_min_ev=3.d0
      xi_mother_max_ev=-1.d0
      xi_mother_min_cnt(4)=3.d0
      xi_mother_max_cnt(4)=-1.d0
      if(m(j_fks).eq.0.d0)then
        maxcnt=2
        call get_emother_j0(i_fks,j_fks,nbranch,xi_i_fks,y_ij_fks,s,m,
     #                      xi_mother_min_ev,xi_mother_max_ev)
        if(int_nlo)then
          call get_emother_j0(i_fks,j_fks,nbranch,zero,y_ij_fks,s,m,
     #                        xi_mother_min_cnt(0),xi_mother_max_cnt(0))
          call get_emother_j0(i_fks,j_fks,nbranch,xi_i_fks,one,s,m,
     #                        xi_mother_min_cnt(1),xi_mother_max_cnt(1))
          call get_emother_j0(i_fks,j_fks,nbranch,zero,one,s,m,
     #                        xi_mother_min_cnt(2),xi_mother_max_cnt(2))
        else
          xi_mother_min_cnt(0)=3.d0
          xi_mother_max_cnt(0)=-1.d0
          xi_mother_min_cnt(1)=3.d0
          xi_mother_max_cnt(1)=-1.d0
          xi_mother_min_cnt(2)=3.d0
          xi_mother_max_cnt(2)=-1.d0
        endif
        xi_mother_min_cnt(4)=min(xi_mother_min_cnt(0),
     #                           xi_mother_min_cnt(1),
     #                           xi_mother_min_cnt(2))
        xi_mother_max_cnt(4)=max(xi_mother_max_cnt(0),
     #                           xi_mother_max_cnt(1),
     #                           xi_mother_max_cnt(2))
      else
        maxcnt=0
        call get_emother_jm(i_fks,j_fks,nbranch,xi_i_fks,y_ij_fks,s,m,
     #                      xi_mother_min_ev,xi_mother_max_ev)
        if(int_nlo)then
          call get_emother_jm(i_fks,j_fks,nbranch,zero,y_ij_fks,s,m,
     #                        xi_mother_min_cnt(0),xi_mother_max_cnt(0))
        else
          xi_mother_min_cnt(0)=3.d0
          xi_mother_max_cnt(0)=-1.d0
        endif
        xi_mother_min_cnt(4)=xi_mother_min_cnt(0)
        xi_mother_max_cnt(4)=xi_mother_max_cnt(0)
      endif
      got_emother_range=xi_mother_min_ev.ne.3.d0 .and.
     #                  xi_mother_max_ev.ne.-1.d0
      if(int_nlo)got_emother_range=got_emother_range .or.
     #                  ( xi_mother_min_cnt(4).ne.3.d0 .and.
     #                    xi_mother_max_cnt(4).ne.-1.d0 )
      if(int_nlo)then
        test=.true.
c test will remain true if all cnts have the same (meaningful) minimum
        do i=0,maxcnt
          test=test.and.
     #         ( xi_mother_min_cnt(i).eq.3.d0 .or.
     #           xi_mother_min_cnt(i).eq.xi_mother_min_cnt(4) )
        enddo
        if(.not.test)then
          xi_mother_min_evpc=min(xi_mother_min_ev,xi_mother_min_cnt(4))
          xi_mother_min_cnt(4)=xi_mother_min_evpc
        else
          xi_mother_min_evpc=xi_mother_min_ev
        endif
c Now do the same for the maxima
        test=.true.
        do i=0,maxcnt
          test=test.and.
     #         ( xi_mother_max_cnt(i).eq.-1.d0 .or.
     #           xi_mother_max_cnt(i).eq.xi_mother_max_cnt(4) )
        enddo
        if(.not.test)then
          xi_mother_max_evpc=max(xi_mother_max_ev,xi_mother_max_cnt(4))
          xi_mother_max_cnt(4)=xi_mother_max_evpc
        else
          xi_mother_max_evpc=xi_mother_max_ev
        endif
      else
        xi_mother_min_evpc=xi_mother_min_ev
        xi_mother_max_evpc=xi_mother_max_ev
      endif
      return
      end


      subroutine get_emother_j0(i_fks,j_fks,nbranch,xi_i_fks,y_ij_fks,s,m,
     #                          xi_mother_min,xi_mother_max)
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      integer i_fks,j_fks,nbranch
      double precision xi_i_fks,y_ij_fks,xi_mother_min,xi_mother_max
      double precision M(-max_branch:max_particles)
      double precision S(-max_branch:0)
      double precision xii,yij,scm,sigm2
      integer i
c
      xii=xi_i_fks
      yij=y_ij_fks
      scm=s(-nbranch)
      sigm2=0.d0
      do i=3,nexternal
        if(i.ne.i_fks.and.i.ne.j_fks)sigm2=sigm2+m(i)
      enddo
      sigm2=sigm2**2
      xi_mother_min=xii
      xi_mother_max=(2-xii**2*(1-yij)-2*sigm2/scm)/(2-xii*(1-yij))
      return
      end


      subroutine get_emother_jm(i_fks,j_fks,nbranch,xi_i_fks,y_ij_fks,s,m,
     #                          xi_mother_min,xi_mother_max)
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      integer i_fks,j_fks,nbranch
      double precision xi_i_fks,y_ij_fks,xi_mother_min,xi_mother_max
      double precision M(-max_branch:max_particles)
      double precision S(-max_branch:0)
      double precision xii,yij,scm,sigm2,xmassj,pfact,xiamaxbar,
     # xfact,sarg,xia1,xia2,xiamax,xiamin
      integer i
c
      xii=xi_i_fks
      yij=y_ij_fks
      scm=s(-nbranch)
      sigm2=0.d0
      do i=3,nexternal
        if(i.ne.i_fks.and.i.ne.j_fks)sigm2=sigm2+m(i)
      enddo
      sigm2=sigm2**2
      xmassj=m(j_fks)
      pfact=2-xii**2-2*(sigm2-xmassj**2)/scm
      if(pfact.lt.0.d0)then
        write(*,*)'Fatal error #1 in get_emother_jm'
        write(*,*)xii,sigm2,xmassj,scm
        stop
      endif
      xiamaxbar=pfact/(2-xii)
      xfact=4-4*xii+xii**2*(1-yij**2)
      sarg=(pfact-xii*(2-xii))**2-4*xmassj**2/scm*xfact
      if(sarg.gt.0.d0)then
        xia1=( (2-xii)*pfact-xii*yij*(xii**2*yij+
     #         sign(1.d0,yij)*sqrt(sarg)) )/xfact
        xia2=( (2-xii)*pfact-xii*yij*(xii**2*yij-
     #         sign(1.d0,yij)*sqrt(sarg)) )/xfact
        if(xia2.lt.xia1)then
          write(*,*)'Fatal error #2 in get_emother_jm'
          write(*,*)xia1,xia2
          stop
        endif
      endif
      if(sarg.le.0.d0)then
        xiamax=xiamaxbar
      else
        if(yij.ge.0.d0)then
          if(xia2.gt.xiamaxbar)then
            xiamax=min(xiamaxbar,xia1)
          else
            xiamax=xiamaxbar
          endif
        else
          if(xia2.lt.xiamaxbar)then
            xiamax=xiamaxbar
          elseif(xia2.ge.xiamaxbar.and.xia1.lt.xiamaxbar)then
            xiamax=xia2
          elseif(xia1.ge.xiamaxbar)then
            xiamax=xia2
          else
            write(*,*)'Fatal error #3 in get_emother_jm'
            write(*,*)xia1,xia2,xiamaxbar
            stop
          endif
        endif
        if( xmassj.eq.0.d0 .and.
     #      ( yij.gt.0.d0.and.xiamax.ne.xia1 .or.
     #        yij.lt.0.d0.and.xiamax.ne.xia2 ) )then
          write(*,*)'Fatal error #4 in get_emother_jm'
          write(*,*)xia1,xia2,xiamax,yij
          stop
        endif
      endif
      xiamin=xii+2*xmassj/sqrt(scm)
      xi_mother_min=xiamin
      xi_mother_max=xiamax
      return
      end


      subroutine get_smother(E_i_fks,y_ij_fks,sgrandmother,xmaunt,xmassj,
     #  smother,dsmotherody)
      implicit none
      double precision E_i_fks,y_ij_fks,sgrandmother,xmaunt,xmassj,
     # smother,dsmotherody,ei,yij,xkbe2,sxkbe2,xmassk2,xmassj2,solden,
     # solnuma,argofsqrtred,xkal2,sqxkal2,dxkal2dy0
      double precision pi
      parameter (pi=3.1415926535897932385d0)
c
      ei=E_i_fks
      yij=y_ij_fks
      xkbe2=sgrandmother
      sxkbe2=sqrt(xkbe2)
      xmassk2=xmaunt**2
      xmassj2=xmassj**2
c
      solden=-(sxkbe2-ei*(1-yij))*(sxkbe2-ei*(1+yij))
      solnuma=-( ei*(xkbe2-2*ei*sxkbe2-xmassk2)*
     #           (sxkbe2-ei*(1-yij**2))+
     #           xmassj2*sxkbe2*(sxkbe2-ei) )
      argofsqrtred=xkbe2*( (xkbe2-2*ei*sxkbe2-xmassk2)**2 -
     #      2*xmassj2*(xkbe2-2*ei*(sxkbe2-ei*(1-yij**2))-
     #      xmassj2/2.d0+xmassk2) )
      if(argofsqrtred.lt.0.d0)then
        smother=-1.d0
      else
        smother=( solnuma+ei*yij*sqrt(argofsqrtred) )/
     #          solden
        xkal2=smother
        sqxkal2=sqrt(smother)
        dxkal2dy0=2*ei*(xkbe2-2*ei*sxkbe2-xmassk2)*yij/solden-
     #   xkbe2*( ( (xkbe2-2*ei*sxkbe2-xmassk2)**2 -
     #     2*xmassj2*(xkbe2-2*ei*(sxkbe2-ei*(1-yij**2))-
     #     xmassj2/2.d0+xmassk2) )+
     #   4*ei**2*yij**2*xmassj2 )/
     #   ( sqrt(xkbe2*( (xkbe2-2*ei*sxkbe2-xmassk2)**2 -
     #          2*xmassj2*(xkbe2-2*ei*(sxkbe2-ei*(1-yij**2))-
     #          xmassj2/2.d0+xmassk2) ))*solden) +
     #   2*ei*yij/solden*xkal2
        dsmotherody=dxkal2dy0
      endif
      return
      end



      subroutine link_to_born2(itree,spropR,tpridR,iconfig,i_fks,j_fks,
     &                        mother,nbranchin,ns_channelin,
     &                        nt_channelin,mapbconf,bconf,biforest)
c Given a real configuration in itree and FKS partons i_fks, j_fks and their
c mother, returns the Born configuration corresponding to that real
c configuration bconf, read from the include file born_conf.inc.
c mapbconf and biforest are the mapconfig and iforest for the Born.
c (iconfig is only used to write the error messages.)
      implicit none
      include 'genps.inc'      
      include "nexternal.inc"
      integer itree(2,-max_branch:-1),BornFromRealTree(2,-max_branch:-1)
      integer i_fks,j_fks,iconfig,bconf,mapbconf(0:lmaxconfigs)
      integer biforest(2,-max_branch:-1,lmaxconfigs)
      integer spropR(-max_branch:-1),ns_channelin,nt_channelin
      integer tpridR(-max_branch:-1),ns_channel,nt_channel
     
      integer i,j,k,l,mother,nbranch,nbranchin,bbranch,surrogate
      integer bup(-max_branch:nexternal),jb,found
      logical done,bornfound(lmaxconfigs),firsttime
      data firsttime/.true./

      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer spropBfR(-max_branch:-1)
      integer tpridBfR(-max_branch:-1)
      integer mapconfig(0:lmaxconfigs)
      include 'born_conf.inc'

      save bornfound

      nbranch=nbranchin-1
      surrogate=0

      if (mother.eq.1) then
         do i=-1,-nbranch,-1
            if (itree(1,i).eq.j_fks.and.itree(2,i).eq.i_fks) then
               surrogate=i
               nt_channel=nt_channelin-1
               ns_channel=ns_channelin
               if (.not.(surrogate.lt.0))then
                  write (*,*) 'Error in link_to_Born',itree(j_fks,i_fks)
                  stop
               endif
            endif
         enddo
      elseif (mother.eq.2) then
         do i=-1,-nbranch,-1
            if (itree(1,i).eq.j_fks.and.itree(2,i).eq.i_fks) then
               surrogate=i
            endif
         enddo
         if (surrogate.eq.0) then
            surrogate=-nbranch 
         endif
         nt_channel=nt_channelin-1
         ns_channel=ns_channelin
      else
        surrogate=mother
        nt_channel=nt_channelin
        ns_channel=ns_channelin-1 
      endif

c Map the itree to the default form of born tree
      do j=-1,-nbranch,-1
         do i=1,2
            if (j.gt.surrogate)then
               if (itree(i,j).lt.surrogate) then
                  BornFromRealTree(i,j)=itree(i,j)+1
               elseif(itree(i,j).eq.surrogate) then
                  BornFromRealTree(i,j)=min(i_fks,j_fks)
               elseif(itree(i,j).gt.surrogate .and. itree(i,j).lt.0) then
                  BornFromRealTree(i,j)=itree(i,j)
               elseif(itree(i,j).lt.max(i_fks,j_fks) .and.
     &                 itree(i,j).gt.0) then
                  BornFromRealTree(i,j)=itree(i,j)
               elseif(itree(i,j).gt.max(i_fks,j_fks)) then
                  BornFromRealTree(i,j)=itree(i,j)-1
               endif
               if (-j.le.ns_channel) then
                  spropBfR(j)=spropR(j)
               else
                  tpridBfR(j)=tpridR(j)
               endif
            elseif(j.lt.surrogate)then
               if (itree(i,j).lt.surrogate) then
                  BornFromRealTree(i,j+1)=itree(i,j)+1
               elseif(itree(i,j).eq.surrogate) then
                  BornFromRealTree(i,j+1)=min(i_fks,j_fks)
               elseif(itree(i,j).gt.surrogate .and. itree(i,j).lt.0) then
                  BornFromRealTree(i,j+1)=itree(i,j)
               elseif(itree(i,j).lt.max(i_fks,j_fks) .and.
     &                 itree(i,j).gt.0 ) then
                  BornFromRealTree(i,j+1)=itree(i,j)
               elseif(itree(i,j).gt.max(i_fks,j_fks)) then
                  BornFromRealTree(i,j+1)=itree(i,j)-1
               endif
               if (-j-1.le.ns_channel) then
                  spropBfR(j+1)=spropR(j)
               else
                  tpridBfR(j+1)=tpridR(j)
               endif
            endif
         enddo
      enddo

      bbranch=nbranch-1  ! Might need to be changed for t channels

c labels for the external particles should be the same between
c the BornFromReal and the Born
      do jb=1,nexternal-1
         bup(jb)=jb
      enddo

c compare the default born tree with the born_conf.inc trees.
      if (firsttime)then
         mapbconf(0)=mapconfig(0)
         do k=1,mapconfig(0)
            bornfound(k)=.false.
            mapbconf(k)=mapconfig(k)
            do i=1,2
               do j=-1,-bbranch,-1
                  biforest(i,j,k) = iforest(i,j,k)
               enddo
            enddo
         enddo
         firsttime=.false.
      endif
      k=1
      done=.false.
      l=1
      do while (.not.done)
         do while (bornfound(k)) ! Always try a new Born
            k=k+1
         enddo
         found=0
         jb=-1
         j=-1
         do while (j.ge.-bbranch .and. jb .ge.-bbranch)
            if ( (bup(BornFromRealTree(1,jb)).eq.iforest(1,j,k).and.
     &            bup(BornFromRealTree(2,jb)).eq.iforest(2,j,k)  ).or.
     &           (bup(BornFromRealTree(1,jb)).eq.iforest(2,j,k).and.
     &            bup(BornFromRealTree(2,jb)).eq.iforest(1,j,k)   ))then
               bup(jb)=j        ! jb in BornFromReal corresponds to j in Borns
c Also check that the intermediate propagators agree
               if (-j.le.ns_channel) then
                  if(sprop(j,k).eq.spropBfR(jb)) then
                     found=found+1
                  else
                     write (*,*) 'skip due to sprop'
                  endif
               else
c For t-channel: particle vs anti-particle is ambiguous. Better use absolute ID for checking
                  if(abs(tprid(j,k)).eq.abs(tpridBfR(jb))) then 
                     found=found+1
                  else
                     write (*,*) 'skip due to tprid'
                  endif
               endif
               if (found.ne.-jb) jb=-bbranch-1 ! goto next Born confi.
               jb=jb-1          ! go to next line in BornFromRealTree
               j=0              ! reset the line in iforest
            endif
            j=j-1
         enddo
         if (found.eq.bbranch)then
            write (*,*) 'found Born conf',k,' for Real conf',iconfig
            bornfound(k)=.true.
            done=.true.
            bconf=k
         endif
 132     continue
         if (k.gt.mapconfig(0))then
            write (*,*) 'FATAL ERROR 1 in link_to_born:'
     &                //' no Born found for config',iconfig
            stop
         endif
         k=k+1
      enddo


      return
      end






    

      subroutine invert_order_iforest_REAL(nbranch,ns_channel,
     &     nt_channel,iconfig)
c This routine inverts the order of the t-channel (from 'top->bottom'
c to 'bottom->top', ie now starting from particle 2) for configuration
c iconfig in configs.inc and props.inc
c It's assumed that the final state branch connecting to particle 2
c is a single particle, ie with a positive id.
c Note that this routine OVERWRITES the existing configs.inc and props.inc
c files, which can be dangerous if this routine is
c called more than once in successive runs.
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      integer nbranch,nt_channel,ns_channel,iconfig
      character*76 buff,printout(1000000)
      character*60 buff60
      character*38 buff38
      character*37 buff37
      character*33 buff33
      character*32 buff32
      character*23 buff23
      character*16 buff16,file
      character*9 dummy9
      character*13 dummy13
      character*16 empty60
      parameter (empty60='                ')
      character*38 empty38
      parameter (empty38='                                      ')
      character*39 empty37
      parameter (empty37='                                       ')
      character*43 empty33
      parameter (empty33='                                           ')
      character*44 empty32
      parameter (empty32='                                            ')
      character*53 empty23
      parameter (
     &  empty23='                                                     ')
      character*60 empty16
      parameter (empty16='                        '//
     &     '                                    ')
      integer number1,number2,curgraph,i,l
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'configs.inc'
      logical done
      write(*,*) "--inverting real configs"

      if (nt_channel.gt.0) then
      open (unit=88,file='configs.inc',status='old')
      rewind (88)
      done=.false.
      l=0
      do while (.not.done)
         read(88,'(a76)',end=192) buff
         l=l+1
         write(printout(l),'(a76)'),buff
         if (buff(1:13).eq.'C     Diagram') then
         write(*,*) "diagrem found in configs.inc"
c found a new graph           
            read(buff,'(a13,i4)') dummy13, curgraph
            if ( curgraph.eq.mapconfig(iconfig) ) then
c this is the graph we are looking for               
               read(88,'(a33)') buff33
               l=l+1
               write(printout(l),'(a76)'),buff33//empty33
c s-channels always come first -> do nothing
               do i=1,ns_channel*2
                  read(88,'(a76)') buff
                  l=l+1
                  write(printout(l),'(a76)'),buff
               enddo
c then the t-channel, for which we have to reverse the order
               do i=1,nt_channel*2+1
                  if (i.eq.1) then
                     read(88,'(a76)') buff
                     number1=2
                     number2=max(iforest(1,-nbranch,iconfig),
     &                    iforest(2,-nbranch,iconfig))
                     write(buff(40:46),'(i3,a1,i3)') number1,',',number2
                     l=l+1
                     write(printout(l),'(a76)'),buff
                  elseif (i.eq.nt_channel*2+1) then
                     read(88,'(a60)') buff60
                     number1=-ns_channel-(i-1)/2
                     number2=iforest(1,-nbranch+(i-1)/2,iconfig)
                     if (number2.eq.1) then
                        number2=iforest(2,-nbranch+(i-1)/2,iconfig)
                     endif
                     write(buff60(40:46),'(i3,a1,i3)')
     &                    number1,',',number2
                     l=l+1
                     write(printout(l),'(a76)'),buff60//empty60
                  elseif(mod(i,2).eq.1) then
                     read(88,'(a76)') buff
                     number1=-ns_channel-(i-1)/2
                     number2=max(iforest(1,-nbranch+(i-1)/2,iconfig),
     &                    iforest(2,-nbranch+(i-1)/2,iconfig))
                     if (number2.eq.1) then
                        number2=min(iforest(1,-nbranch+(i-1)/2,iconfig),
     &                       iforest(2,-nbranch+(i-1)/2,iconfig))
                     endif
                     write(buff(40:46),'(i3,a1,i3)') number1,',',number2
                     l=l+1
                     write(printout(l),'(a76)'),buff
                  elseif(mod(i,2).eq.0) then
                     read(88,'(a38)') buff38
                     write(buff38(30:36),'(i6)')
     &                    tprid(-nbranch+i/2,iconfig)
                     l=l+1
                     write(printout(l),'(a76)'),buff38//empty38
                  endif
               enddo
            endif
         endif
      enddo
 192  continue
      rewind(88)
      if (l.gt.1000000) then
         write (*,*)
     &        'too many lines in configs.inc for t-channel inversion'
         stop
      endif
      do i=1,l
         write(88,'(a76)') printout(i)
      enddo
      close(unit=88)
      endif
      

      l=0
c Order of t-channel propagators has changed.
c Also rewrite props.inc
      if (nt_channel.gt.1) then
         done=.false.
         open (unit=88,file='props.inc',status='old')
         rewind(88)
         do while (.not. done)
            read (88,'(a76)',end=193) buff
            l=l+1
            write(printout(l),'(a76)'),buff
            if (buff(1:12).eq.'      PMASS(') then
              write(*,*)"propagator found in props.inc"
               read (buff(17:20),'(i4)') curgraph
               if (curgraph.eq.mapconfig(iconfig)) then
                  l=l-1
                  backspace(88)
c s-channels always come first               
                  do i=1,ns_channel*3
                     read (88,'(a76)') buff
                     l=l+1
                     write(printout(l),'(a76)'),buff
                  enddo
c then the t-channel
                  do i=1,nt_channel
                     read(88,'(a37)') buff37
                     write(buff37(13:15),'(i3)') -nbranch+i
                     if (index(buff37,'ABS').gt.0) then
                        l=l+1
                        write(printout(l),'(a76)'),buff37//empty37
                     else
                        l=l+1
                        write(printout(l),'(a76)'),buff37(1:32)//empty32
                     endif
                     read(88,'(a38)') buff38
                     write(buff38(14:16),'(i3)') -nbranch+i
                     if (index(buff38,'ABS').gt.0) then
                        l=l+1
                        write(printout(l),'(a76)'),buff38//empty38
                     else
                        l=l+1
                        write(printout(l),'(a76)'),buff38(1:33)//empty33
                     endif
                     read(88,'(a23)') buff23
                     write(buff23(11:13),'(i3)') -nbranch+i
                     l=l+1
                     write(printout(l),'(a76)'),buff23//empty23
                  enddo
               endif
            endif
         enddo
 193     continue
         rewind(88)
         if (l.gt.1000000) then
            write (*,*)
     &           'too many lines in props.inc for t-channel inversion'
            stop
         endif
         do i=1,l
            write(88,'(a76)') printout(i)
         enddo
         close(unit=88)
      endif


      return
      end


      subroutine invert_order_iforest_BORN(nbranch,ns_channel,
     &     nt_channel,iconfig)
c This routine inverts the order of the t-channel (from 'top->bottom'
c to 'bottom->top', ie now starting from particle 2) for Born configuration
c borngraph in born_props.inc and born_conf.inc.
c It's assumed that the final state branch connecting to particle 2
c is a single particle, ie with a positive id.
c Note that this routine OVERWRITES the existing born_conf.inc
c and born_props.inc files, which can be dangerous if this routine is
c called more than once in successive runs.
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      integer nbranch,nt_channel,ns_channel,iconfig
      character*76 buff,printout(1000000)
      character*60 buff60
      character*38 buff38
      character*37 buff37
      character*33 buff33
      character*32 buff32
      character*23 buff23
      character*16 buff16,file
      character*9 dummy9
      character*13 dummy13
      character*16 empty60
      parameter (empty60='                ')
      character*38 empty38
      parameter (empty38='                                      ')
      character*39 empty37
      parameter (empty37='                                       ')
      character*43 empty33
      parameter (empty33='                                           ')
      character*44 empty32
      parameter (empty32='                                            ')
      character*53 empty23
      parameter (
     &  empty23='                                                     ')
      character*60 empty16
      parameter (empty16='                        '//
     &     '                                    ')
      integer number1,number2,curgraph,i,l
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'born_conf.inc'
      logical done
      write(*,*) "--inverting born configs"


      if (nt_channel.gt.0) then
      open (unit=88,file='born_conf.inc',status='old')
      rewind (88)
      done=.false.
      l=0
      do while (.not.done)
         read(88,'(a76)',end=192) buff
         l=l+1
         write(printout(l),'(a76)'),buff
         if (buff(1:13).eq.'C     Diagram') then
         write(*,*) "diagrem found in born_conf.inc"
c found a new graph           
            read(buff,'(a13,i4)') dummy13, curgraph
               if (curgraph.eq.mapconfig(iconfig)) then
c this is the graph we are looking for               
               read(88,'(a33)') buff33
               l=l+1
               write(printout(l),'(a76)'),buff33//empty33
c s-channels always come first -> do nothing
               do i=1,ns_channel*2
                  read(88,'(a76)') buff
                  l=l+1
                  write(printout(l),'(a76)'),buff
               enddo
c then the t-channel, for which we have to reverse the order
               do i=1,nt_channel*2+1
                  if (i.eq.1) then
                     read(88,'(a76)') buff
                     number1=2
                     number2=max(iforest(1,-nbranch,iconfig),
     &                    iforest(2,-nbranch,iconfig))
                     write(buff(40:46),'(i3,a1,i3)') number1,',',number2
                     l=l+1
                     write(printout(l),'(a76)'),buff
                  elseif (i.eq.nt_channel*2+1) then
                     read(88,'(a60)') buff60
                     number1=-ns_channel-(i-1)/2
                     number2=iforest(1,-nbranch+(i-1)/2,iconfig)
                     if (number2.eq.1) then
                        number2=iforest(2,-nbranch+(i-1)/2,iconfig)
                     endif
                     write(buff60(40:46),'(i3,a1,i3)')
     &                    number1,',',number2
                     l=l+1
                     write(printout(l),'(a76)'),buff60//empty60
                  elseif(mod(i,2).eq.1) then
                     read(88,'(a76)') buff
                     number1=-ns_channel-(i-1)/2
                     number2=max(iforest(1,-nbranch+(i-1)/2,iconfig),
     &                    iforest(2,-nbranch+(i-1)/2,iconfig))
                     if (number2.eq.1) then
                        number2=min(iforest(1,-nbranch+(i-1)/2,iconfig),
     &                       iforest(2,-nbranch+(i-1)/2,iconfig))
                     endif
                     write(buff(40:46),'(i3,a1,i3)') number1,',',number2
                     l=l+1
                     write(printout(l),'(a76)'),buff
                  elseif(mod(i,2).eq.0) then
                     read(88,'(a38)') buff38
                     write(buff38(30:36),'(i6)')
     &                    tprid(-nbranch+i/2,iconfig)
                     l=l+1
                     write(printout(l),'(a76)'),buff38//empty38
                  endif
               enddo
            endif
         endif
      enddo
 192  continue
      rewind(88)
      if (l.gt.1000000) then
         write (*,*)
     &        'too many lines in born_conf.inc for t-channel inversion'
         stop
      endif
      do i=1,l
         write(88,'(a76)') printout(i)
      enddo
      close(unit=88)
      endif
      
      l=0
c Order of t-channel propagators has changed.
c Also rewrite props.inc
      if (nt_channel.gt.1) then
         done=.false.
         open (unit=88,file='born_props.inc',status='old')
         rewind(88)
         do while (.not. done)
            read (88,'(a76)',end=193) buff
            l=l+1
            write(printout(l),'(a76)'),buff
            if (buff(1:12).eq.'      PMASS(') then
              write(*,*)"propagator found in born_props.inc"
               read (buff(17:20),'(i4)') curgraph
               if (curgraph.eq.mapconfig(iconfig)) then
                  l=l-1
                  backspace(88)
c s-channels always come first               
                  do i=1,ns_channel*3
                     read (88,'(a76)') buff
                     l=l+1
                     write(printout(l),'(a76)'),buff
                  enddo
c then the t-channel
                  do i=1,nt_channel
                     read(88,'(a37)') buff37
                     write(buff37(13:15),'(i3)') -nbranch+i
                     if (index(buff37,'ABS').gt.0) then
                        l=l+1
                        write(printout(l),'(a76)'),buff37//empty37
                     else
                        l=l+1
                        write(printout(l),'(a76)'),buff37(1:32)//empty32
                     endif
                     read(88,'(a38)') buff38
                     write(buff38(14:16),'(i3)') -nbranch+i
                     if (index(buff38,'ABS').gt.0) then
                        l=l+1
                        write(printout(l),'(a76)'),buff38//empty38
                     else
                        l=l+1
                        write(printout(l),'(a76)'),buff38(1:33)//empty33
                     endif
                     read(88,'(a23)') buff23
                     write(buff23(11:13),'(i3)') -nbranch+i
                     l=l+1
                     write(printout(l),'(a76)'),buff23//empty23
                  enddo
               endif
            endif
         enddo
 193     continue
         rewind(88)
         if (l.gt.1000000) then
            write (*,*)
     &           'too many lines in born_props.inc '
     &           //'for t-channel inversion'
            stop
         endif
         do i=1,l
            write(88,'(a76)') printout(i)
         enddo
         close(unit=88)
      endif


      return
      end





      double precision function dsig(pp,wgt,vegaswgt)
c Here are the subtraction terms, the Sij function, 
c the f-damping function, and the single diagram
c enhanced multi-channel factor included
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "fks.inc"
      include "fks_powers.inc"
      include 'coupl.inc'

      double precision pp(0:3,nexternal),wgt,vegaswgt

      double precision fks_Sij,f_damp,dot,dlum
      external fks_Sij,f_damp,dot,dlum

      double precision x,xtot,s_ev,s_c,s_s,s_sc,ffact,fx_ev,fx_c,
     #                 fx_s,fx_sc,ev_wgt,cnt_wgt_c,cnt_wgt_s,cnt_wgt_sc,
     #                 bsv_wgt,plot_wgt,cnt_swgt_s,cnt_swgt_sc,cnt_sc,cnt_s,
     #                 prefact_cnt_ssc,prefact_cnt_ssc_c,prefact_coll,
     #                 prefact_coll_c,born_wgt,prefact_deg,prefact,prefact_c,
     #                 prefact_deg_sxi,prefact_deg_slxi,deg_wgt,deg_swgt,
     #                 deg_xi_c,deg_lxi_c,deg_xi_sc,deg_lxi_sc,
     #                 cnt_swgt,cnt_wgt,xlum_ev,xlum_c,xlum_s,xlum_sc
      integer i,iplot

      integer izero,ione,itwo,mohdr,iplot_ev,iplot_cnt,iplot_born
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)
      parameter (mohdr=-100)
      parameter (iplot_ev=11)
      parameter (iplot_cnt=12)
      parameter (iplot_born=20)

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision shattmp
      double precision pi
      parameter (pi=3.1415926535897932385d0)

      logical nocntevents
      common/cnocntevents/nocntevents

      double precision xnormsv
      common/cxnormsv/xnormsv

c Multi channel stuff:
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2

      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      common/to_mconfigs/mapconfig, iconfig

      integer mapbconf(0:lmaxconfigs)
      integer b_from_r(lmaxconfigs)
      integer r_from_b(lmaxconfigs)
      include "bornfromreal.inc"

      double complex wgt1(2)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip

      double precision ev_enh,enhance,rwgt
      logical firsttime,passcuts
      data firsttime /.true./
      integer inoborn_ev,inoborn_cnt
      double precision xnoborn_ev,xnoborn_cnt

c FKS stuff:
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt

      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision zero,one
      parameter (zero=0d0,one=1d0)

      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision delta_used
      common /cdelta_used/delta_used
      double precision xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      integer diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor

      logical multi_chan(lmaxconfigs)
      common /to_multi_chan/multi_chan

      character*4 abrv
      common /to_abrv/ abrv

      logical nbodyonly
      common/cnbodyonly/nbodyonly

      double precision vegas_weight
      common/cvegas_weight/vegas_weight

c For the MINT folding
      integer fold
      common /cfl/fold

c Random numbers to be used in the plotting routine
      logical needrndec
      parameter (needrndec=.true.)

      real*8 ran2
      external ran2

      real*8 rndec(10)
      common/crndec/rndec

c For tests
      real*8 fksmaxwgt,xisave,ysave
      common/cfksmaxwgt/fksmaxwgt,xisave,ysave
      integer icou_calls,icou_kinev,icou_sev,icou_meev,icou_kincnt,
     #  icou_scnt,icou_mecnt
      common/counters/icou_calls,icou_kinev,icou_sev,icou_meev,
     #                           icou_kincnt,icou_scnt,icou_mecnt
      integer itotalpoints
      common/ctotalpoints/itotalpoints

      logical ExceptPSpoint
      integer iminmax
      common/cExceptPSpoint/iminmax,ExceptPSpoint

      double precision central_wgt_saved
      save central_wgt_saved

      double precision dsig_max,dsig_min
      double precision total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min
      common/csum_of_wgts/total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min

c For process mirroring
      double precision selproc(2)
      double precision sumprob, r, rescale_mir
      integer imirror, k
      common /cmirror/imirror
      include "mirrorprocs.inc"
      data selproc /2*0d0/
      integer totpts, mirpts, nomirpts
      data totpts/ 0 /
      data mirpts/ 0 /
      data nomirpts/ 0 /

      double precision pmass(nexternal)
      include "pmass.inc"

      vegas_weight=vegaswgt

      if (fold.eq.0) then
         calculatedBorn=.false.
         call get_helicity(i_fks,j_fks)
      endif

      if (firsttime)then
         inoborn_ev=0
         xnoborn_ev=0.d0
         inoborn_cnt=0
         xnoborn_cnt=0.d0
         firsttime=.false.

         fksmaxwgt=0.d0
      endif

      prefact=xinorm_ev/xi_i_fks_ev*
     #        1/(1-y_ij_fks_ev)

      if( (.not.nocntevents) .and.
     #    (.not.(abrv.eq.'born' .or. abrv(1:2).eq.'vi')) )then
        prefact_cnt_ssc=xinorm_ev/min(xiimax_ev,xiScut_used)*
     #                  log(xicut_used/min(xiimax_ev,xiScut_used))*
     #                  1/(1-y_ij_fks_ev)
        if(pmass(j_fks).eq.0.d0)then
          prefact_c=xinorm_cnt(ione)/xi_i_fks_cnt(ione)*
     #              1/(1-y_ij_fks_ev)
          prefact_cnt_ssc_c=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                      log(xicut_used/min(xiimax_cnt(ione),xiScut_used))*
     #                      1/(1-y_ij_fks_ev)
          prefact_coll=xinorm_cnt(ione)/xi_i_fks_cnt(ione)*
     #                 log(delta_used/deltaS)/deltaS
          prefact_coll_c=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                   log(xicut_used/min(xiimax_cnt(ione),xiScut_used))*
     #                   log(delta_used/deltaS)/deltaS
          prefact_deg=xinorm_cnt(ione)/xi_i_fks_cnt(ione)*
     #                1/deltaS
          prefact_deg_sxi=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                    log(xicut_used/min(xiimax_cnt(ione),xiScut_used))*
     #                    1/deltaS
          prefact_deg_slxi=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                     ( log(xicut_used)**2 -
     #                       log(min(xiimax_cnt(ione),xiScut_used))**2 )*
     #                     1/(2.d0*deltaS)
        endif
      endif

      if(needrndec)then
        do i=1,10
          rndec(i)=ran2()
        enddo
      endif

c If there was an exceptional phase-space point found for the 
c virtual corrections, at the end of this subroutine, goto 44
c and compute also the "possible" minimal and maximal weight
c these points could have gotton (based upon previous PS
c points)
      ExceptPSpoint=.false.
      iminmax=-1
 44   continue
      iminmax=iminmax+1

      ev_wgt=0.d0
      cnt_wgt=0.d0
      cnt_wgt_s=0.d0
      cnt_wgt_c=0.d0
      cnt_wgt_sc=0.d0
      bsv_wgt=0.d0
      born_wgt=0.d0
      cnt_swgt=0.d0
      cnt_swgt_s=0.d0
      cnt_swgt_sc=0.d0
      deg_wgt=0.d0
      deg_swgt=0.d0
      plot_wgt=0.d0
      iplot=-3


c Compute pdf weight for the process and its mirror configuration
      sumprob=0.d0
      if (abrv.eq.'born'.or.abrv(1:2).eq.'vi') then
        call set_cms_stuff(izero)
      else
        call set_cms_stuff(mohdr)
      endif
      do k=1,2
        if (k.eq.1.or.mirrorproc) then
          imirror=k
          selproc(k)=dlum()
          sumprob=sumprob+selproc(k)
        endif
      enddo

c Choose between process and mirror
      r=ran2()*sumprob
      imirror=1
      if (r.gt.selproc(1)) imirror=2

c If mirror process rotate (all) the momenta around the x-axis
      totpts = totpts+1
      if (imirror.eq.2) then
        mirpts = mirpts+1
        p_i_fks_ev(2)= - p_i_fks_ev(2)
        p_i_fks_ev(3)= - p_i_fks_ev(3)
        do i=-2,2
          p_i_fks_cnt(2,i)= - p_i_fks_cnt(2,i)
          p_i_fks_cnt(3,i)= - p_i_fks_cnt(3,i)
        enddo
        do k=1,nexternal
          pp(2,k)= - pp(2,k)
          pp(3,k)= - pp(3,k)
          do i=-2,2
            p1_cnt(2,k,i)= - p1_cnt(2,k,i)
            p1_cnt(3,k,i)= - p1_cnt(3,k,i)
          enddo
        enddo
        do k=1,nexternal-1
          p_born(2,k)= - p_born(2,k)
          p_born(3,k)= - p_born(3,k)
        enddo
      else
        nomirpts = nomirpts+1
      endif

      if (mod(totpts, 1000).eq.0) write(*,*) mirpts, nomirpts, totpts

      if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') goto 540
c Real contribution:
c Set the ybst_til_tolab before applying the cuts. 
      call set_cms_stuff(mohdr)
      call get_mirror_rescale(rescale_mir)
      if (passcuts(pp,rwgt)) then
        call set_alphaS(pp)
        x = abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
        ffact = f_damp(x)
        s_ev = fks_Sij(pp,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
        if(s_ev.gt.0.d0)then
           call sreal(pp,xi_i_fks_ev,y_ij_fks_ev,fx_ev)
           xlum_ev = dlum()*rescale_mir
           ev_wgt = fx_ev*xlum_ev*s_ev*ffact*wgt*prefact*rwgt
        endif
      endif
c
c All counterevent have the same final-state kinematics. Check that
c one of them passes the hard cuts, and they exist at all
 540  continue

c Set the ybst_til_tolab before applying the cuts. Update below
c for the collinear, soft and/or soft-collinear subtraction terms
      call set_cms_stuff(izero)
      if ( (.not.passcuts(p1_cnt(0,1,0),rwgt)) .or.
     #     nocntevents ) goto 547
      call set_alphaS(p1_cnt(0,1,0))
c If mirroring and dlum has not been called, we need to flip back the
c    beams

      if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') goto 545
c
c Collinear subtraction term:
      if( y_ij_fks_ev.gt.1d0-deltaS .and.
     #    pmass(j_fks).eq.0.d0 )then
         call set_cms_stuff(ione)
         call get_mirror_rescale(rescale_mir)
         s_c = fks_Sij(p1_cnt(0,1,1),i_fks,j_fks,xi_i_fks_cnt(ione),one)
         if(s_c.gt.0.d0)then
            if(abs(s_c-1.d0).gt.1.d-6.and.j_fks.le.nincoming)then
              write(*,*)'Wrong S function in dsig[c]',s_c
              stop
            endif
            call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(ione),one,fx_c)
            xlum_c = dlum()*rescale_mir
            cnt_wgt_c=cnt_wgt_c-
     &           fx_c*xlum_c*s_c*jac_cnt(1)*(prefact_c+prefact_coll)*rwgt
            call sreal_deg(p1_cnt(0,1,1),xi_i_fks_cnt(ione),one,
     #                     deg_xi_c,deg_lxi_c)
            deg_wgt=deg_wgt+( deg_xi_c+deg_lxi_c*log(xi_i_fks_cnt(ione)) )*
     #                      jac_cnt(1)*prefact_deg*rwgt/(shat/(32*pi**2))*
     #                      xlum_c
            iplot=1
         endif
      endif
c Soft subtraction term:
 545  continue
      if (xi_i_fks_ev .lt. max(xiScut_used,xiBSVcut_used)) then
         call set_cms_stuff(izero)
         call get_mirror_rescale(rescale_mir)
         s_s = fks_Sij(p1_cnt(0,1,0),i_fks,j_fks,zero,y_ij_fks_ev)
         if(nbodyonly)s_s=1.d0
         if(s_s.gt.0.d0)then
            xlum_s = dlum()*rescale_mir
            if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') goto 546
            if (xi_i_fks_ev .lt. xiScut_used) then
              call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fx_s)
              cnt_s=fx_s*xlum_s*s_s*jac_cnt(0)
              cnt_wgt_s=cnt_wgt_s-cnt_s*prefact*rwgt
              cnt_swgt_s=cnt_swgt_s-cnt_s*prefact_cnt_ssc*rwgt
            endif
 546        continue
            if (abrv.eq.'real') goto 548
            if (xi_i_fks_ev .lt. xiBSVcut_used) then
              xnormsv=s_s*jac_cnt(0)*xinorm_ev/
     #                (min(xiimax_ev,xiBSVcut_used)*shat/(16*pi**2))*
     #                rwgt*xlum_s
              call bornsoftvirtual(p1_cnt(0,1,0),bsv_wgt,born_wgt)
              bsv_wgt=bsv_wgt*xnormsv
              born_wgt=born_wgt*xnormsv
            endif
 548        continue
            iplot=0
         endif
      endif
c Soft-Collinear subtraction term:
      if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') goto 547
      if (xi_i_fks_cnt(ione) .lt. xiScut_used .and.
     #    y_ij_fks_ev .gt. 1d0-deltaS .and.
     #    pmass(j_fks).eq.0.d0 )then
         call set_cms_stuff(itwo)
         call get_mirror_rescale(rescale_mir)
         s_sc = fks_Sij(p1_cnt(0,1,2),i_fks,j_fks,zero,one)
         if(s_sc.gt.0.d0)then
            if(abs(s_sc-1.d0).gt.1.d-6.and.j_fks.le.nincoming)then
              write(*,*)'Wrong S function in dsig[sc]',s_sc
              stop
            endif
            xlum_sc = dlum()*rescale_mir
            call sreal(p1_cnt(0,1,2),zero,one,fx_sc)
            cnt_sc=fx_sc*xlum_sc*s_sc*jac_cnt(2)
            cnt_wgt_sc=cnt_wgt_sc+cnt_sc*(prefact_c+prefact_coll)*rwgt
            cnt_swgt_sc=cnt_swgt_sc+
     &           cnt_sc*(prefact_cnt_ssc_c+prefact_coll_c)*rwgt
            call sreal_deg(p1_cnt(0,1,2),zero,one,
     #                     deg_xi_sc,deg_lxi_sc)
            deg_wgt=deg_wgt-( deg_xi_sc+deg_lxi_sc*log(xi_i_fks_cnt(ione)) )*
     #                     jac_cnt(2)*prefact_deg*rwgt/(shat/(32*pi**2))*
     #                     xlum_sc
            deg_swgt=deg_swgt-( deg_xi_sc*prefact_deg_sxi +
     #                     deg_lxi_sc*prefact_deg_slxi )*
     #                     jac_cnt(2)*rwgt/(shat/(32*pi**2))*
     #                     xlum_sc
            if(iplot.ne.0)iplot=2
         endif
      endif
 547  continue

c
c Enhance the one channel for multi-channel integration
c
      enhance=1.d0
      if ((ev_wgt.ne.0d0.or.cnt_wgt_c.ne.0d0.or.cnt_wgt_s.ne.0d0.or.
     &     cnt_wgt_sc.ne.0d0.or.bsv_wgt.ne.0d0.or.deg_wgt.ne.0d0.or.
     &     deg_swgt.ne.0d0.or.cnt_swgt_s.ne.0d0.or.cnt_swgt_sc.ne.0d0)
     &     .and. multi_channel) then
         if (bsv_wgt.eq.0d0.and.deg_wgt.eq.0d0.and.deg_swgt.eq.0d0.and.
     &       cnt_wgt_c.eq.0d0 ) CalculatedBorn=.false.

         if (.not.calculatedBorn .and. p_born(0,1).gt.0d0)then
            call sborn(p_born,wgt1)
         elseif(p_born(0,1).lt.0d0)then
            enhance=0d0
         endif

         if (enhance.eq.0d0)then
            xnoborn_cnt=xnoborn_cnt+1.d0
            if(log10(xnoborn_cnt).gt.inoborn_cnt)then
               write (*,*) 
     #           'Function dsig: no Born momenta more than 10**',
     #           inoborn_cnt,'times'
               inoborn_cnt=inoborn_cnt+1
            endif
         else
            xtot=0d0
            if (mapbconf(0).eq.0) then
               write (*,*) 'Fatal error in dsig, no Born diagrams '
     &           ,mapbconf,'. Check bornfromreal.inc'
               write (*,*) 'Is fks_singular compiled correctly?'
               stop
            endif
            if (onlyBorn) then
               do i=1, mapbconf(0)
                  if (multi_chan(mapbconf(i))) then
                     xtot=xtot+amp2(mapbconf(i))
                  endif
               enddo
            else
               do i=1,mapbconf(0)
                  xtot=xtot+amp2(mapbconf(i))
               enddo
            endif
            if (xtot.ne.0d0) then
               enhance=amp2(b_from_r(mapconfig(iconfig)))/xtot
               enhance=enhance*diagramsymmetryfactor
            else
               enhance=0d0
            endif
         endif
      endif

      cnt_wgt = cnt_wgt_c + cnt_wgt_s + cnt_wgt_sc
      cnt_swgt = cnt_swgt_s + cnt_swgt_sc

      ev_wgt = ev_wgt * enhance
      cnt_wgt = cnt_wgt * enhance
      cnt_swgt = cnt_swgt * enhance
      bsv_wgt = bsv_wgt * enhance
      born_wgt = born_wgt * enhance
      deg_wgt = deg_wgt * enhance
      deg_swgt = deg_swgt * enhance

      
      if(iminmax.eq.0) then
         dsig = (ev_wgt+cnt_wgt)*fkssymmetryfactor +
     &        cnt_swgt*fkssymmetryfactor +
     &        bsv_wgt*fkssymmetryfactorBorn +
     &        deg_wgt*fkssymmetryfactorDeg +
     &        deg_swgt*fkssymmetryfactorDeg

         if (dsig.ne.dsig) then
            write (*,*) 'ERROR #51 in dsig:',dsig,'skipping event'
            dsig=0d0
            return
         endif

         total_wgt_sum=total_wgt_sum+dsig*vegaswgt
         central_wgt_saved=dsig

c For tests
         if(abs(dsig).gt.fksmaxwgt)then
            fksmaxwgt=abs(dsig)
            xisave=xi_i_fks_ev
            ysave=y_ij_fks_ev
         endif
         if (dsig.ne.0d0) itotalpoints=itotalpoints+1
         icou_calls=icou_calls+1
         if(pp(0,1).gt.0.d0)icou_kinev=icou_kinev+1
         if(s_ev.gt.0.d0)icou_sev=icou_sev+1
         if(s_ev.gt.0.d0.and.abs(ev_wgt).gt.0.d0)icou_meev=icou_meev+1
         if( p1_cnt(0,1,0).gt.0.d0 .or.
     &        p1_cnt(0,1,1).gt.0.d0 .or.
     &        p1_cnt(0,1,2).gt.0.d0 )icou_kincnt=icou_kincnt+1
         if( s_s.gt.0.d0.or.s_c.gt.0.d0.or.
     &        s_sc.gt.0.d0 )icou_scnt=icou_scnt+1
         if( (s_s.gt.0.d0.or.s_c.gt.0.d0.or.
     &        s_sc.gt.0.d0).and.abs(cnt_wgt).gt.0.d0)
     &        icou_mecnt=icou_mecnt+1
         
c Plot observables for event

         plot_wgt=ev_wgt*fkssymmetryfactor*vegaswgt
         if(abs(plot_wgt).gt.1.d-20.and.pp(0,1).ne.-99d0)
     &        call outfun(pp,ybst_til_tolab,plot_wgt,iplot_ev)
c Plot observables for counterevents and Born
         plot_wgt=( cnt_wgt*fkssymmetryfactor +
     &              cnt_swgt*fkssymmetryfactor +
     &              bsv_wgt*fkssymmetryfactorBorn +
     &              deg_wgt*fkssymmetryfactorDeg +
     &              deg_swgt*fkssymmetryfactorDeg )*vegaswgt
         if(abs(plot_wgt).gt.1.d-20.and.p1_cnt(0,1,iplot).ne.-99d0)then
            if(iplot.eq.-3)then
               write(*,*)'Error #1 in dsig'
               stop
            endif
            call outfun(p1_cnt(0,1,iplot),ybst_til_tolab,plot_wgt,
     &                  iplot_cnt)
         endif
c Plot observables for Born; pass cnt momenta assuming they are
c identical to Born ones
         plot_wgt=born_wgt*fkssymmetryfactorBorn*vegaswgt
         if(abs(plot_wgt).gt.1.d-20.and.p1_cnt(0,1,iplot).ne.-99d0)
     &        call outfun(p1_cnt(0,1,iplot),ybst_til_tolab,plot_wgt,
     &                    iplot_born)

      elseif (iminmax.eq.1 .and. ExceptPSpoint) then
c for except PS points, this is the maximal approx for the virtual         
         dsig_max = (ev_wgt+cnt_wgt)*fkssymmetryfactor +
     &        cnt_swgt*fkssymmetryfactor +
     &        bsv_wgt*fkssymmetryfactorBorn +
     &        deg_wgt*fkssymmetryfactorDeg +
     &        deg_swgt*fkssymmetryfactorDeg
         total_wgt_sum_max=total_wgt_sum_max+
     &        ((dsig_max - central_wgt_saved)*vegaswgt)**2

      elseif (iminmax.eq.2 .and. ExceptPSpoint) then
c for except PS points, this is the minimal approx for the virtual         
         dsig_min = (ev_wgt+cnt_wgt)*fkssymmetryfactor +
     &        cnt_swgt*fkssymmetryfactor +
     &        bsv_wgt*fkssymmetryfactorBorn +
     &        deg_wgt*fkssymmetryfactorDeg +
     &        deg_swgt*fkssymmetryfactorDeg
         total_wgt_sum_min=total_wgt_sum_min+
     &        ((central_wgt_saved - dsig_min)*vegaswgt)**2
      else
         write (*,*) 'Error #12 in dsig',iminmax
         stop
      endif

c If exceptional PS point found, go back to beginning recompute
c the weight for this PS point using an approximation
c based on previous PS points (done in LesHouches.f)
      if (ExceptPSpoint .and. iminmax.le.1) goto 44

      return
      end


      double precision function dsigH(pp,wgt,vegaswgt)
c Here are the subtraction terms, the Sij function, 
c the f-damping function, and the single diagram
c enhanced multi-channel factor included
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "fks.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      include 'coupl.inc'

      double precision pp(0:3,nexternal),wgt,vegaswgt

      double precision fks_Sij,fks_Hij,f_damp,dot,dlum
      external fks_Sij,fks_Hij,f_damp,dot,dlum

      double precision prefact,ev_wgt,xmc_wgt,plot_wgt,x,
     # ffact,s_ev,fx_ev,gfactsf,gfactcl,xmcMC,xmcME,xmc,totH_wgt,
     # xtot,xlum_ev,xlum_mc,xlum_mc_save,dummy,fx_c,fx_s,
     # fx_sc,s_c,s_s,s_sc,xlum_c,xlum_s,xlum_sc,prefact_c
      double precision probne,sevmc,get_ptrel
      integer i

      integer mohdr,iplot_ev
      parameter (mohdr=-100)
      parameter (iplot_ev=11)

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      integer izero,ione,itwo
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)

c Multi channel stuff:
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2

      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      common/to_mconfigs/mapconfig, iconfig

      integer mapbconf(0:lmaxconfigs)
      integer b_from_r(lmaxconfigs)
      integer r_from_b(lmaxconfigs)
      include "bornfromreal.inc"

      double complex wgt1(2)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip

      double precision enhance,rwgt
      logical firsttime,passcuts
      data firsttime /.true./
      integer inoborn_ev,inoborn_cnt
      double precision xnoborn_ev,xnoborn_cnt

c FKS stuff:
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt

      double precision zero,one
      parameter (zero=0d0,one=1d0)

      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      integer diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor

      logical multi_chan(lmaxconfigs)
      common /to_multi_chan/multi_chan

      character*4 abrv
      common /to_abrv/ abrv

      double precision zhw_used
c MC stuff
      double precision zhw(nexternal),xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc

c Stuff to be written (depending on AddInfoLHE) onto the LHE file
      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe

c CKWW scale, from cuts.f
      double precision scale_CKKW
      common/cscale_CKKW/scale_CKKW

c For the MINT folding
      integer fold
      common /cfl/fold

      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt

c For plots
      logical plotEv,plotKin
      common/cEvKinplot/plotEv,plotKin

c For tests
      real*8 fksmaxwgt,xisave,ysave
      common/cfksmaxwgt/fksmaxwgt,xisave,ysave

      double precision pmass(nexternal)
      include "pmass.inc"

      if (fold.eq.0) then
         call get_helicity(i_fks,j_fks)
         calculatedBorn=.false.
      endif

      if (firsttime)then
         inoborn_ev=0
         xnoborn_ev=0.d0
         inoborn_cnt=0
         xnoborn_cnt=0.d0
         fksmaxwgt=0.d0
         firsttime=.false.
      endif

      prefact=xinorm_ev/xi_i_fks_ev*
     #        1/(1-y_ij_fks_ev)
      if(pmass(j_fks).eq.0.d0)then
        prefact_c=xinorm_cnt(ione)/xi_i_fks_cnt(ione)*
     #            1/(1-y_ij_fks_ev)
      endif

      ev_wgt=0.d0
      xmc_wgt=0.d0
      plot_wgt=0.d0

      if(AddInfoLHE)then
        iSorH_lhe=2
        ifks_lhe=i_fks
        jfks_lhe=j_fks
        fksfather_lhe=0
        ipartner_lhe=0
        scale1_lhe=0.d0
        scale2_lhe=0.d0
      endif

      if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') then
         write (*,*) 'No need to generate Hevents when doing: ',abrv
         stop
      endif

      probne=1.d0
c MC counterterms. Cuts should be with the Born-type momenta. Rest is
c with the event momenta.
      call set_cms_stuff(izero)
      if (passcuts(p1_cnt(0,1,0),rwgt)) then
        gfactsf=1.d0
        gfactcl=1.d0
        sevmc=1.d0
        xmcMC=0.d0
        xmcME=0.d0
        call set_cms_stuff(mohdr)
        call set_alphaS(pp)
        if(UseSfun)then
           x = abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
           ffact = f_damp(x)
           sevmc = fks_Sij(pp,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
           sevmc = sevmc*ffact
        else
           x = abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
           ffact = f_damp(x)
           sevmc = fks_Hij(pp,i_fks,j_fks)
           sevmc = sevmc*ffact
        endif
        call xmcsubt(pp,xi_i_fks_ev,y_ij_fks_ev,gfactsf,gfactcl,probne,
     #               dummy,nofpartners,lzone,flagmc,zhw,xmcxsec)
        if(sevmc.gt.0.d0.and.flagmc)then
          xlum_mc_save=-1.d8
          do i=1,nofpartners
            if(lzone(i))then
              zhw_used=zhw(i)
              call get_mc_lum(j_fks,zhw_used,xi_i_fks_ev,
     #                        xlum_mc_save,xlum_mc)
              xmcMC=xmcMC+xmcxsec(i)*xlum_mc
            endif
          enddo
          xmcMC=-xmcMC*sevmc*wgt*prefact*rwgt
        endif
c
        if( (.not.flagmc).and.gfactsf.eq.1.d0 .and.
     #      xi_i_fks_ev.lt.0.02d0 .and. particle_type(i_fks).eq.8)then
          write(*,*)'Error in dsigH: will diverge'
          stop
        endif
c
        if(gfactsf.lt.1.d0.and.probne.gt.0.d0)then
          call set_cms_stuff(izero)
          call set_alphaS(p1_cnt(0,1,0))
          s_s = fks_Sij(p1_cnt(0,1,0),i_fks,j_fks,zero,y_ij_fks_ev)
          if(s_s.gt.0.d0)then
            call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fx_s)
            xlum_s = dlum()
            xmcME=fx_s*xlum_s*s_s*jac_cnt(0)*prefact*rwgt
          endif
          if(gfactcl.lt.1.d0.and.pmass(j_fks).eq.0.d0)then
            call set_cms_stuff(ione)
            s_c = fks_Sij(p1_cnt(0,1,1),i_fks,j_fks,xi_i_fks_cnt(ione),one)
            if(s_c.gt.0.d0)then
              if(abs(s_c-1.d0).gt.1.d-6.and.j_fks.le.nincoming)then
                write(*,*)'Wrong S function in dsigH[c]',s_c
                stop
              endif
              call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(ione),one,fx_c)
              xlum_c = dlum()
              xmcME=xmcME+
     #          fx_c*xlum_c*s_c*jac_cnt(1)*prefact_c*rwgt*(1-gfactcl)
c$$$     #           fx_c*xlum_c*s_c*jac_cnt(1)*(prefact_c+prefact_coll)*rwgt
            endif
            call set_cms_stuff(itwo)
            s_sc = fks_Sij(p1_cnt(0,1,2),i_fks,j_fks,zero,one)
            if(s_sc.gt.0.d0)then
              if(abs(s_sc-1.d0).gt.1.d-6.and.j_fks.le.nincoming)then
                write(*,*)'Wrong S function in dsigH[sc]',s_sc
                stop
              endif
              xlum_sc = dlum()
              call sreal(p1_cnt(0,1,2),zero,one,fx_sc)
              xmcME=xmcME-
     #          fx_sc*xlum_sc*s_sc*jac_cnt(2)*prefact_c*rwgt*(1-gfactcl)
c$$$     #        fx_sc*xlum_sc*s_sc*jac_cnt(1)*(prefact_c+prefact_coll)*rwgt
            endif
          endif
          xmcME=-xmcME*(1-gfactsf)*probne
        endif
c
        xmc_wgt=xmcMC+xmcME
      endif

c Real contribution
c
c Set the ybst_til_tolab before applying the cuts. 
      call set_cms_stuff(mohdr)
      if (passcuts(pp,rwgt).and.probne.gt.0.d0) then
        call set_alphaS(pp)
        x = abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
        ffact = f_damp(x)
        s_ev = fks_Sij(pp,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
        if(s_ev.gt.0.d0)then
          call sreal(pp,xi_i_fks_ev,y_ij_fks_ev,fx_ev)
          xlum_ev = dlum()
          ev_wgt = fx_ev*xlum_ev*s_ev*ffact*wgt*prefact*rwgt*probne
        endif
        if(AddInfoLHE)scale2_lhe=get_ptrel(pp,i_fks,j_fks)
      endif

      if(AddInfoLHE.and.UseCKKW)then
        if(scale1_lhe.eq.0.d0)scale1_lhe=scale2_lhe
        scale2_lhe=scale_CKKW
      endif

      totH_wgt = ev_wgt+xmc_wgt
c
c Enhance the one channel for multi-channel integration
c
      enhance=1.d0
      if (totH_wgt.ne.0d0 .and. multi_channel) then
         if (xmc.eq.0d0) CalculatedBorn=.false.
         if (.not.calculatedBorn .and. p_born(0,1).gt.0d0)then
            call sborn(p_born,wgt1)
         elseif(p_born(0,1).lt.0d0)then
            enhance=0d0
         endif

         if (enhance.eq.0d0)then
            xnoborn_cnt=xnoborn_cnt+1.d0
            if(log10(xnoborn_cnt).gt.inoborn_cnt)then
               write (*,*) 
     #           'Function dsigH: no Born momenta more than 10**',
     #           inoborn_cnt,'times'
               inoborn_cnt=inoborn_cnt+1
            endif
         else
            xtot=0d0
            if (mapbconf(0).eq.0) then
               write (*,*) 'Fatal error in dsigH, no Born diagrams '
     &           ,mapbconf,'. Check bornfromreal.inc'
               write (*,*) 'Is fks_singular compiled correctly?'
               stop
            endif
            if (onlyBorn) then
               do i=1, mapbconf(0)
                  if (multi_chan(mapbconf(i))) then
                     xtot=xtot+amp2(mapbconf(i))
                  endif
               enddo
            else
               do i=1,mapbconf(0)
                  xtot=xtot+amp2(mapbconf(i))
               enddo
            endif
            if (xtot.ne.0d0) then
               enhance=amp2(b_from_r(mapconfig(iconfig)))/xtot
               enhance=enhance*diagramsymmetryfactor
            else
               enhance=0d0
            endif
         endif
      endif

      totH_wgt = totH_wgt * enhance

      dsigH = totH_wgt*fkssymmetryfactor

      if(dsigH.ne.0.d0)
     #  call set_shower_scale(pp,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)

c For tests
      if(abs(dsigH).gt.fksmaxwgt)then
        fksmaxwgt=abs(dsigH)
        xisave=xi_i_fks_ev
        ysave=y_ij_fks_ev
      endif

c Plot observables for event
      if (.not.unwgt) then
         plot_wgt=totH_wgt*fkssymmetryfactor*vegaswgt 
         if( abs(plot_wgt).gt.1.d-20.and.pp(0,1).ne.-99d0. and.
     &       (plotEv.or.plotKin) )
     &      call outfun(pp,ybst_til_tolab,plot_wgt,iplot_ev)
      endif
      return
      end


      double precision function dsigS(pp,wgt,vegaswgt)
c Here are the subtraction terms, the Sij function, 
c the f-damping function, and the single diagram
c enhanced multi-channel factor included
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "fks.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      include 'coupl.inc'

      double precision pp(0:3,nexternal),wgt,vegaswgt

      double precision fks_Sij,fks_Hij,f_damp,dot,dlum
      external fks_Sij,fks_Hij,f_damp,dot,dlum

      double precision x,xtot,s_ev,s_c,s_s,s_sc,ffact,fx_c,
     #                 fx_s,fx_sc,xmc_wgt,cnt_wgt_c,cnt_wgt_s,cnt_wgt_sc,
     #                 bsv_wgt,plot_wgt,cnt_swgt_s,cnt_swgt_sc,cnt_sc,cnt_s,
     #                 prefact_cnt_ssc,prefact_cnt_ssc_c,prefact_coll,
     #                 prefact_coll_c,born_wgt,prefact_deg,prefact,prefact_c,
     #                 prefact_deg_sxi,prefact_deg_slxi,deg_wgt,deg_swgt,
     #                 deg_xi_c,deg_lxi_c,deg_xi_sc,deg_lxi_sc,
     #                 cnt_swgt,cnt_wgt,gfactsf,gfactcl,xmcMC,xmcME,
     #                 xlum_c,xlum_s,xlum_sc,xlum_mc,xlum_mc_save,
     #                 dummy,ev_wgt,fx_ev,probne,sevmc,xlum_ev,get_ptrel
      integer i,j

      integer izero,ione,itwo,mohdr,iplot_ev,iplot_cnt,iplot_born
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)
      parameter (mohdr=-100)
      parameter (iplot_ev=11)
      parameter (iplot_cnt=12)
      parameter (iplot_born=20)

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision shattmp
      double precision pi
      parameter (pi=3.1415926535897932385d0)

      logical nocntevents
      common/cnocntevents/nocntevents

c Multi channel stuff:
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2

      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      common/to_mconfigs/mapconfig, iconfig

      integer mapbconf(0:lmaxconfigs)
      integer b_from_r(lmaxconfigs)
      integer r_from_b(lmaxconfigs)
      include "bornfromreal.inc"

      double complex wgt1(2)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip

      double precision ev_enh,enhance,rwgt
      logical firsttime,passcuts
      data firsttime /.true./
      integer inoborn_ev,inoborn_cnt
      double precision xnoborn_ev,xnoborn_cnt

c FKS stuff:
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt

      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision zero,one
      parameter (zero=0d0,one=1d0)

      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision delta_used
      common /cdelta_used/delta_used
      double precision xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      integer diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor

      logical multi_chan(lmaxconfigs)
      common /to_multi_chan/multi_chan

      logical nbodyonly
      common/cnbodyonly/nbodyonly

      character*4 abrv
      common /to_abrv/ abrv

      double precision zhw_used
c MC stuff
      double precision zhw(nexternal),xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc

c Stuff to be written (depending on AddInfoLHE) onto the LHE file
      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe

c CKWW scale, from cuts.f
      double precision scale_CKKW
      common/cscale_CKKW/scale_CKKW

c For the MINT folding
      integer fold
      common /cfl/fold

      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt

c For plots
      logical plotEv,plotKin
      common/cEvKinplot/plotEv,plotKin

c For tests
      real*8 fksmaxwgt,xisave,ysave
      common/cfksmaxwgt/fksmaxwgt,xisave,ysave
      integer icou_calls,icou_kinev,icou_sev,icou_meev,icou_kincnt,
     #  icou_scnt,icou_mecnt
      common/counters/icou_calls,icou_kinev,icou_sev,icou_meev,
     #                           icou_kincnt,icou_scnt,icou_mecnt
      logical ExceptPSpoint
      integer iminmax
      common/cExceptPSpoint/iminmax,ExceptPSpoint

      double precision pmass(nexternal)
      include "pmass.inc"

      if (fold.eq.0) then
         calculatedBorn=.false.
         call get_helicity(i_fks,j_fks)
      endif

      if (firsttime)then
         inoborn_ev=0
         xnoborn_ev=0.d0
         inoborn_cnt=0
         xnoborn_cnt=0.d0
         firsttime=.false.

         fksmaxwgt=0.d0
      endif

      prefact=xinorm_ev/xi_i_fks_ev*
     #        1/(1-y_ij_fks_ev)

      if( (.not.nocntevents) .and.
     #    (.not.(abrv.eq.'born' .or. abrv(1:2).eq.'vi')) )then
        prefact_cnt_ssc=xinorm_ev/min(xiimax_ev,xiScut_used)*
     #                  log(xicut_used/min(xiimax_ev,xiScut_used))*
     #                  1/(1-y_ij_fks_ev)
        if(pmass(j_fks).eq.0.d0)then
          prefact_c=xinorm_cnt(ione)/xi_i_fks_cnt(ione)*
     #              1/(1-y_ij_fks_ev)
          prefact_cnt_ssc_c=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                      log(xicut_used/min(xiimax_cnt(ione),xiScut_used))*
     #                      1/(1-y_ij_fks_ev)
          prefact_coll=xinorm_cnt(ione)/xi_i_fks_cnt(ione)*
     #                 log(delta_used/deltaS)/deltaS
          prefact_coll_c=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                   log(xicut_used/min(xiimax_cnt(ione),xiScut_used))*
     #                   log(delta_used/deltaS)/deltaS
          prefact_deg=xinorm_cnt(ione)/xi_i_fks_cnt(ione)*
     #                1/deltaS
          prefact_deg_sxi=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                    log(xicut_used/min(xiimax_cnt(ione),xiScut_used))*
     #                    1/deltaS
          prefact_deg_slxi=xinorm_cnt(ione)/min(xiimax_cnt(ione),xiScut_used)*
     #                     ( log(xicut_used)**2 -
     #                       log(min(xiimax_cnt(ione),xiScut_used))**2 )*
     #                     1/(2.d0*deltaS)
        endif
      endif

      iminmax=0
      ev_wgt=0.d0
      xmc_wgt=0.d0
      cnt_wgt=0.d0
      cnt_wgt_s=0.d0
      cnt_wgt_c=0.d0
      cnt_wgt_sc=0.d0
      bsv_wgt=0.d0
      born_wgt=0.d0
      cnt_swgt=0.d0
      cnt_swgt_s=0.d0
      cnt_swgt_sc=0.d0
      deg_wgt=0.d0
      deg_swgt=0.d0
      plot_wgt=0.d0
c
      if(AddInfoLHE)then
        iSorH_lhe=1
        ifks_lhe=i_fks
        jfks_lhe=j_fks
        fksfather_lhe=0
        ipartner_lhe=0
        scale1_lhe=0.d0
        scale2_lhe=0.d0
      endif
c
      probne=1.d0
c
c All counterevent have the same final-state kinematics. Check that
c one of them passes the hard cuts, and they exist at all
c
c Set the ybst_til_tolab before applying the cuts. Update below
c for the collinear, soft and/or soft-collinear subtraction terms
      call set_cms_stuff(izero)
      if ( (.not.passcuts(p1_cnt(0,1,0),rwgt)) .or.
     #      nocntevents ) goto 547

      gfactsf=1.d0
      gfactcl=1.d0
      sevmc=1.d0
      xmcMC=0.d0
      xmcME=0.d0

      if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') goto 545

      call set_cms_stuff(mohdr)
      call set_alphaS(pp)
      if(UseSfun)then
         x = abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
         ffact = f_damp(x)
         sevmc = fks_Sij(pp,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
         sevmc = sevmc*ffact
      else
         x = abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
         ffact = f_damp(x)
         sevmc = fks_Hij(pp,i_fks,j_fks)
         sevmc = sevmc*ffact
      endif
      call xmcsubt(pp,xi_i_fks_ev,y_ij_fks_ev,gfactsf,gfactcl,probne,
     #             dummy,nofpartners,lzone,flagmc,zhw,xmcxsec)
      if(sevmc.gt.0.d0.and.flagmc)then
        xlum_mc_save=-1.d8
        do i=1,nofpartners
          if(lzone(i))then
            zhw_used=zhw(i)
            call get_mc_lum(j_fks,zhw_used,xi_i_fks_ev,
     #                      xlum_mc_save,xlum_mc)
            xmcMC=xmcMC+xmcxsec(i)*xlum_mc
          endif
        enddo
        xmcMC=xmcMC*sevmc*wgt*prefact*rwgt
      endif
c
      if( (.not.flagmc).and.gfactsf.eq.1.d0 .and.
     #   xi_i_fks_ev.lt.0.02d0  .and. particle_type(i_fks).eq.8)then
        write(*,*)'Error in dsigS: will diverge'
        stop
      endif
c
c Collinear subtraction term:
      if( ( y_ij_fks_ev.gt.1d0-deltaS .or. 
     #     (gfactsf.lt.1.d0.and.gfactcl.lt.1.d0 .and.
     #      probne.gt.0.d0) ) .and.
     #    pmass(j_fks).eq.0.d0 )then
         call set_cms_stuff(ione)
         call set_alphaS(p1_cnt(0,1,1))
         s_c = fks_Sij(p1_cnt(0,1,1),i_fks,j_fks,xi_i_fks_cnt(ione),one)
         if(s_c.gt.0.d0)then
            if(abs(s_c-1.d0).gt.1.d-6.and.j_fks.le.nincoming)then
               write(*,*)'Wrong S function in dsigS[c]',s_c
               stop
            endif
            call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(ione),one,fx_c)
            xlum_c = dlum()
            xmcME=xmcME+
     #         fx_c*xlum_c*s_c*jac_cnt(1)*prefact_c*rwgt*(1-gfactcl)
            if( y_ij_fks_ev.gt.1d0-deltaS )then
               cnt_wgt_c=cnt_wgt_c-
     &          fx_c*xlum_c*s_c*jac_cnt(1)*(prefact_c+prefact_coll)*rwgt
               call sreal_deg(p1_cnt(0,1,1),xi_i_fks_cnt(ione),one,
     #                        deg_xi_c,deg_lxi_c)
               deg_wgt=deg_wgt+( deg_xi_c+deg_lxi_c*log(xi_i_fks_cnt(ione)) )*
     #                         jac_cnt(1)*prefact_deg*rwgt/(shat/(32*pi**2))*
     #                         xlum_c
            endif
         endif
      endif
c Soft subtraction term:
 545  continue
      if ( xi_i_fks_ev .lt. max(xiScut_used,xiBSVcut_used) .or.
     &     (gfactsf.lt.1.d0.and.probne.gt.0.d0) ) then
         call set_cms_stuff(izero)
         call set_alphaS(p1_cnt(0,1,0))
         s_s = fks_Sij(p1_cnt(0,1,0),i_fks,j_fks,zero,y_ij_fks_ev)
         if(nbodyonly)s_s=1.d0
         if(s_s.gt.0.d0)then
            xlum_s = dlum()
            if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') goto 546
            call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fx_s)
            xmcME=xmcME+fx_s*xlum_s*s_s*jac_cnt(0)*prefact*rwgt
            if (xi_i_fks_ev .lt. xiScut_used) then
              cnt_s=fx_s*xlum_s*s_s*jac_cnt(0)
              cnt_wgt_s=cnt_wgt_s-cnt_s*prefact*rwgt
              cnt_swgt_s=cnt_swgt_s-cnt_s*prefact_cnt_ssc*rwgt
            endif
 546        continue
            if (xi_i_fks_ev .lt. xiBSVcut_used) then
              call bornsoftvirtual(p1_cnt(0,1,0),bsv_wgt,born_wgt)
              bsv_wgt=bsv_wgt*s_s*jac_cnt(0)*xinorm_ev/
     #                (min(xiimax_ev,xiBSVcut_used)*shat/(16*pi**2))*
     #                rwgt*xlum_s
              born_wgt=born_wgt*s_s*jac_cnt(0)*xinorm_ev/
     #                (min(xiimax_ev,xiBSVcut_used)*shat/(16*pi**2))*
     #                rwgt*xlum_s
            endif
 548        continue
         endif
      endif
c Soft-Collinear subtraction term:
      if (abrv.eq.'born' .or. abrv(1:2).eq.'vi') goto 547
      if ( ( (xi_i_fks_cnt(ione) .lt. xiScut_used .and.
     #        y_ij_fks_ev .gt. 1d0-deltaS) .or.
     #        (gfactsf.lt.1.d0.and.gfactcl.lt.1.d0 .and.
     #         probne.gt.0.d0) ) .and.
     #        pmass(j_fks).eq.0.d0 )then
         call set_cms_stuff(itwo)
         call set_alphaS(p1_cnt(0,1,2))
         s_sc = fks_Sij(p1_cnt(0,1,2),i_fks,j_fks,zero,one)
         if(s_sc.gt.0.d0)then
            if(abs(s_sc-1.d0).gt.1.d-6.and.j_fks.le.nincoming)then
              write(*,*)'Wrong S function in dsigS[sc]',s_sc
              stop
            endif
            call sreal(p1_cnt(0,1,2),zero,one,fx_sc)
            xlum_sc = dlum()
            xmcME=xmcME-
     #        fx_sc*xlum_sc*s_sc*jac_cnt(2)*prefact_c*rwgt*(1-gfactcl)
            if(xi_i_fks_cnt(ione) .lt. xiScut_used .and.
     #          y_ij_fks_ev .gt. 1d0-deltaS)then
            cnt_sc=fx_sc*xlum_sc*s_sc*jac_cnt(2)
            cnt_wgt_sc=cnt_wgt_sc+cnt_sc*(prefact_c+prefact_coll)*rwgt
            cnt_swgt_sc=cnt_swgt_sc+
     &           cnt_sc*(prefact_cnt_ssc_c+prefact_coll_c)*rwgt
            call sreal_deg(p1_cnt(0,1,2),zero,one,
     #                     deg_xi_sc,deg_lxi_sc)
            deg_wgt=deg_wgt-( deg_xi_sc+deg_lxi_sc*log(xi_i_fks_cnt(ione)) )*
     #                     jac_cnt(2)*prefact_deg*rwgt/(shat/(32*pi**2))*
     #                     xlum_sc
            deg_swgt=deg_swgt-( deg_xi_sc*prefact_deg_sxi +
     #                     deg_lxi_sc*prefact_deg_slxi )*
     #                     jac_cnt(2)*rwgt/(shat/(32*pi**2))*
     #                     xlum_sc
           endif
        endif
      endif
      xmcME=xmcME*(1-gfactsf)*probne
      xmc_wgt=xmc_wgt+xmcMC+xmcME

 547  continue

c Real contribution
c
c Set the ybst_til_tolab before applying the cuts. 
      call set_cms_stuff(mohdr)
      if (passcuts(pp,rwgt).and.probne.lt.1.d0) then
        call set_alphaS(pp)
        x = abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
        ffact = f_damp(x)
        s_ev = fks_Sij(pp,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
        if(s_ev.gt.0.d0)then
          call sreal(pp,xi_i_fks_ev,y_ij_fks_ev,fx_ev)
          xlum_ev = dlum()
          ev_wgt = fx_ev*xlum_ev*s_ev*ffact*wgt*prefact*rwgt*(1-probne)
        endif
        if(AddInfoLHE)scale2_lhe=get_ptrel(pp,i_fks,j_fks)
      endif

      if(AddInfoLHE.and.UseCKKW)then
        if(scale1_lhe.eq.0.d0)scale1_lhe=scale2_lhe
        scale2_lhe=scale_CKKW
      endif

c
c Enhance the one channel for multi-channel integration
c
      enhance=1.d0
      if ((xmc_wgt.ne.0d0.or.cnt_wgt_c.ne.0d0.or.cnt_wgt_s.ne.0d0.or.
     &     cnt_wgt_sc.ne.0d0.or.bsv_wgt.ne.0d0.or.deg_wgt.ne.0d0.or.
     &     deg_swgt.ne.0d0.or.cnt_swgt_s.ne.0d0.or.cnt_swgt_sc.ne.0d0.or.
     &     ev_wgt.ne.0d0) .and. multi_channel) then
         if (bsv_wgt.eq.0d0.and.deg_wgt.eq.0d0.and.deg_swgt.eq.0d0.and.
     &       cnt_wgt_c.eq.0d0 ) CalculatedBorn=.false.

         if (.not.calculatedBorn .and. p_born(0,1).gt.0d0)then
            call sborn(p_born,wgt1)
         elseif(p_born(0,1).lt.0d0)then
            enhance=0d0
         endif

         if (enhance.eq.0d0)then
            xnoborn_cnt=xnoborn_cnt+1.d0
            if(log10(xnoborn_cnt).gt.inoborn_cnt)then
               write (*,*) 
     #           'Function dsigS: no Born momenta more than 10**',
     #           inoborn_cnt,'times'
               inoborn_cnt=inoborn_cnt+1
            endif
         else
            xtot=0d0
            if (mapbconf(0).eq.0) then
               write (*,*) 'Fatal error in dsigS, no Born diagrams '
     &           ,mapbconf,'. Check bornfromreal.inc'
               write (*,*) 'Is fks_singular compiled correctly?'
               stop
            endif
            if (onlyBorn) then
               do i=1, mapbconf(0)
                  if (multi_chan(mapbconf(i))) then
                     xtot=xtot+amp2(mapbconf(i))
                  endif
               enddo
            else
               do i=1,mapbconf(0)
                  xtot=xtot+amp2(mapbconf(i))
               enddo
            endif
            if (xtot.ne.0d0) then
               enhance=amp2(b_from_r(mapconfig(iconfig)))/xtot
               enhance=enhance*diagramsymmetryfactor
            else
               enhance=0d0
            endif
         endif
      endif

      cnt_wgt = cnt_wgt_c + cnt_wgt_s + cnt_wgt_sc
      cnt_swgt = cnt_swgt_s + cnt_swgt_sc

      ev_wgt = ev_wgt * enhance
      xmc_wgt = xmc_wgt * enhance
      cnt_wgt = cnt_wgt * enhance
      cnt_swgt = cnt_swgt * enhance
      bsv_wgt = bsv_wgt * enhance
      born_wgt = born_wgt * enhance
      deg_wgt = deg_wgt * enhance
      deg_swgt = deg_swgt * enhance

      dsigS = (ev_wgt+xmc_wgt+cnt_wgt)*fkssymmetryfactor +
     &     cnt_swgt*fkssymmetryfactor +
     &     bsv_wgt*fkssymmetryfactorBorn +
     &     deg_wgt*fkssymmetryfactorDeg +
     &     deg_swgt*fkssymmetryfactorDeg

      if (dsigS.ne.dsigS) then
         write (*,*) 'ERROR, ',dsigS,
     &        ' found for dsigS, setting dsigS to 0 for this event'
         dsigS=0
      endif

      if(dsigS.ne.0.d0)
     #  call set_shower_scale(p1_cnt(0,1,0),i_fks,j_fks,
     #                        xi_i_fks_ev,y_ij_fks_ev)
c For tests
      if(abs(dsigS).gt.fksmaxwgt)then
        fksmaxwgt=abs(dsigS)
        xisave=xi_i_fks_ev
        ysave=y_ij_fks_ev
      endif
c Plot observables for counterevents and Born
      if (.not.unwgt) then
         plot_wgt=( (ev_wgt+xmc_wgt+cnt_wgt)*fkssymmetryfactor +
     &        cnt_swgt*fkssymmetryfactor +
     &        bsv_wgt*fkssymmetryfactorBorn +
     &        deg_wgt*fkssymmetryfactorDeg +
     &        deg_swgt*fkssymmetryfactorDeg )*vegaswgt
         if( abs(plot_wgt).gt.1.d-20.and.p1_cnt(0,1,0).ne.-99d0 .and.
     &       (plotEv.or.plotKin) )
     &      call outfun(p1_cnt(0,1,0),ybst_til_tolab,plot_wgt,iplot_cnt)
      endif

      return
      end


      subroutine set_shower_scale(pp,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      implicit none
      include "nexternal.inc"
      include "madfks_mcatnlo.inc"
      double precision pp(0:3,nexternal)
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision shat,dot
      external dot
      integer i_fks,j_fks
      logical Hevents
      common/SHevents/Hevents
      double precision emsca
      common/cemsca/emsca
      character*4 abrv
      common /to_abrv/ abrv

c MC shower scale
      double precision SCALUP
      common /cshowerscale/SCALUP
c
      if(pp(0,1).lt.0.d0)then
        write(*,*)'Error #0 in set_shower_scale, for events:',Hevents
        stop
      endif
      shat=2d0*dot(pp(0,1),pp(0,2))
      if(Hevents)then
        SCALUP=sqrt(shat)
      else
        if(dampMCsubt.and.abrv(1:4).ne.'born'.and.abrv(1:2).ne.'vi')then
          SCALUP=min( emsca,sqrt(shat) )
        else
          SCALUP=sqrt(shat)
        endif
      endif
c
      if(SCALUP.le.0.d0)then
        write(*,*)'Scale too small in set_shower_scale:',SCALUP
        stop
      endif
c
      return
      end


      subroutine sreal(pp,xi_i_fks,y_ij_fks,wgt)
c Wrapper for the n+1 contribution. Returns the n+1 matrix element
c squared reduced by the FKS damping factor xi**2*(1-y).
c Close to the soft or collinear limits it calls the corresponding
c Born and multiplies with the AP splitting function or eikonal factors.
      implicit none
      include "nexternal.inc"
      include "coupl.inc"

      double precision pp(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks

      double precision shattmp,dot
      integer i,j

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision zero,tiny
      parameter (zero=0d0)

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision pmass(nexternal)
      include "pmass.inc"

      if (softtest.or.colltest) then
         tiny=1d-8
      else
         tiny=1d-6
      endif

      if(pp(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
        wgt=0.d0
        return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      shattmp=2d0*dot(pp(0,1),pp(0,2))
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

      if (1d0-y_ij_fks.lt.tiny)then
         if (pmass(j_fks).eq.zero.and.j_fks.le.2)then
            call sborncol_isr(pp,xi_i_fks,y_ij_fks,wgt)
         elseif (pmass(j_fks).eq.zero.and.j_fks.ge.3)then
            call sborncol_fsr(pp,xi_i_fks,y_ij_fks,wgt)
         else
            wgt=0d0
         endif
      elseif (xi_i_fks.lt.tiny)then
         if (i_type.eq.8 .and. pmass(i_fks).eq.0d0)then
c i_fks is gluon
            call sbornsoft(pp,xi_i_fks,y_ij_fks,wgt)
         elseif (abs(i_type).eq.3)then
c i_fks is (anti-)quark
            wgt=0d0
         else
            write(*,*) 'FATAL ERROR #1 in sreal',i_type,i_fks
            stop
         endif
      else
         call smatrix(pp,wgt)
         wgt=wgt*xi_i_fks**2*(1d0-y_ij_fks)
      endif

      if(wgt.lt.0.d0)then
         write(*,*) 'Fatal error #2 in sreal',wgt,xi_i_fks,y_ij_fks
         do i=1,nexternal
            write(*,*) 'particle ',i,', ',(pp(j,i),j=0,3)
         enddo
         stop
      endif

      return
      end



      subroutine sborncol_fsr(p,xi_i_fks,y_ij_fks,wgt)
      implicit none
      include "nexternal.inc"
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
C  
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      logical rotategranny
      common/crotategranny/rotategranny

      double precision cthbe,sthbe,cphibe,sphibe
      common/cbeangles/cthbe,sthbe,cphibe,sphibe

      double precision p_born_rot(0:3,nexternal-1)

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip

      integer i,imother_fks
      double precision t,z,ap,E_j_fks,E_i_fks,Q,cphi_mother,
     # sphi_mother,pi(0:3),pj(0:3)
      double complex wgt1(2),W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))
C  
      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_fsr"
         wgt=0.d0
         return
      endif

      E_j_fks = p(0,j_fks)
      E_i_fks = p(0,i_fks)
      z = 1d0 - E_i_fks/(E_i_fks+E_j_fks)
      t = z * shat/4d0
      if(rotategranny .and. nexternal-1.ne.3)then
c Exclude 2->1 (at the Born level) processes: matrix elements are
c independent of the PS point, but non-zero helicity configurations
c might flip when rotating the momenta.
        do i=1,nexternal-1
          call trp_rotate_invar(p_born(0,i),p_born_rot(0,i),
     #                          cthbe,sthbe,cphibe,sphibe)
        enddo
        call sborn(p_born_rot,wgt1)
        CalculatedBorn=.false.
      else
        call sborn(p_born,wgt1)
      endif
      call AP_reduced(j_type,i_type,t,z,ap)
      if (abs(j_type).eq.3 .and. i_type.eq.8) then
         Q=0d0
         wgt1(2)=0d0
      elseif (m_type.eq.8) then
c Insert <ij>/[ij] which is not included by sborn()
         if (1d0-y_ij_fks.lt.vtiny)then
            azifact=xij_aor
         else
            do i=0,3
               pi(i)=p_i_fks_ev(i)
               pj(i)=p(i,j_fks)
            enddo
            if(rotategranny)then
              call trp_rotate_invar(pi,pi,cthbe,sthbe,cphibe,sphibe)
              call trp_rotate_invar(pj,pj,cthbe,sthbe,cphibe,sphibe)
            endif
            CALL IXXXXX(pi ,ZERO ,+1,+1,W1)        
            CALL OXXXXX(pj ,ZERO ,-1,+1,W2)        
            CALL IXXXXX(pi ,ZERO ,-1,+1,W3)        
            CALL OXXXXX(pj ,ZERO ,+1,+1,W4)        
            Wij_angle=(0d0,0d0)
            Wij_recta=(0d0,0d0)
            do i=1,4
               Wij_angle = Wij_angle + W1(i)*W2(i)
               Wij_recta = Wij_recta + W3(i)*W4(i)
            enddo
            azifact=Wij_angle/Wij_recta
         endif
c Insert the extra factor due to Madgraph convention for polarization vectors
         imother_fks=min(i_fks,j_fks)
         if(rotategranny)then
           call getaziangles(p_born_rot(0,imother_fks),
     #                       cphi_mother,sphi_mother)
         else
           call getaziangles(p_born(0,imother_fks),
     #                       cphi_mother,sphi_mother)
         endif
         wgt1(2) = -(cphi_mother-ximag*sphi_mother)**2 *
     #             wgt1(2) * azifact
         call Qterms_reduced_timelike(j_type, i_type, t, z, Q)
      else
         write(*,*) 'FATAL ERROR in sborncol_fsr',i_type,j_type,i_fks,j_fks
         stop
      endif
      wgt=dble(wgt1(1)*ap+wgt1(2)*Q)
      return
      end



      subroutine sborncol_isr(p,xi_i_fks,y_ij_fks,wgt)
      implicit none
      include "nexternal.inc"
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
C  
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision p_born_rot(0:3,nexternal-1)

      integer i
      double precision t,z,ap,Q,cphi_mother,sphi_mother,pi(0:3),pj(0:3)
      double complex wgt1(2),W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))
C  
      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_isr"
         wgt=0.d0
         return
      endif

      z = 1d0 - xi_i_fks
c sreal return {\cal M} of FKS except for the partonic flux 1/(2*s).
c Thus, an extra factor z (implicit in the flux of the reduced Born
c in FKS) has to be inserted here
      t = z*shat/4d0
      if(j_fks.eq.2 .and. nexternal-1.ne.3)then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed.
c Exclude 2->1 (at the Born level) processes: matrix elements are
c independent of the PS point, but non-zero helicity configurations
c might flip when rotating the momenta.
        do i=1,nexternal-1
          p_born_rot(0,i)=p_born(0,i)
          p_born_rot(1,i)=-p_born(1,i)
          p_born_rot(2,i)=p_born(2,i)
          p_born_rot(3,i)=-p_born(3,i)
        enddo
        call sborn(p_born_rot,wgt1)
        CalculatedBorn=.false.
      else
        call sborn(p_born,wgt1)
      endif
      call AP_reduced(m_type,i_type,t,z,ap)
      if (abs(m_type).eq.3) then
         Q=0d0
         wgt1(2)=0d0
      else
c Insert <ij>/[ij] which is not included by sborn()
         if (1d0-y_ij_fks.lt.vtiny)then
            azifact=xij_aor
         else
            do i=0,3
               pi(i)=p_i_fks_ev(i)
               pj(i)=p(i,j_fks)
            enddo
            if(j_fks.eq.2)then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed
               pi(1)=-pi(1)
               pi(3)=-pi(3)
               pj(1)=-pj(1)
               pj(3)=-pj(3)
            endif
            CALL IXXXXX(pi ,ZERO ,+1,+1,W1)        
            CALL OXXXXX(pj ,ZERO ,-1,+1,W2)        
            CALL IXXXXX(pi ,ZERO ,-1,+1,W3)        
            CALL OXXXXX(pj ,ZERO ,+1,+1,W4)        
            Wij_angle=(0d0,0d0)
            Wij_recta=(0d0,0d0)
            do i=1,4
               Wij_angle = Wij_angle + W1(i)*W2(i)
               Wij_recta = Wij_recta + W3(i)*W4(i)
            enddo
            azifact=Wij_angle/Wij_recta
         endif
c Insert the extra factor due to Madgraph convention for polarization vectors
         if(j_fks.eq.2)then
           cphi_mother=-1.d0
           sphi_mother=0.d0
         else
           cphi_mother=1.d0
           sphi_mother=0.d0
         endif
         wgt1(2) = -(cphi_mother+ximag*sphi_mother)**2 *
     #             wgt1(2) * dconjg(azifact)
         call Qterms_reduced_spacelike(m_type, i_type, t, z, Q)
      endif
      wgt=dble(wgt1(1)*ap+wgt1(2)*Q)
      return
      end



      subroutine AP_reduced(part1, part2, t, z, ap)
c Returns Altarelli-Parisi splitting function summed/averaged over helicities
c times prefactors such that |M_n+1|^2 = ap * |M_n|^2. This means
c    AP_reduced = (1-z) P_{S(part1,part2)->part1+part2}(z) * gS^2/t
c Therefore, the labeling conventions for particle IDs are not as in FKS:
c part1 and part2 are the two particles emerging from the branching.
c part1 and part2 can be either gluon (8) or (anti-)quark (+-3). z is the
c fraction of the energy of part1 and t is the invariant mass of the mother.
      implicit none

      integer part1, part2
      double precision z,ap,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         ap = 2d0 * CA * ( (1d0-z)**2/z + z + z*(1d0-z)**2 )

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         ap = TR * ( z**2 + (1d0-z)**2 )*(1d0-z)
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         ap = CF * (1d0+z**2)

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         ap = CF * (1d0+(1d0-z)**2)*(1d0-z)/z
      else
         write (*,*) 'Fatal error in AP_reduced',part1,part2
         stop
      endif

      ap = ap*g**2/t

      return
      end



      subroutine AP_reduced_prime(part1, part2, t, z, apprime)
c Returns (1-z)*P^\prime * gS^2/t, with the same conventions as AP_reduced
      implicit none

      integer part1, part2
      double precision z,apprime,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         apprime = 0d0

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         apprime = -2 * TR * z * (1d0-z)**2
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         apprime = - CF * (1d0-z)**2

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         apprime = - CF * z * (1d0-z)
      else
         write (*,*) 'Fatal error in AP_reduced_prime',part1,part2
         stop
      endif

      apprime = apprime*g**2/t

      return
      end



      subroutine Qterms_reduced_timelike(part1, part2, t, z, Qterms)
c Eq's B.31 to B.34 of FKS paper, times (1-z)*gS^2/t. The labeling
c conventions for particle IDs are the same as those in AP_reduced
      implicit none

      integer part1, part2
      double precision z,Qterms,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         Qterms = -4d0 * CA * z*(1d0-z)**2

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         Qterms = 4d0 * TR * z*(1d0-z)**2
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         Qterms = 0d0

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         Qterms = 0d0
      else
         write (*,*) 'Fatal error in Qterms_reduced_timelike',part1,part2
         stop
      endif

      Qterms = Qterms*g**2/t

      return
      end



      subroutine Qterms_reduced_spacelike(part1, part2, t, z, Qterms)
c Eq's B.42 to B.45 of FKS paper, times (1-z)*gS^2/t. The labeling
c conventions for particle IDs are the same as those in AP_reduced.
c Thus, part1 has momentum fraction z, and it is the one off-shell
c (see (FKS.B.41))
      implicit none

      integer part1, part2
      double precision z,Qterms,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         Qterms = -4d0 * CA * (1d0-z)**2/z

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         Qterms = 0d0
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         Qterms = 0d0

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         Qterms = -4d0 * CF * (1d0-z)**2/z
      else
         write (*,*) 'Fatal error in Qterms_reduced_spacelike',part1,part2
         stop
      endif

      Qterms = Qterms*g**2/t

      return
      end


      subroutine AP_reduced_SUSY(part1, part2, t, z, ap)
c Same as AP_reduced, except for the fact that it only deals with
c   go -> go g
c   sq -> sq g
c splittings in SUSY. We assume this function to be called with 
c part2==colour(i_fks)
      implicit none

      integer part1, part2
      double precision z,ap,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part2.ne.8)then
         write (*,*) 'Fatal error #0 in AP_reduced_SUSY',part1,part2
         stop
      endif

      if (part1.eq.8)then
c go->gog splitting
         ap = CA * (1d0+z**2)

      elseif(abs(part1).eq.3)then
c sq->sqg splitting
         ap = 2d0 * CF * z

      else
         write (*,*) 'Fatal error in AP_reduced_SUSY',part1,part2
         stop
      endif

      ap = ap*g**2/t

      return
      end


      subroutine sbornsoft(pp,xi_i_fks,y_ij_fks,wgt)
      implicit none

      include "nexternal.inc"
      include "fks.inc"
      include "coupl.inc"

      integer m,n

      double precision softcontr,pp(0:3,nexternal),wgt,eik,xi_i_fks,y_ij_fks
      integer i,j

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision zero,pmass(nexternal)
      parameter(zero=0d0)
      include "pmass.inc"

      softcontr=0d0

      do i=1,fks_j_from_i(i_fks,0)
         do j=1,i
            m=fks_j_from_i(i_fks,i)
            n=fks_j_from_i(i_fks,j)
            if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)) .and.
     &           n.ne.i_fks.and.m.ne.i_fks) then
               call sborn_sf(p_born,m,n,wgt)
               if (wgt.ne.0d0) then
                  call eikonal_reduced(pp,m,n,i_fks,j_fks,
     #                                 xi_i_fks,y_ij_fks,eik)
                  softcontr=softcontr+wgt*eik
               endif
            endif
         enddo
      enddo

      wgt=softcontr
c Add minus sign to compensate the minus in the color factor
c of the color-linked Borns (b_sf_0??.f)
c Factor two to fix the limits.
      wgt=-2d0*wgt
      return
      end






      subroutine eikonal_reduced(pp,m,n,i_fks,j_fks,xi_i_fks,y_ij_fks,eik)
c     Returns the eikonal factor
      implicit none

      include "nexternal.inc"
      double precision eik,pp(0:3,nexternal),xi_i_fks,y_ij_fks
      double precision dot,dotnm,dotni,dotmi,fact
      integer n,m,i_fks,j_fks,i
      integer softcol

      include "coupl.inc"

      external dot
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      real*8 phat_i_fks(0:3)

      double precision zero,pmass(nexternal),tiny
      parameter(zero=0d0)
      parameter(tiny=1d-6)
      include "pmass.inc"

c Define the reduced momentum for i_fks
      softcol=0
      if (1d0-y_ij_fks.lt.tiny)softcol=2
      if(p_i_fks_cnt(0,softcol).lt.0d0)then
        if(xi_i_fks.eq.0.d0)then
           write (*,*) 'Error #1 in eikonal_reduced',
     #                 softcol,xi_i_fks,y_ij_fks
           stop
        endif
        if(pp(0,i_fks).ne.0.d0)then
          write(*,*)'WARNING in eikonal_reduced: no cnt momenta',
     #      softcol,xi_i_fks,y_ij_fks
          do i=0,3
            phat_i_fks(i)=pp(i,i_fks)/xi_i_fks
          enddo
        else
          write (*,*) 'Error #2 in eikonal_reduced',
     #                 softcol,xi_i_fks,y_ij_fks
          stop
        endif
      else
        do i=0,3
          phat_i_fks(i)=p_i_fks_cnt(i,softcol)
        enddo
      endif
c Calculate the eikonal factor
      dotnm=dot(pp(0,n),pp(0,m))
      if ((m.ne.j_fks .and. n.ne.j_fks) .or. pmass(j_fks).ne.ZERO) then
         dotmi=dot(pp(0,m),phat_i_fks)
         dotni=dot(pp(0,n),phat_i_fks)
         fact= 1d0-y_ij_fks
      elseif (m.eq.j_fks .and. n.ne.j_fks .and.
     &        pmass(j_fks).eq.ZERO) then
         dotni=dot(pp(0,n),phat_i_fks)
         dotmi=sqrtshat/2d0 * pp(0,j_fks)
         fact= 1d0
      elseif (m.ne.j_fks .and. n.eq.j_fks .and.
     &        pmass(j_fks).eq.ZERO) then
         dotni=sqrtshat/2d0 * pp(0,j_fks)
         dotmi=dot(pp(0,m),phat_i_fks)
         fact= 1d0
      else
         write (*,*) 'Error #3 in eikonal_reduced'
         stop
      endif

      eik = dotnm/(dotni*dotmi)*fact

      eik = eik * g**2

      return
      end


      subroutine sreal_deg(p,xi_i_fks,y_ij_fks,
     #                     collrem_xi,collrem_lxi)
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "coupl.inc"
      include "run.inc"

      double precision p(0:3,nexternal),collrem_xi,collrem_lxi
      double precision xi_i_fks,y_ij_fks

      double complex wgt1(2)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision delta_used
      common /cdelta_used/delta_used

      double precision rwgt,shattmp,dot,born_wgt,oo2pi,z,t,ap,
     # apprime,xkkern,xnorm
      external dot

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type
      
      double precision one,pi
      parameter (one=1.d0)
      parameter (pi=3.1415926535897932385d0)

      if(j_fks.gt.nincoming)then
c Do not include this contribution for final-state branchings
         collrem_xi=0.d0
         collrem_lxi=0.d0
         return
      endif

      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sreal_deg"
         collrem_xi=0.d0
         collrem_lxi=0.d0
         return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      shattmp=2d0*dot(p(0,1),p(0,2))
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

      call sborn(p_born,wgt1)
      born_wgt=dble(wgt1(1))

c A factor gS^2 is included in the Altarelli-Parisi kernels
      oo2pi=one/(8d0*PI**2)

      z = 1d0 - xi_i_fks
      t = one
      call AP_reduced(m_type,i_type,t,z,ap)
      call AP_reduced_prime(m_type,i_type,t,z,apprime)

c Insert here proper functions for PDF change of scheme. With xkkern=0.d0
c one assumes MSbar
      xkkern=0.d0

      collrem_xi=ap*log(shat*delta_used/(2*q2fact(j_fks))) -
     #           apprime - xkkern 
      collrem_lxi=2*ap

c The partonic flux 1/(2*s) is inserted in genps. Thus, an extra 
c factor z (implicit in the flux of the reduced Born in FKS) 
c has to be inserted here
      xnorm=1.d0/z

      collrem_xi=oo2pi * born_wgt * collrem_xi * xnorm
      collrem_lxi=oo2pi * born_wgt * collrem_lxi * xnorm

      return
      end


      subroutine set_cms_stuff(icountevts)
      implicit none
      include "run.inc"

      integer icountevts

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev

      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt

      double precision tau_ev,ycm_ev
      common/cbjrk12_ev/tau_ev,ycm_ev

      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt

      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt

      integer imirror
      common/cmirror/imirror

c rapidity of boost from \tilde{k}_1+\tilde{k}_2 c.m. frame to lab frame --
c same for event and counterevents
c This is the rapidity that enters in the arguments of the sinh() and
c cosh() of the boost, in such a way that
c       y(k)_lab = y(k)_tilde - ybst_til_tolab
c where y(k)_lab and y(k)_tilde are the rapidities computed with a generic
c four-momentum k, in the lab frame and in the \tilde{k}_1+\tilde{k}_2 
c c.m. frame respectively
      ybst_til_tolab=-ycm_cnt(0)
      if(icountevts.eq.-100)then
c set Bjorken x's in run.inc for the computation of PDFs in auto_dsig
        xbk(1)=xbjrk_ev(1)
        xbk(2)=xbjrk_ev(2)
c shat=2*k1.k2 -- consistency of this assignment with momenta checked
c in phspncheck_nocms
        shat=shat_ev
        sqrtshat=sqrtshat_ev
c rapidity of boost from \tilde{k}_1+\tilde{k}_2 c.m. frame to 
c k_1+k_2 c.m. frame
        ybst_til_tocm=ycm_ev-ycm_cnt(0)
      else
c do the same as above for the counterevents
        xbk(1)=xbjrk_cnt(1,icountevts)
        xbk(2)=xbjrk_cnt(2,icountevts)
        shat=shat_cnt(icountevts)
        sqrtshat=sqrtshat_cnt(icountevts)
        ybst_til_tocm=ycm_cnt(icountevts)-ycm_cnt(0)
      endif

c if mirroring just flip the sign of ybsts
      if (imirror.eq.2) then
c        ybst_til_tolab = - ybst_til_tolab
c        ybst_til_tocm  = - ybst_til_tocm
      endif
      return
      end


      subroutine get_mc_lum(j_fks,zhw_used,xi_i_fks,xlum_mc_save,xlum_mc)
      implicit none
      include "run.inc"
      include "nexternal.inc"
      integer j_fks
      double precision dlum
      external dlum
      double precision zhw_used,xi_i_fks,xlum_mc_save,xlum_mc
      
      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt
      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo
      logical Hevents
      common/SHevents/Hevents

      if(zhw_used.lt.0.d0.or.zhw_used.gt.1.d0)then
        write(*,*)'Error #1 in get_mc_lum',zhw_used
        stop
      endif
      if(j_fks.gt.nincoming)then
        if(xlum_mc_save.ne.-1.d8)then
          xlum_mc=xlum_mc_save
        else
          xbk(1)=xbjrk_cnt(1,0)
          xbk(2)=xbjrk_cnt(2,0)
          xlum_mc = dlum()
          xlum_mc_save = xlum_mc
        endif
      elseif(j_fks.eq.1)then
        if (MonteCarlo.eq.'HERWIG6') then
          xbk(1)=xbjrk_cnt(1,0)/zhw_used
          xbk(2)=xbjrk_cnt(2,0)
        elseif (MonteCarlo.eq.'PYTHIA6Q') then
          if(Hevents)then
            xbk(1)=xbjrk_ev(1)
            xbk(2)=xbjrk_ev(2)
          else
            xbk(1)=xbjrk_cnt(1,0)/zhw_used
            xbk(2)=xbjrk_cnt(2,0)
          endif
        else
          write(*,*)'Error in get_mc_lum'
          write(*,*)'Unknown MC type:',MonteCarlo
          stop
        endif
        if(xbk(1).gt.1.d0)then
          xlum_mc = 0.d0
        else
          xlum_mc = dlum()
          xlum_mc = xlum_mc * (1-xi_i_fks)/zhw_used
        endif
      elseif(j_fks.eq.2)then
        if (MonteCarlo.eq.'HERWIG6') then
          xbk(1)=xbjrk_cnt(1,0)
          xbk(2)=xbjrk_cnt(2,0)/zhw_used
        elseif (MonteCarlo.eq.'PYTHIA6Q') then
          if(Hevents)then
            xbk(1)=xbjrk_ev(1)
            xbk(2)=xbjrk_ev(2)
          else
            xbk(1)=xbjrk_cnt(1,0)
            xbk(2)=xbjrk_cnt(2,0)/zhw_used
          endif
        else
          write(*,*)'Error in get_mc_lum'
          write(*,*)'Unknown MC type:',MonteCarlo
          stop
        endif
        if(xbk(2).gt.1.d0)then
          xlum_mc = 0.d0
        else
          xlum_mc = dlum()
          xlum_mc = xlum_mc * (1-xi_i_fks)/zhw_used
        endif
      else
        write(*,*)'Error in get_mc_lum: unknown j_fks',j_fks
        stop
      endif
      if( xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #    ( (xbk(1).gt.1.d0.or.xbk(2).gt.1.d0).and.
     #      j_fks.gt.nincoming ) .or.
     #    (xbk(2).gt.1.d0.and.j_fks.eq.1) .or.
     #    (xbk(1).gt.1.d0.and.j_fks.eq.2) )then
        write(*,*)'Error in get_mc_lum: x_i',xbk(1),xbk(2)
        stop
      endif
      return
      end


      subroutine xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,
     #                        p_i_fks_ev,p_i_fks_cnt,
     #                        xi_i_fks_ev,y_ij_fks_ev)
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      integer i_fks,j_fks
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision jac,jac_cnt(-2:2)
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      double precision xi_i_fks_ev,y_ij_fks_ev
      integer izero,ione,itwo,iunit,isum
      logical verbose
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)
      parameter (iunit=6)
      parameter (verbose=.false.)
c
      isum=0
      if(jac_cnt(0).gt.0.d0)isum=isum+1
      if(jac_cnt(1).gt.0.d0)isum=isum+2
      if(jac_cnt(2).gt.0.d0)isum=isum+4
c
      if(isum.eq.0.or.isum.eq.1.or.isum.eq.2.or.isum.eq.4)then
c Nothing to be done: 0 or 1 configurations computed
        if(verbose)write(iunit,*)'none'
      elseif(isum.eq.3.or.isum.eq.5.or.isum.eq.7)then
c Soft is taken as reference
        if(isum.eq.7)then
          if(verbose)then
            write(iunit,*)'all'
            write(iunit,*)'    '
            write(iunit,*)'C/S'
          endif
          call xmcompare(verbose,ione,izero,i_fks,j_fks,p,p1_cnt)
          if(verbose)then
            write(iunit,*)'    '
            write(iunit,*)'SC/S'
          endif
          call xmcompare(verbose,itwo,izero,i_fks,j_fks,p,p1_cnt)
        elseif(isum.eq.3)then
          if(verbose)then
            write(iunit,*)'C+S'
            write(iunit,*)'    '
            write(iunit,*)'C/S'
          endif
          call xmcompare(verbose,ione,izero,i_fks,j_fks,p,p1_cnt)
        elseif(isum.eq.5)then
          if(verbose)then
            write(iunit,*)'SC+S'
            write(iunit,*)'    '
            write(iunit,*)'SC/S'
          endif
          call xmcompare(verbose,itwo,izero,i_fks,j_fks,p,p1_cnt)
        endif
      elseif(isum.eq.6)then
c Collinear is taken as reference
        if(verbose)then
          write(iunit,*)'SC+C'
          write(iunit,*)'    '
          write(iunit,*)'SC/C'
        endif
        call xmcompare(verbose,itwo,ione,i_fks,j_fks,p,p1_cnt)
      else
        write(6,*)'Fatal error in xmom_compare',isum
        stop
      endif
c
      if(jac_cnt(0).gt.0.d0.and.jac.gt.0.d0)
     #  call p_ev_vs_cnt(izero,i_fks,j_fks,p,p1_cnt,
     #                   p_i_fks_ev,p_i_fks_cnt,
     #                   xi_i_fks_ev,y_ij_fks_ev)
      if(jac_cnt(1).gt.0.d0.and.jac.gt.0.d0)
     #  call p_ev_vs_cnt(ione,i_fks,j_fks,p,p1_cnt,
     #                   p_i_fks_ev,p_i_fks_cnt,
     #                   xi_i_fks_ev,y_ij_fks_ev)
c
      return
      end


      subroutine xmcompare(verbose,inum,iden,i_fks,j_fks,p,p1_cnt)
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      include 'coupl.inc'
      logical verbose
      integer inum,iden,i_fks,j_fks,iunit,ipart,i,j,k
      double precision tiny,vtiny,xnum,xden,xrat
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      parameter (iunit=6)
      parameter (tiny=1.d-4)
      parameter (vtiny=1.d-10)
      double precision pmass(nexternal),zero
      parameter (zero=0d0)
      include "pmass.inc"
c
      do ipart=1,nexternal
        do i=0,3
          xnum=p1_cnt(i,ipart,inum)
          xden=p1_cnt(i,ipart,iden)
          if(verbose)then
            if(i.eq.0)then
              write(iunit,*)' '
              write(iunit,*)'part=',ipart
            endif
            call xprintout(iunit,xnum,xden)
          else
            if(ipart.ne.i_fks.and.ipart.ne.j_fks)then
              if(xden.ne.0.d0)then
                xrat=abs(1-xnum/xden)
              else
                xrat=abs(xnum)
              endif
              if(xrat.gt.tiny .and.
     &          (pmass(ipart).eq.0d0.or.xnum/pmass(ipart).gt.vtiny))then
                 write(*,*)'Kinematics of counterevents'
                 write(*,*)inum,iden
                 write(*,*)'is different. Particle:',ipart
                 write(*,*) xrat,xnum,xden
                 do j=1,6
                    write(*,*) j,(p1_cnt(k,j,0),k=0,3)
                 enddo
                 do j=1,6
                    write(*,*) j,(p1_cnt(k,j,1),k=0,3)
                 enddo
                 stop
              endif
            endif
          endif
        enddo
      enddo
      do i=0,3
        if(j_fks.gt.2)then
          xnum=p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
          xden=p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        else
          xnum=-p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
          xden=-p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        endif
        if(verbose)then
          if(i.eq.0)then
            write(iunit,*)' '
            write(iunit,*)'part=i+j'
          endif
          call xprintout(iunit,xnum,xden)
        else
          if(xden.ne.0.d0)then
            xrat=abs(1-xnum/xden)
          else
            xrat=abs(xnum)
          endif
          if(xrat.gt.tiny)then
            write(*,*)'Kinematics of counterevents'
            write(*,*)inum,iden
            write(*,*)'is different. Particle i+j'
            stop
          endif
        endif
      enddo
      return
      end


      subroutine xmcompare_fsr(verbose,inum,iden,i_fks,j_fks,p,p1_cnt)
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      logical verbose
      integer inum,iden,i_fks,j_fks,iunit,ipart,i
      double precision tiny,xnum,xden,xrat
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      parameter (iunit=6)
      parameter (tiny=1.d-4)
c
      do ipart=1,nexternal
        do i=0,3
          xnum=p1_cnt(i,ipart,inum)
          xden=p1_cnt(i,ipart,iden)
          if(verbose)then
            if(i.eq.0)then
              write(iunit,*)' '
              write(iunit,*)'part=',ipart
            endif
            call xprintout(iunit,xnum,xden)
          else
            if(ipart.ne.i_fks.and.ipart.ne.j_fks)then
              if(xden.ne.0.d0)then
                xrat=abs(1-xnum/xden)
              else
                xrat=abs(xnum)
              endif
              if(xrat.gt.tiny)then
                write(*,*)'Kinematics of counterevents'
                write(*,*)inum,iden
                write(*,*)'is different. Particle:',ipart
                stop
              endif
            endif
          endif
        enddo
      enddo
      do i=0,3
        xnum=p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
        xden=p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        if(verbose)then
          if(i.eq.0)then
            write(iunit,*)' '
            write(iunit,*)'part=i+j'
          endif
          call xprintout(iunit,xnum,xden)
        else
          if(xden.ne.0.d0)then
            xrat=abs(1-xnum/xden)
          else
            xrat=abs(xnum)
          endif
          if(xrat.gt.tiny)then
            write(*,*)'Kinematics of counterevents'
            write(*,*)inum,iden
            write(*,*)'is different. Particle i+j'
            stop
          endif
        endif
      enddo
      return
      end


      subroutine xprintout(iunit,xv,xlim)
      implicit real*8(a-h,o-z)
c
      if(abs(xlim).gt.1.d-30)then
        write(iunit,*)xv/xlim,xv,xlim
      else
        write(iunit,*)xv,xlim
      endif
      return
      end


      subroutine p_ev_vs_cnt(icnt,i_fks,j_fks,p,p1_cnt,
     #                       p_i_fks_ev,p_i_fks_cnt,
     #                       xi_i_fks_ev,y_ij_fks_ev)
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      integer icnt,i_fks,j_fks,ipart,i
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      double precision xi_i_fks_ev,y_ij_fks_ev,tiny
      double precision rat(0:3,nexternal+3),den(0:3,nexternal+3)
      integer maxrat
c
c This routine is obsolete; the convergence checks are done elsewhere
      return

      do ipart=1,nexternal
        do i=0,3
          den(i,ipart)=p1_cnt(i,ipart,icnt)
          if(den(i,ipart).ne.0.d0)then
            rat(i,ipart)=p(i,ipart)/den(i,ipart)
          else
            rat(i,ipart)=p(i,ipart)
          endif
        enddo
      enddo
c
      do i=0,3
        den(i,nexternal+1)=p1_cnt(i,i_fks,icnt)+p1_cnt(i,j_fks,icnt)
        if(den(i,nexternal+1).ne.0.d0)then
          rat(i,nexternal+1)=(p(i,i_fks)+p(i,j_fks))/den(i,nexternal+1)
        else
          rat(i,nexternal+1)=p(i,i_fks)+p(i,j_fks)
        endif
      enddo
c
      if(icnt.eq.0)then
        tiny=4*xi_i_fks_ev
        maxrat=nexternal+3
        do i=0,3
          den(i,nexternal+2)=p_i_fks_cnt(i,0)
          if(den(i,nexternal+2).ne.0.d0)then
            rat(i,nexternal+2)=p_i_fks_ev(i)/den(i,nexternal+2)
          else
            rat(i,nexternal+2)=p_i_fks_ev(i)
          endif
        enddo
        do i=0,3
          den(i,nexternal+3)=p_i_fks_cnt(i,0)
          if(den(i,nexternal+3).ne.0.d0)then
            rat(i,nexternal+3)=p(i,i_fks)/den(i,nexternal+3)
          else
            rat(i,nexternal+3)=p(i,i_fks)
          endif
        enddo
      else
        tiny=2*sqrt(1-y_ij_fks_ev)
        maxrat=nexternal+1
      endif
c
      return
      end


c The following has been derived with minor modifications from the
c analogous routine written for VBF
      subroutine checkres(xsecvc,xseclvc,wgt,wgtl,xp,lxp,
     #                    iflag,imax,iev,nexternal,i_fks,j_fks,iret)
c Checks that the sequence xsecvc(i), i=1,imax, converges to xseclvc.
c Due to numerical inaccuracies, the test is deemed OK if there are
c at least ithrs+1 consecutive elements in the sequence xsecvc(i)
c which are closer to xseclvc than the preceding element of the sequence.
c The counting is started when an xsecvc(i0) is encountered, which is
c such that |xsecvc(i0)/xseclvc-1|<0.1 if xseclvc#0, or such that
c |xsecvc(i0)|<0.1 if xseclvc=0. In order for xsecvc(i+1 )to be defined 
c closer to xseclvc than xsecvc(i), the condition
c   |xsecvc(i)/xseclvc-1|/|xsecvc(i+1)/xseclvc-1| > rat
c if xseclvc#0, or 
c   |xsecvc(i)|/|xsecvc(i+1)| > rat
c if xseclvc=0 must be fulfilled; the value of rat is set equal to 8 and to 2
c for soft and collinear limits respectively, since the cross section is 
c expected to scale as xii**2 and sqrt(1-yi**2), and the values of xii and yi
c are chosen as powers of 10 (thus, if scaling would be exact, rat should
c be set equal to 10 and sqrt(10)).
c If the test is passed, icount=ithrs, else icount<ithrs; in the former
c case iret=0, in the latter iret=1.
c When the test is not passed, one may choose to stop the program dead here;
c in such a case, set istop=1 below. Each time the test is not passed,
c the results are written onto fort.77; set iwrite=0 to prevent the writing
      implicit none
      real*8 xsecvc(15),xseclvc,wgt(15),wgtl,lxp(0:3,21),xp(15,0:3,21)
      real*8 ckc(15),rckc(15),rat
      integer iflag,imax,iev,nexternal,i_fks,j_fks,iret,ithrs,istop,
     # iwrite,i,k,l,imin,icount
      parameter (ithrs=3)
      parameter (istop=0)
      parameter (iwrite=1)
c
      if(imax.gt.15)then
        write(6,*)'Error in checkres: imax is too large',imax
        stop
      endif
      do i=1,imax
        if(xseclvc.eq.0.d0)then
          ckc(i)=abs(xsecvc(i))
        else
          ckc(i)=abs(xsecvc(i)/xseclvc-1.d0)
        endif
      enddo
      if(iflag.eq.0)then
        rat=8.d0
      elseif(iflag.eq.1)then
        rat=2.d0
      else
        write(6,*)'Error in checkres: iflag=',iflag
        write(6,*)' Must be 0 for soft, 1 for collinear'
        stop
      endif
c
      i=1
      dowhile(ckc(i).gt.0.1d0)
        i=i+1
      enddo
      imin=i
      do i=imin,imax-1
        if(ckc(i+1).ne.0.d0)then
          rckc(i)=ckc(i)/ckc(i+1)
        else
          rckc(i)=1.d8
        endif
      enddo
      icount=0
      i=imin
      dowhile(icount.lt.ithrs.and.i.lt.imax)
        if(rckc(i).gt.rat)then
          icount=icount+1
        else
          icount=0
        endif
        i=i+1
      enddo
c
      iret=0
      if(icount.ne.ithrs)then
        iret=1
        if(istop.eq.1)then
          write(6,*)'Test failed',iflag
          write(6,*)'Event #',iev
          stop
        endif
        if(iwrite.eq.1)then
          write(77,*)'    '
          if(iflag.eq.0)then
            write(77,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(77,*)'Collinear #',iev
          endif
          write(77,*)'ME*wgt:'
          do i=1,imax
             call xprintout(77,xsecvc(i),xseclvc)
          enddo
          write(77,*)'wgt:'
          do i=1,imax
             call xprintout(77,wgt(i),wgtl)
          enddo
c
          write(78,*)'    '
          if(iflag.eq.0)then
            write(78,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(78,*)'Collinear #',iev
          endif
          do k=1,nexternal
            write(78,*)''
            write(78,*)'part:',k
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,k),lxp(l,k))
              enddo
            enddo
          enddo
          if(iflag.eq.0)then
            write(78,*)''
            write(78,*)'part: i_fks reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,nexternal+1),
     #                            lxp(l,nexternal+1))
              enddo
            enddo
            write(78,*)''
            write(78,*)'part: i_fks full/reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks),
     #                            xp(i,l,nexternal+1))
              enddo
            enddo
          elseif(iflag.eq.1)then
            write(78,*)''
            write(78,*)'part: i_fks+j_fks'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks)+xp(i,l,j_fks),
     #                            lxp(l,i_fks)+lxp(l,j_fks))
              enddo
            enddo
          endif
        endif
      endif
      return
      end




      subroutine checksij(xsijvc,xsijlvc,xsijlim,
     #                    xsumvc,xsumlvc,xsumlim,
     #                    check,checkl,tolerance,
     #                    iflag,imax,iev,ki,kk,ll,
     #                    i_fks,j_fks,ilim,iret)
c Analogous to checkres. Relevant to S functions
      implicit none
      real*8 xsijvc(15),xsijlvc,xsumvc(15),xsumlvc,check(15),checkl
      real*8 xsijlim,xsumlim,tolerance
      real*8 xsecvc(15),xseclvc
      real*8 ckc(15),rckc(15),rat
      logical found
      integer iflag,imax,iev,ki,kk,ll,i_fks,j_fks,ilim,iret,ithrs,
     # istop,iwrite,i,imin,icount,itype
      parameter (ithrs=3)
      parameter (istop=0)
      parameter (iwrite=1)
c
      if(imax.gt.15)then
        write(6,*)'Error in checksij: imax is too large',imax
        stop
      endif
      itype=1
      iret=0
 100  continue
      if(itype.eq.1)then
        do i=1,imax
          xsecvc(i)=xsijvc(i)
        enddo
        xseclvc=xsijlvc
      elseif(itype.eq.2)then
        do i=1,imax
          xsecvc(i)=xsumvc(i)
        enddo
        xseclvc=xsumlvc
      else
        write(6,*)'Error in checksij: itype=',itype
        stop
      endif
      do i=1,imax
        if(xseclvc.eq.0.d0)then
          ckc(i)=abs(xsecvc(i))
        else
          ckc(i)=abs(xsecvc(i)/xseclvc-1.d0)
        endif
      enddo
      if(iflag.eq.0)then
        rat=8.d0
      elseif(iflag.eq.1)then
        rat=2.d0
      else
        write(6,*)'Error in checksij: iflag=',iflag
        write(6,*)' Must be 0 for soft, 1 for collinear'
        stop
      endif
c
      i=1
      dowhile(ckc(i).gt.0.1d0)
        i=i+1
      enddo
      imin=i
      do i=imin,imax-1
        if(ckc(i+1).gt.1.d-8)then
c If this condition is replaced by .eq.0, the test will fail if the series
c is made of elements all equal to the limit
          rckc(i)=ckc(i)/ckc(i+1)
        else
c Element #i+1 of series equal to the limit, so it must pass the test
          rckc(i)=rat*1.1d0
        endif
      enddo
      icount=0
      i=imin
      dowhile(icount.lt.ithrs.and.i.lt.imax)
        if(rckc(i).gt.rat)then
          icount=icount+1
        else
          icount=0
        endif
        i=i+1
      enddo
c
      if(icount.ne.ithrs)then
        iret=iret+itype
        if(istop.eq.1)then
          write(6,*)'Test failed',iflag
          write(6,*)'Event #',iev
          stop
        endif
      endif
      if(itype.eq.1.and.ki.eq.1.and.iflag.eq.0)then
        itype=2
        goto 100
      endif
c
      if(ki.eq.1.and.ilim.eq.1)then
        found=.false.
        i=0
        do while ((.not.found).and.i.lt.imax)
          i=i+1
          if(abs(check(i)-1.d0).gt.tolerance)then
            found=.true.
            itype=4
          endif
        enddo
        if(.not.found)then
          if(abs(checkl-1.d0).gt.tolerance)itype=4
        endif
        if(itype.eq.4)iret=iret+itype
      endif
c
      if( iwrite.eq.1 .and.
     #    iret.eq.1 .or.(iret.gt.1.and.ki.eq.1) )then
        if(iret.gt.7)then
          write(6,*)'Error in checksij: iret=',iret
          stop
        endif
        write(77,*)'    '
        if(iflag.eq.0)then
          write(77,*)'Soft #',iev
        elseif(iflag.eq.1)then
          write(77,*)'Collinear #',iev
        endif
        write(77,*)'iret:',iret
        write(77,*)'i_fks,j_fks:',i_fks,j_fks
        if(iret.eq.1.or.iret.eq.3.or.iret.eq.5.or.iret.eq.7)then
          write(77,*)'S_kl'
          write(77,*)'k,kk,ll',ki,kk,ll
          do i=1,imax
             call xprintout(77,xsijvc(i),xsijlvc)
          enddo
        endif
        if(iret.eq.2.or.iret.eq.3.or.iret.eq.6.or.iret.eq.7)then
          write(77,*)'sum of S'
          do i=1,imax
             call xprintout(77,xsumvc(i),xsumlvc)
          enddo
        endif
        if(iret.eq.4.or.iret.eq.5.or.iret.eq.6.or.iret.eq.7)then
          write(77,*)'check to one'
          do i=1,imax
             call xprintout(77,check(i),checkl)
          enddo
        endif
      endif
c
      if(ilim.eq.1)then
        if( abs(xsijlvc-xsijlim).gt.1.d-6 .and. 
     #    xsijlim.ne.-1.d0 )iret=iret+10
        if( abs(xsumlvc-xsumlim).gt.1.d-6 .and.
     #    xsumlim.ne.-1.d0 .and. iflag.eq.0)iret=iret+20
        if(iwrite.eq.1.and.iret.ge.10)then
          write(77,*)'    '
          if(iflag.eq.0)then
            write(77,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(77,*)'Collinear #',iev
          endif
          write(77,*)'iret:',iret
          write(77,*)'i_fks,j_fks:',i_fks,j_fks
          if((iret.ge.10.and.iret.lt.20).or.iret.ge.30)then
            write(77,*)'limit of S_kl'
            write(77,*)'k,kk,ll',ki,kk,ll
            write(77,*)xsijlvc,xsijlim
          endif
          if(iret.ge.20)then
            write(77,*)'limit of sum_j S_ij'
            write(77,*)xsumlvc,xsumlim
          endif
        endif
      endif
      return
      end


      subroutine bornsoftvirtual(p,bsv_wgt,born_wgt)
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "coupl.inc"
      include "fks.inc"
      include "run.inc"
      include "fks_powers.inc"
      include "q_es.inc"
      double precision p(0:3,nexternal),bsv_wgt,born_wgt
      double precision pp(0:3,nexternal)
      
      double complex wgt1(2)
      double precision rwgt,ao2pi,Q,Ej,wgt,contr,eikIreg,m1l_W_finite_CDR
      double precision shattmp,dot
      integer i,j,aj,m,n,k

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision xicut_used
      common /cxicut_used/xicut_used

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision pi
      parameter (pi=3.1415926535897932385d0)

      double precision c(0:1),gamma(0:1),gammap(0:1)
      common/fks_colors/c,gamma,gammap
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision double,single,xmu2
      logical ComputePoles,fksprefact
      parameter (ComputePoles=.false.)
      parameter (fksprefact=.true.)

      logical firsttime
      data firsttime/.true./

      double precision beta0
      common/cbeta0/beta0

c power of alphaS at the Born level. Computed from matrix elements here
      double precision bpower

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip

c For tests of virtuals
      double precision vobmax,vobmin
      common/cvirt0test/vobmax,vobmin
      double precision vNsumw,vAsumw,vSsumw,vNsumf,vAsumf,vSsumf
      common/cvirt1test/vNsumw,vAsumw,vSsumw,vNsumf,vAsumf,vSsumf
      integer nvtozero
      logical doVirtTest
      common/cvirt2test/nvtozero,doVirtTest
      double precision xnormsv
      common/cxnormsv/xnormsv
      double precision vrat

      double precision virt_wgt,ran2
      external ran2

      character*4 abrv
      common /to_abrv/ abrv

      logical ExceptPSpoint
      integer iminmax
      common/cExceptPSpoint/iminmax,ExceptPSpoint

c For the MINT folding
      integer fold
      common /cfl/fold
      double precision virt_wgt_save
      save virt_wgt_save

      double precision pmass(nexternal),zero,tiny
      parameter (zero=0d0)
      parameter (tiny=1d-6)
      include "pmass.inc"

      ao2pi=g**2/(8d0*PI**2)

      if (particle_type(i_fks).eq.8 .or. abrv.eq.'born') then

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
         shattmp=2d0*dot(p(0,1),p(0,2))
         if(abs(shattmp/shat-1.d0).gt.1.d-5)then
           write(*,*)'Error in sreal: inconsistent shat'
           write(*,*)shattmp,shat
           stop
         endif

         call sborn(p_born,wgt1)

c Born contribution:
         bsv_wgt=dble(wgt1(1))
         born_wgt=dble(wgt1(1))

c For the first non-zero phase-space point, compute bpower
         if (firsttime .and. born_wgt.ne.0d0) then
           firsttime=.false.

c Multiply the strong coupling by 10
           if (g.ne.0d0) then
             g=10d0*g
           else
             write(*,*)'Error in bornsoftvirtual'
             write(*,*)'Strong coupling is zero'
             stop
           endif

c Update alphaS-dependent couplings
           call setpara('param_card.dat',.false.)

c recompute the Born with the new couplings
           calculatedBorn=.false.
           call sborn(p_born,wgt1)

c Compute bpower
           bpower=Log10(dble(wgt1(1))/born_wgt)/2d0
           if(abs(bpower-dble(nint(bpower))) .gt. tiny) then
             write(*,*)'Error in computation of bpower:'
             write(*,*)' not an integer',bpower
             stop
           elseif (bpower.lt.-tiny) then
             write(*,*)'Error in computation of bpower:'
             write(*,*)' negative value',bpower
             stop
           else
c set it to the integer exactly
             bpower=dble(nint(bpower))
             write(*,*)'bpower is', bpower
           endif

c Change couplings back and recompute the Born to make sure that 
c nothing funny happens later on
           g=g/10d0
           call setpara('param_card.dat',.false.)
           calculatedBorn=.false.
           call sborn(p_born,wgt1)
         endif

         if (abrv.eq.'born') goto 549
         if (abrv.eq.'virt' .or. abrv.eq.'viSC' .or.
     #       abrv.eq.'viLC') goto 547

c Q contribution eq 5.5 and 5.6 of FKS
         Q=0d0
         do i=nincoming+1,nexternal
            if (i.ne.i_fks .and. particle_type(i).ne.1 .and. 
     #          pmass(i).eq.ZERO)then
               if (particle_type(i).eq.8) then
                  aj=0
               elseif(abs(particle_type(i)).eq.3) then
                  aj=1
               endif
               Ej=p(0,i)
               if(abrv.eq.'novA')then
c 2+3+4
                  Q = Q
     &             -2*dlog(shat/QES2)*dlog(xicut_used)*c(aj)
     &             -( dlog(deltaO/2d0)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &               +2*dlog(xicut_used)**2*c(aj) )
     &             +gammap(aj)
     &             +2d0*c(aj)*dlog(2d0*Ej/sqrtshat)**2
     &             -2d0*gamma(aj)*dlog(2d0*Ej/sqrtshat)
               elseif(abrv.eq.'novB')then
c 2+3+4_mu
                  Q = Q
     &             -2*dlog(shat/QES2)*dlog(xicut_used)*c(aj)
     &             -( dlog(deltaO/2d0)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &               +2*dlog(xicut_used)**2*c(aj) )
               elseif(abrv.eq.'viSA')then
c 1                
                  Q = Q
     &              -dlog(shat/QES2)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/sqrtshat) )
               elseif(abrv.eq.'viSB')then
c 1+4_L
                  Q = Q
     &              -dlog(shat/QES2)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/sqrtshat) )
     &             +gammap(aj)
     &             +2d0*c(aj)*dlog(2d0*Ej/sqrtshat)**2
     &             -2d0*gamma(aj)*dlog(2d0*Ej/sqrtshat)
               elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #                abrv.ne.'viLC')then
c 1+2+3+4
                  Q = Q+gammap(aj)
     &              -dlog(shat*deltaO/2d0/QES2)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &              +2d0*c(aj)*( dlog(2d0*Ej/sqrtshat)**2
     &              -dlog(xicut_used)**2 )
     &              -2d0*gamma(aj)*dlog(2d0*Ej/sqrtshat)
               else
                  write(*,*)'Error in bornsoftvirtual'
                  write(*,*)'abrv in Q:',abrv
                  stop
               endif
            endif
         enddo
c
         do i=1,nincoming
            if (particle_type(i).ne.1)then
               if (particle_type(i).eq.8) then
                  aj=0
               elseif(abs(particle_type(i)).eq.3) then
                  aj=1
               endif
               if(abrv.eq.'novA'.or.abrv.eq.'novB')then
c 2+3+4 or 2+3+4_mu
                  Q=Q-2*dlog(shat/QES2)*dlog(xicut_used)*c(aj)
     &               -dlog(q2fact(i)/shat)*(
     &                  gamma(aj)+2d0*c(aj)*dlog(xicut_used) )
               elseif(abrv.eq.'viSA'.or.abrv.eq.'viSB')then
c 1 or 1+4_L
                  Q=Q-dlog(shat/QES2)*gamma(aj)
               elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #                abrv.ne.'viLC')then
c 1+2+3+4
                  Q=Q-dlog(q2fact(i)/QES2)*(
     &                 gamma(aj)+2d0*c(aj)*dlog(xicut_used))
               else
                  write(*,*)'Error in bornsoftvirtual'
                  write(*,*)'abrv in Q:',abrv
                  stop
               endif
            endif
         enddo

         bsv_wgt=bsv_wgt+ao2pi*Q*dble(wgt1(1))

c        If doing MC over helicities, must sum over the two
c        helicity contributions for the Q-terms of collinear limit.
 547     continue
         if (abrv.eq.'virt' .or. abrv.eq.'viSC' .or.
     #       abrv.eq.'viLC') goto 548

c I(reg) terms, eq 5.5 of FKS
         contr=0d0
         do i=1,fks_j_from_i(i_fks,0)
            do j=1,i
               m=fks_j_from_i(i_fks,i)
               n=fks_j_from_i(i_fks,j)
               if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)).and.
     &              n.ne.i_fks.and.m.ne.i_fks) then
                  call sborn_sf(p_born,m,n,wgt)
                  if (wgt.ne.0d0) then
                     call eikonal_Ireg(p,m,n,xicut_used,eikIreg)
                     contr=contr+wgt*eikIreg
                  endif
               endif
            enddo
         enddo

C WARNING: THE FACTOR -2 BELOW COMPENSATES FOR THE MISSING -2 IN THE
C COLOUR LINKED BORN -- SEE ALSO SBORNSOFT().
C If the colour-linked Borns were normalized as reported in the paper
c we should set
c   bsv_wgt=bsv_wgt+ao2pi*contr  <-- DO NOT USE THIS LINE
c
         bsv_wgt=bsv_wgt-2*ao2pi*contr

 548     continue
c Finite part of one-loop corrections
c convert to Les Houches Accord standards
         if (ran2().le.1d0/virt_fraction .and. abrv(1:3).ne.'nov') then
            if (fold.eq.0) then
               Call LesHouches(p_born,born_wgt,virt_wgt)
               virt_wgt_save = virt_wgt
            elseif (fold.eq.1) then
               virt_wgt=virt_wgt_save
            else
               write (*,*) 'Error with fold (bornsoftvirtual)',fold
            endif
            bsv_wgt=bsv_wgt+virt_wgt*virt_fraction
         endif

c eq.(MadFKS.C.13)
         if(abrv.eq.'viSA'.or.abrv.eq.'viSB')then
           bsv_wgt=bsv_wgt + 2*pi*beta0*bpower*log(shat/QES2)*
     #                       ao2pi*dble(wgt1(1))
         elseif(abrv.eq.'novA'.or.abrv.eq.'novB')then
           bsv_wgt=bsv_wgt + 2*pi*beta0*bpower*log(q2fact(1)/shat)*
     #                       ao2pi*dble(wgt1(1))
         elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #          abrv.ne.'viLC')then
           bsv_wgt=bsv_wgt + 2*pi*beta0*bpower*log(q2fact(1)/QES2)*
     #                       ao2pi*dble(wgt1(1))
         endif
c eq.(MadFKS.C.14)
         if(abrv(1:2).ne.'vi')then
           bsv_wgt=bsv_wgt - 2*pi*beta0*bpower*log(q2fact(1)/scale**2)*
     #                       ao2pi*dble(wgt1(1))
         endif


 549     continue

         if (abrv(1:2).eq.'vi') then
            bsv_wgt=bsv_wgt-born_wgt

            if(doVirtTest .and. iminmax.eq.0)then
              if(born_wgt.ne.0.d0)then
                vrat=bsv_wgt/(ao2pi*born_wgt)
                if(vrat.gt.vobmax)vobmax=vrat
                if(vrat.lt.vobmin)vobmin=vrat
                vNsumw=vNsumw+xnormsv
                vAsumw=vAsumw+vrat*xnormsv
                vSsumw=vSsumw+vrat**2*xnormsv
                vNsumf=vNsumf+1.d0
                vAsumf=vAsumf+vrat
                vSsumf=vSsumf+vrat**2
              else
                if(bsv_wgt.ne.0.d0)nvtozero=nvtozero+1
              endif
            endif

            born_wgt=0d0
         endif


         if (ComputePoles) then
            call sborn(p_born,wgt1)
            born_wgt=dble(wgt1(1))

            print*,"           "
            write(*,123)((p(i,j),i=0,3),j=1,nexternal)
            xmu2=q2fact(1)
            call getpoles(p,xmu2,double,single,fksprefact)
            print*,"BORN",born_wgt!/conv
            print*,"DOUBLE",double/born_wgt/ao2pi
            print*,"SINGLE",single/born_wgt/ao2pi
c            print*,"LOOP",virt_wgt!/born_wgt/ao2pi*2d0
c            print*,"LOOP2",(virtcor+born_wgt*4d0/3d0-double*pi**2/6d0)
c            stop
 123        format(4(1x,d22.16))
         endif


      else
         bsv_wgt=0d0
         born_wgt=0d0
      endif

      return
      end


      subroutine eikonal_Ireg(p,m,n,xicut_used,eikIreg)
      implicit none
      double precision zero,pi,pi2
      parameter (zero=0.d0)
      parameter (pi=3.1415926535897932385d0)
      parameter (pi2=pi**2)
      include "nexternal.inc"
      include 'coupl.inc'
      include "q_es.inc"
      double precision p(0:3,nexternal),xicut_used,eikIreg
      integer m,n

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      character*4 abrv
      common /to_abrv/ abrv

      double precision Ei,Ej,kikj,rij,tmp,xmj,betaj,betai,xmi2,xmj2,
     # vij,xi0,alij,tHVvl,tHVv,arg1,arg2,arg3,arg4,xi1a,xj1a,
     # dot,ddilog
      external dot,ddilog

      double precision pmass(nexternal)
      include "pmass.inc"

      tmp=0.d0
      if(pmass(m).eq.0.d0.and.pmass(n).eq.0.d0)then
        if(m.eq.n)then
          write(*,*)'Error #2 in eikonal_Ireg',m,n
          stop
        endif
        Ei=p(0,n)
        Ej=p(0,m)
        kikj=dot(p(0,n),p(0,m))
        rij=kikj/(2*Ei*Ej)
        if(abs(rij-1.d0).gt.1.d-6)then
          if(abrv.eq.'novA')then
c 2+3+4
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2+
     #          2*dlog(xicut_used)*dlog(rij)-
     #          ddilog(rij)+1d0/2d0*dlog(rij)**2-
     #          dlog(1-rij)*dlog(rij)
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2+
     #          2*dlog(xicut_used)*dlog(rij)
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=1d0/2d0*dlog(shat/QES2)**2+
     #          dlog(shat/QES2)*dlog(rij)
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=1d0/2d0*dlog(shat/QES2)**2+
     #          dlog(shat/QES2)*dlog(rij)-
     #          ddilog(rij)+1d0/2d0*dlog(rij)**2-
     #          dlog(1-rij)*dlog(rij)
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            tmp=1d0/2d0*dlog(xicut_used**2*shat/QES2)**2+
     #          dlog(xicut_used**2*shat/QES2)*dlog(rij)-
     #          ddilog(rij)+1d0/2d0*dlog(rij)**2-
     #          dlog(1-rij)*dlog(rij)
          else
             write(*,*)'Error #11 in eikonal_Ireg',abrv
             stop
          endif
        else
          if(abrv.eq.'novA')then
c 2+3+4
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2-pi2/6.d0
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=1d0/2d0*dlog(shat/QES2)**2
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=1d0/2d0*dlog(shat/QES2)**2-pi2/6.d0
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            tmp=1d0/2d0*dlog(xicut_used**2*shat/QES2)**2-pi2/6.d0
          else
             write(*,*)'Error #12 in eikonal_Ireg',abrv
             stop
          endif
        endif
      elseif( (pmass(m).ne.0.d0.and.pmass(n).eq.0.d0) .or.
     #        (pmass(m).eq.0.d0.and.pmass(n).ne.0.d0) )then
        if(m.eq.n)then
          write(*,*)'Error #3 in eikonal_Ireg',m,n
          stop
        endif
        if(pmass(m).ne.0.d0.and.pmass(n).eq.0.d0)then
          Ei=p(0,n)
          Ej=p(0,m)
          xmj=pmass(m)
          betaj=sqrt(1-xmj**2/Ej**2)
        else
          Ei=p(0,m)
          Ej=p(0,n)
          xmj=pmass(n)
          betaj=sqrt(1-xmj**2/Ej**2)
        endif
        kikj=dot(p(0,n),p(0,m))
        rij=kikj/(2*Ei*Ej)

        if(abrv.eq.'novA')then
c 2+3+4
          tmp=dlog(xicut_used)*dlog(shat/QES2)+
     #        dlog(xicut_used)**2+
     #        2*dlog(xicut_used)*dlog(kikj/(xmj*Ei))-
     #        ddilog(1-(1+betaj)/(2*rij))+ddilog(1-2*rij/(1-betaj))+
     #        1/2.d0*log(2*rij/(1-betaj))**2-pi2/12.d0-
     #        1/4.d0*dlog((1+betaj)/(1-betaj))**2
        elseif(abrv.eq.'novB')then
c 2+3+4_mu
          tmp=dlog(xicut_used)*dlog(shat/QES2)+
     #        dlog(xicut_used)**2+
     #        2*dlog(xicut_used)*dlog(kikj/(xmj*Ei))
        elseif(abrv.eq.'viSA')then
c 1                
          tmp=1/4.d0*dlog(shat/QES2)**2+
     #        dlog(shat/QES2)*dlog(kikj/(xmj*Ei))
        elseif(abrv.eq.'viSB')then
c 1+4_L
          tmp=1/4.d0*dlog(shat/QES2)**2+
     #        dlog(shat/QES2)*dlog(kikj/(xmj*Ei))-
     #        ddilog(1-(1+betaj)/(2*rij))+ddilog(1-2*rij/(1-betaj))+
     #        1/2.d0*log(2*rij/(1-betaj))**2-pi2/12.d0-
     #        1/4.d0*dlog((1+betaj)/(1-betaj))**2
        elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #         abrv.ne.'viLC')then
c 1+2+3+4
          tmp=dlog(xicut_used)*( dlog(xicut_used*shat/QES2)+
     #                           2*dlog(kikj/(xmj*Ei)) )-
     #        ddilog(1-(1+betaj)/(2*rij))+ddilog(1-2*rij/(1-betaj))+
     #        1/2.d0*log(2*rij/(1-betaj))**2+
     #        dlog(shat/QES2)*dlog(kikj/(xmj*Ei))-pi2/12.d0+
     #        1/4.d0*dlog(shat/QES2)**2-
     #        1/4.d0*dlog((1+betaj)/(1-betaj))**2
        else
           write(*,*)'Error #13 in eikonal_Ireg',abrv
           stop
        endif
      elseif(pmass(m).ne.0.d0.and.pmass(n).ne.0.d0)then
        if(n.eq.m)then
          Ei=p(0,n)
          betai=sqrt(1-pmass(n)**2/Ei**2)
          if(abrv.eq.'novA')then
c 2+3+4
            tmp=2*dlog(xicut_used)-
     #          1/betai*dlog((1+betai)/(1-betai))
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=2*dlog(xicut_used)
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=dlog(shat/QES2)
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=dlog(shat/QES2)-
     #          1/betai*dlog((1+betai)/(1-betai))
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            tmp=dlog(xicut_used**2*shat/QES2)-
     #          1/betai*dlog((1+betai)/(1-betai))
          else
             write(*,*)'Error #14 in eikonal_Ireg',abrv
             stop
          endif
        else
          Ei=p(0,n)
          Ej=p(0,m)
          betai=sqrt(1-pmass(n)**2/Ei**2)
          betaj=sqrt(1-pmass(m)**2/Ej**2)
          xmi2=pmass(n)**2
          xmj2=pmass(m)**2
          kikj=dot(p(0,n),p(0,m))
          vij=sqrt(1-xmi2*xmj2/kikj**2)
          alij=kikj*(1+vij)/xmi2
          tHVvl=(alij**2*xmi2-xmj2)/2.d0
          tHVv=tHVvl/(alij*Ei-Ej)
          arg1=alij*Ei
          arg2=arg1*betai
          arg3=Ej
          arg4=arg3*betaj
          xi0=1/vij*log((1+vij)/(1-vij))
          xi1a=kikj**2*(1+vij)/xmi2*( xj1a(arg1,arg2,tHVv,tHVvl)-
     #                                xj1a(arg3,arg4,tHVv,tHVvl) )

          if(abrv.eq.'novA')then
c 2+3+4
            tmp=xi0*dlog(xicut_used)+1/2.d0*xi1a
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=xi0*dlog(xicut_used)
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=1/2.d0*xi0*dlog(shat/QES2)
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=1/2.d0*xi0*dlog(shat/QES2)+1/2.d0*xi1a
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            tmp=1/2.d0*xi0*dlog(xicut_used**2*shat/QES2)+1/2.d0*xi1a
          else
             write(*,*)'Error #15 in eikonal_Ireg',abrv
             stop
          endif
        endif
      else
        write(*,*)'Error #4 in eikonal_Ireg',m,n,pmass(m),pmass(n)
        stop
      endif
      eikIreg=tmp
      return
      end


      function xj1a(x,y,tHVv,tHVvl)
      implicit none
      real*8 xj1a,x,y,tHVv,tHVvl,ddilog
      external ddilog
c
      xj1a=1/(2*tHVvl)*( dlog((x-y)/(x+y))**2+4*ddilog(1-(x+y)/tHVv)+
     #                   4*ddilog(1-(x-y)/tHVv) )
      return
      end


      FUNCTION DDILOG(X)
*
* $Id: imp64.inc,v 1.1.1.1 1996/04/01 15:02:59 mclareni Exp $
*
* $Log: imp64.inc,v $
* Revision 1.1.1.1  1996/04/01 15:02:59  mclareni
* Mathlib gen
*
*
* imp64.inc
*
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION C(0:19)
      PARAMETER (Z1 = 1, HF = Z1/2)
      PARAMETER (PI = 3.14159 26535 89793 24D0)
      PARAMETER (PI3 = PI**2/3, PI6 = PI**2/6, PI12 = PI**2/12)
      DATA C( 0) / 0.42996 69356 08136 97D0/
      DATA C( 1) / 0.40975 98753 30771 05D0/
      DATA C( 2) /-0.01858 84366 50145 92D0/
      DATA C( 3) / 0.00145 75108 40622 68D0/
      DATA C( 4) /-0.00014 30418 44423 40D0/
      DATA C( 5) / 0.00001 58841 55418 80D0/
      DATA C( 6) /-0.00000 19078 49593 87D0/
      DATA C( 7) / 0.00000 02419 51808 54D0/
      DATA C( 8) /-0.00000 00319 33412 74D0/
      DATA C( 9) / 0.00000 00043 45450 63D0/
      DATA C(10) /-0.00000 00006 05784 80D0/
      DATA C(11) / 0.00000 00000 86120 98D0/
      DATA C(12) /-0.00000 00000 12443 32D0/
      DATA C(13) / 0.00000 00000 01822 56D0/
      DATA C(14) /-0.00000 00000 00270 07D0/
      DATA C(15) / 0.00000 00000 00040 42D0/
      DATA C(16) /-0.00000 00000 00006 10D0/
      DATA C(17) / 0.00000 00000 00000 93D0/
      DATA C(18) /-0.00000 00000 00000 14D0/
      DATA C(19) /+0.00000 00000 00000 02D0/
      IF(X .EQ. 1) THEN
       H=PI6
      ELSEIF(X .EQ. -1) THEN
       H=-PI12
      ELSE
       T=-X
       IF(T .LE. -2) THEN
        Y=-1/(1+T)
        S=1
        A=-PI3+HF*(LOG(-T)**2-LOG(1+1/T)**2)
       ELSEIF(T .LT. -1) THEN
        Y=-1-T
        S=-1
        A=LOG(-T)
        A=-PI6+A*(A+LOG(1+1/T))
       ELSE IF(T .LE. -HF) THEN
        Y=-(1+T)/T
        S=1
        A=LOG(-T)
        A=-PI6+A*(-HF*A+LOG(1+T))
       ELSE IF(T .LT. 0) THEN
        Y=-T/(1+T)
        S=-1
        A=HF*LOG(1+T)**2
       ELSE IF(T .LE. 1) THEN
        Y=T
        S=1
        A=0
       ELSE
        Y=1/T
        S=-1
        A=PI6+HF*LOG(T)**2
       ENDIF
       H=Y+Y-1
       ALFA=H+H
       B1=0
       B2=0
       DO 1 I = 19,0,-1
       B0=C(I)+ALFA*B1-B2
       B2=B1
    1  B1=B0
       H=-(S*(B0-H*B2)+A)
      ENDIF
      DDILOG=H
      RETURN
      END


      subroutine getpoles(p,xmu2,double,single,fksprefact)
c Returns the residues of double and single poles according to 
c eq.(B.1) and eq.(B.2) if fksprefact=.true.. When fksprefact=.false.,
c the prefactor (mu2/Q2)^ep in eq.(B.1) is expanded, and giving an
c extra contribution to the single pole
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "fks.inc"
      include 'coupl.inc'
      include "q_es.inc"
      double precision p(0:3,nexternal),xmu2,double,single
      logical fksprefact
      double precision c(0:1),gamma(0:1),gammap(0:1)
      common/fks_colors/c,gamma,gammap
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double complex wgt1(2)
      double precision born,wgt,kikj,dot,vij,aso2pi
      integer aj,i,j,m,n
      double precision pmass(nexternal),zero,pi
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0d0)
      include "pmass.inc"
c
      double=0.d0
      single=0.d0
c Born terms
      call sborn(p_born,wgt1)
      born=dble(wgt1(1))
      do i=1,nexternal
        if(i.ne.i_fks .and. particle_type(i).ne.1)then
          if (particle_type(i).eq.8) then
             aj=0
          elseif(abs(particle_type(i)).eq.3) then
             aj=1
          endif
          if(pmass(i).eq.ZERO)then
            double=double-c(aj)
            single=single-gamma(aj)
          else
            single=single-c(aj)
          endif
        endif
      enddo

      double=double*born
      single=single*born
c Colour-linked Born terms
      do i=1,fks_j_from_i(i_fks,0)
        do j=1,i
          m=fks_j_from_i(i_fks,i)
          n=fks_j_from_i(i_fks,j)
          if( m.ne.n .and. n.ne.i_fks .and. m.ne.i_fks )then
            call sborn_sf(p_born,m,n,wgt)
c The factor -2 compensate for that missing in sborn_sf
            wgt=-2*wgt
            if(wgt.ne.0.d0)then
              if(pmass(m).eq.zero.and.pmass(n).eq.zero)then
                kikj=dot(p(0,n),p(0,m))
                single=single+log(2*kikj/QES2)*wgt
              elseif(pmass(m).ne.zero.and.pmass(n).eq.zero)then
                single=single-0.5d0*log(pmass(m)**2/QES2)*wgt
                kikj=dot(p(0,n),p(0,m))
                single=single+log(2*kikj/QES2)*wgt
              elseif(pmass(m).eq.zero.and.pmass(n).ne.zero)then
                single=single-0.5d0*log(pmass(n)**2/QES2)*wgt
                kikj=dot(p(0,n),p(0,m))
                single=single+log(2*kikj/QES2)*wgt
              elseif(pmass(m).ne.zero.and.pmass(n).ne.zero)then
                kikj=dot(p(0,n),p(0,m))
                vij=sqrt(1-(pmass(n)*pmass(m)/kikj)**2)
                single=single+0.5d0*1/vij*log((1+vij)/(1-vij))*wgt
              else
                write(*,*)'Error in getpoles',i,j,n,m,pmass(n),pmass(m)
                stop
              endif
            endif
          endif
        enddo
      enddo
      aso2pi=g**2/(8*pi**2)
      double=double*aso2pi
      single=single*aso2pi
      if(.not.fksprefact)single=single+double*log(xmu2/QES2)
c
      return
      end


      function m1l_finite_CDR(p,born)
c Returns the finite part of virtual contribution, according to the
c definitions given in (B.1) and (B.2). This function must include
c the factor as/(2*pi)
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "fks.inc"
      include 'coupl.inc'
      include "q_es.inc"
      double precision p(0:3,nexternal),m1l_finite_CDR,born
      double precision CF,pi,aso2pi,shat,dot,xlgq2os
      parameter (CF=4d0/3d0)
      parameter (pi=3.1415926535897932385d0)
c
      aso2pi=g**2/(8*pi**2)
c This is relevant to e+e- --> qqbar
      shat=2d0*dot(p(0,1),p(0,2))
      xlgq2os=log(QES2/shat)
      m1l_finite_CDR=-aso2pi*CF*(xlgq2os**2+3*xlgq2os-pi**2+8.d0)*born
      return
      end


      function m1l_W_finite_CDR(p,born)
c Returns the finite part of virtual contribution, according to the
c definitions given in (B.1) and (B.2). This function must include
c the factor as/(2*pi)
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "fks.inc"
      include 'coupl.inc'
      include "q_es.inc"
      double precision p(0:3,nexternal),m1l_W_finite_CDR,born
      double precision CF,pi,aso2pi,shat,dot,xlgq2os
      parameter (CF=4d0/3d0)
      parameter (pi=3.1415926535897932385d0)
c
      aso2pi=g**2/(8*pi**2)
      shat=2d0*dot(p(0,1),p(0,2))
      xlgq2os=log(QES2/shat)

c This is relevant to qqbar -> W 
c$$$      m1l_W_finite_CDR=aso2pi*CF*(-xlgq2os**2-3d0*xlgq2os+pi**2-8d0)
c$$$      m1l_W_finite_CDR=m1l_W_finite_CDR*born

c This is relevant to gg -> H
      m1l_W_finite_CDR=aso2pi*(-3d0*xlgq2os**2+11d0+3d0*pi**2)
      m1l_W_finite_CDR=m1l_W_finite_CDR*born

      return
      end


      subroutine setfksfactor(iconfig)
      implicit none

      double precision CA,CF,Nf,PI
      parameter (CA=3d0,CF=4d0/3d0,Nf=5d0)
c$$$      parameter (CA=3d0,CF=4d0/3d0,Nf=4d0)
C SET NF=0 WHEN NOT CONSIDERING G->QQ SPLITTINGS. FOR TESTS ONLY
c$$$      parameter (CA=3d0,CF=4d0/3d0,Nf=0d0)
      parameter (pi=3.1415926535897932385d0)

      double precision c(0:1),gamma(0:1),gammap(0:1)
      common/fks_colors/c,gamma,gammap

      double precision beta0
      common/cbeta0/beta0

      logical softtest,colltest
      common/sctests/softtest,colltest

      integer config_fks,i,j,iconfig,fac1,fac2

      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks

      include 'coupl.inc'
      include 'genps.inc'
      include "nexternal.inc"
      include 'fks_powers.inc'
      include 'fks.inc'

      integer mapbconf(0:lmaxconfigs)
      integer b_from_r(lmaxconfigs)
      integer r_from_b(lmaxconfigs)
      include "bornfromreal.inc"
      integer            mapconfig(0:lmaxconfigs), this_config
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include "configs.inc"

      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision delta_used
      common /cdelta_used/delta_used
      double precision xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      logical xexternal
      common /toxexternal/ xexternal
      logical rotategranny
      common/crotategranny/rotategranny
      integer diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel

      logical multi_chan(lmaxconfigs)
      common /to_multi_chan/multi_chan

      character*1 integrate
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fac_i,fac_j,i_fks_pdg,j_fks_pdg

      character*13 filename

      character*4 abrv
      common /to_abrv/ abrv

      logical nbodyonly
      common/cnbodyonly/nbodyonly

      integer fold
      common /cfl/fold

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type


c First check if we need to integrate this directory
c The file "integrate.fks" should have been created by genint_fks
      open(unit=19,file="integrate.fks",status="old",err=99)
      read(19,'(a)') integrate
      if (integrate.eq.'N') then
         read(19,'(I2)') config_fks
         write (*,*) 'No need to integrate this directory...'
         write (*,*) 'Integrate directory number ',config_fks,' instead'
         stop
      elseif (integrate.eq.'Y') then
         write (*,*)
     &        'This directory should be included for symmetry reasons'
      else
         write (*,*) "Don't know what to do: ", integrate
         stop
      endif
      close(19)
c When doing nbodyonly, we should check that we need this dir or not
c The file "nbodyonly.fks" should have been created by genint_fks
      if (nbodyonly) then
         open(unit=19,file="nbodyonly.fks",status="old",err=99)
         read(19,'(a)') integrate
         if (integrate.eq.'N') then
            write (*,*) 'No need to integrate this directory when'//
     &           ' doing only the n-body integration'
            stop
         elseif (integrate.eq.'Y') then
            write (*,*) 'This directory should be included for n-body'
         else
            write (*,*) "Don't know what to do: ", integrate
            stop
         endif
         close(19)
      endif

c Check to see if this channel needs to be included in the multi-channeling
      diagramsymmetryfactor=0d0
      if (multi_channel) then
         if (onlyBorn) then
            open (unit=19,file="symfact.dat",status="old",err=12)
            do i=1,mapbconf(0)
               read (19,*,err=23) fac1,fac2
               if (fac2.gt.0) then
                  multi_chan(fac1)=.true.
               else
                  multi_chan(fac1)=.false.
               endif
            enddo
            if (multi_chan(b_from_r(iconfig))) then
               diagramsymmetryfactor=1d0
            else
               write (*,*) 'No need to integrate this channel'
               stop
            endif
            close(19)
         else                   ! not onlyBorn
            open (unit=19,file="symfact.dat",status="old",err=12)
            do i=1,mapbconf(0)
               read (19,*,err=23) fac1,fac2
               if (fac1.eq.b_from_r(iconfig)) then
                  if (fac2.gt.0) then
                     write (*,*) 'diagram symmetry factor',fac2
                     diagramsymmetryfactor=dble(fac2)
                  elseif(fac2.lt.0) then
                     write (*,*)
     &                    'diagram symmetry factor is negative', fac2
                     write (*,*)
     &                    'it is not needed to integrate this channel'
                     diagramsymmetryfactor=1d0
                  else
                     write (*,*) 'diagram symmetry factor is zero', fac2
                     stop
                  endif
               endif
            enddo
            if (diagramsymmetryfactor.eq.0d0) then
               write (*,*) 'error in diagramsymmetryfactor',iconfig
               stop
            endif
            close(19)
         endif
      
      else                      ! no multi_channel
         write (*,*) 'Setting diagram symmetry factor to 1,'//
     &        ' because no suppression.'
         diagramsymmetryfactor=1d0
      endif
 14    continue

c Set xexternal to true to use the x's from external vegas in the
c x_to_f_arg subroutine
      xexternal=.true.

c The value of rotategranny may be superseded later if phase space
c parametrization allows it
      rotategranny=.false.

      softtest=.false.
      colltest=.false.
      fold=0
      open (unit=19,file="config.fks",status="old")
      read (19,*) config_fks
      close (19)
      if (fks_j(config_fks).gt.nincoming)then
         delta_used=deltaO
      else
         delta_used=deltaI
      endif
      
      xicut_used=xicut
      xiScut_used=xiScut
      if( nbodyonly .or.
     #    (abrv.eq.'born' .or. abrv(1:2).eq.'vi') )then
        xiBSVcut_used=1.d0
      else
        xiBSVcut_used=xiBSVcut
      endif

      c(0)=CA
      c(1)=CF
      gamma(0)=( 11d0*CA-2d0*Nf )/6d0
      gamma(1)=CF*3d0/2d0
      gammap(0)=( 67d0/9d0 - 2d0*PI**2/3d0 )*CA - 23d0/18d0*Nf
      gammap(1)=( 13/2d0 - 2d0*PI**2/3d0 )*CF
            
c Beta_0 defined according to (MadFKS.C.5)
      beta0=gamma(0)/(2*pi)

c---------------------------------------------------------------------
c              Symmetry Factors
c---------------------------------------------------------------------
c fkssymmetryfactor:
c Calculate the FKS symmetry factors to be able to reduce the number
c of directories to (maximum) 4 (neglecting quark flavors):
c     1. i_fks=gluon, j_fks=gluon 
c     2. i_fks=gluon, j_fks=quark
c     3. i_fks=gluon, j_fks=anti-quark
c     4. i_fks=quark, j_fks=anti-quark (or vice versa).
c This sets the fkssymmetryfactor (in which the quark flavors are taken
c into account) for the subtracted reals.
c
c fkssymmetryfactorBorn:
c Note that in the Born's included here, the final state identical
c particle factor is set equal to the identical particle factor
c for the real contribution to be able to get the correct limits for the
c subtraction terms and the approximated real contributions.
c However when we want to calculate the Born contributions only, we
c have to correct for this difference. Since we only include the Born
c related to a soft limit (this uniquely defines the Born for a given real)
c the difference is always n!/(n-1)!=n, where n is the number of final state
c gluons in the real contribution.
c
c Furthermore, because we are not integrating all the directories, we also
c have to include a fkssymmetryfactor for the Born contributions. However,
c this factor is not the same as the factor defined above, because in this
c case i_fks is fixed to the extra gluon (which goes soft and defines the
c Born contribution) and should therefore not be taken into account when
c calculating the symmetry factor. Together with the factor n above this
c sets the fkssymmetryfactorBorn equal to the fkssymmetryfactor for the
c subtracted reals.
c
c We set fkssymmetryfactorBorn to zero when i_fks not a gluon
c
      fkssymmetryfactor=0d0
      fkssymmetryfactorDeg=0d0
      fkssymmetryfactorBorn=0d0

      i_fks=fks_i(config_fks)
      j_fks=fks_j(config_fks)
      if (i_fks.le.j_fks) then
         write (*,*) 'ERROR in setfksfactor, i_fks.le.j_fks: '//
     &        'terrible things might happen',i_fks,j_fks
         stop
      endif
      i_fks_pdg=pdg_type(i_fks)
      j_fks_pdg=pdg_type(j_fks)
      
      fac_i=0
      fac_j=0
      do i=nincoming+1,nexternal
         if (i_fks_pdg.eq.pdg_type(i)) fac_i = fac_i + 1
         if (j_fks_pdg.eq.pdg_type(i)) fac_j = fac_j + 1
      enddo
c Overwrite if initial state singularity
      if(j_fks.le.nincoming) fac_j=1

c i_fks and j_fks of the same type? -> subtract 1 to avoid double counting
      if (j_fks.gt.nincoming .and. i_fks_pdg.eq.j_fks_pdg) fac_j=fac_j-1

c THESE TESTS WORK ONLY FOR FINAL STATE SINGULARITIES
      if (j_fks.gt.nincoming) then
         if ( i_fks_pdg.eq.j_fks_pdg .and. i_fks_pdg.ne.21) then
            write (*,*) 'ERROR, if PDG type of i_fks and j_fks '//
     &           'are equal, they MUST be gluons',
     &           i_fks,j_fks,i_fks_pdg,j_fks_pdg
            stop
         elseif(abs(particle_type(i_fks)).eq.3) then
            if ( particle_type(i_fks).ne.-particle_type(j_fks) .or.
     &           pdg_type(i_fks).ne.-pdg_type(j_fks)) then
               write (*,*) 'ERROR, if i_fks is a color triplet, j_fks'//
     &              ' must be its anti-particle,'//
     &              ' or an initial state gluon.',
     &              i_fks,j_fks,particle_type(i_fks),
     &              particle_type(j_fks),pdg_type(i_fks),pdg_type(j_fks)
               stop
            endif
         elseif(i_fks_pdg.ne.21) then ! if not already above, it MUST be a gluon
            write (*,*) 'ERROR, i_fks is not a gluon and falls not'//
     &           ' in other categories',i_fks,j_fks,i_fks_pdg,j_fks_pdg
         endif
      endif

      ngluons=0
      do i=nincoming+1,nexternal
         if (pdg_type(i).eq.21) ngluons=ngluons+1
      enddo

      if (nbodyonly.and.i_fks_pdg.eq.21) then
         if (ngluons.le.0) then
            write (*,*)
     &           'ERROR, number of gluons should be larger than 1',
     &           ngluons
            stop
         endif
         fkssymmetryfactor=dble(ngluons)
         fkssymmetryfactorDeg=dble(ngluons)
         fkssymmetryfactorBorn=dble(ngluons)
         write (*,*) 'nbodyonly: fks symmetry factor has been put to ',
     &        fkssymmetryfactor
         write (*,*) 'nbodyonly, fks symmetry factor for Born has'//
     &        ' been put to ', fkssymmetryfactorBorn
      else
         fkssymmetryfactor=dble(fac_i*fac_j)
         fkssymmetryfactorDeg=dble(fac_i*fac_j)
         if (i_fks_pdg.eq.21) then
            fkssymmetryfactorBorn=dble(fac_i*fac_j)
         else
            fkssymmetryfactorBorn=0d0
         endif
         if (abrv.eq.'grid') then
            write (*,*) 'Setting grids using Born'
            fkssymmetryfactorBorn=1d0
            fkssymmetryfactor=0d0
            fkssymmetryfactorDeg=0d0
            abrv='born'
         endif
         write (*,*) 'fks symmetry factor is ', fkssymmetryfactor
         write (*,*) 'fks symmetry factor for Born is ',
     &        fkssymmetryfactorBorn
      endif

      if ((abrv.eq.'born' .or. abrv(1:2).eq.'vi') .and.
     &     fkssymmetryfactorBorn.eq.0d0) then
         write (*,*) 'Not needed to run this subprocess '//
     &        'because doing only Born or virtual'
      endif

c Set color types of i_fks, j_fks and fks_mother.
      i_type=particle_type(i_fks)
      j_type=particle_type(j_fks)
      if (abs(i_type).eq.abs(j_type)) then
         m_type=8
         if ( (j_fks.le.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &        (j_fks.gt.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
            write(*,*)'Flavour mismatch #1 in setfksfactor',
     &           i_fks,j_fks,i_type,j_type
            stop
         endif
      elseif(abs(i_type).eq.3 .and. j_type.eq.8)then
         if(j_fks.le.nincoming)then
            m_type=-i_type
         else
            write (*,*) 'Error in setfksfactor: (i,j)=(q,g)'
            stop
         endif
      elseif(i_type.eq.8 .and. abs(j_type).eq.3)then
         if (j_fks.le.nincoming) then
            m_type=j_type
         else
            m_type=j_type
         endif
      else
         write(*,*)'Flavour mismatch #2 in setfksfactor',
     &        i_type,j_type,m_type
         stop
      endif

      filename="contract.file"
      call LesHouchesInit(filename)

c Set matrices used by MC counterterms
      call set_mc_matrices

      return

 99   continue
      write (*,*) '"integrate.fks" or "nbodyonly.fks" not found.'
      write (*,*) 'make and run "genint_fks" first.'
      stop
 23   continue
      write (*,*) '"symfact.dat" is not of the correct format'
      stop
 12   continue
      diagramsymmetryfactor=1d0
      goto 14
      end


      subroutine get_helicity(i_fks,j_fks)
      implicit none
      include "nexternal.inc"
      include "born_nhel.inc"
      integer NHEL(nexternal,max_bhel*2),IHEL
chel  include "helicities.inc"
      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip
      integer hel_wgt,hel_wgt_born,hel_wgt_real
      integer nhelreal(nexternal,4),goodhelreal(4)
      integer nhelrealall(nexternal,max_bhel*2)
      common /c_nhelreal/ nhelreal,nhelrealall,goodhelreal,hel_wgt_real
      integer nhelborn(nexternal-1,2),goodhelborn(2)
      integer nhelbornall(nexternal-1,max_bhel)
      common /c_nhelborn/ nhelborn,nhelbornall,goodhelborn,hel_wgt_born

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel

      integer i,nexthel,j,i_fks,j_fks,ngood,k
      data nexthel /0/
      data ngood /0/
      logical done,firsttime,all_set,chckr
      data firsttime/.true./
      integer goodhelr(0:4,max_bhel/2),goodhelb(0:2,max_bhel/2)
      save goodhelr,goodhelb,all_set,chckr
      double precision rnd,ran2
      external ran2

      character*4 abrv
      common /to_abrv/ abrv
      logical Hevents
      common/SHevents/Hevents
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint

c Do not change these two lines, because ./bin/compile_madfks.sh might
c need to change them automatically
      logical HelSum
      parameter (HelSum=.true.)

c************
c goodhelr=2, real emission matrix element not yet calculated
c             for this helicity
c goodhelr=1, real emission matrix element calculated and non-zero
c goodhelr=0, real emission matrix element calculated and zero,
c             so can be skipped next time.
c************
      if (HelSum) return

      if (isum_hel.ne.0) then ! MC over helicities
c First, set the goodhelr and goodhelb to their starting values
      if (firsttime) then
         if ((mint.and. .not. Hevents) .or.
     &        (.not.mint .and.
     &                    (abrv.eq.'born' .or. abrv(1:2).eq.'vi'))) then
c           if computing only the Born diagrams, should not
c           consider real emission helicities            
            chckr=.false.
         else
            chckr=.true.
         endif
         skip=1
c read from file if possible
         open(unit=65,file='goodhel.dat',status='old',err=532)
         all_set=.true.
         do j=0,4
            read (65,*,err=532) (goodhelr(j,i),i=1,max_bhel/2)
         enddo
         do j=0,2
            read (65,*,err=532) (goodhelb(j,i),i=1,max_bhel/2)
         enddo
         read(65,*,err=532) hel_wgt
         hel_wgt_born=hel_wgt
         hel_wgt_real=hel_wgt
         do i=1,max_bhel/2
            if ((chckr .and.
     &           (goodhelb(0,i).eq.2 .or. goodhelr(0,i).eq.2)) .or.
     &           (.not.chckr.and.goodhelb(0,i).eq.2)) all_set=.false.
         enddo
         close(65)
         goto 533
c if file does not exist or has wrong format, set all to 2
 532     close(65)
         write (*,*) 'Good helicities not found in file'
         all_set=.false.
         do j=0,4
            do i=1,max_bhel/2
               goodhelr(j,i)=2
            enddo
         enddo
         do j=0,2
            do i=1,max_bhel/2
               goodhelb(j,i)=2
            enddo
         enddo
         hel_wgt=max_bhel/2
         hel_wgt_born=hel_wgt
         hel_wgt_real=hel_wgt
 533     continue
         firsttime=.false.
         goto 534 ! no previous event, so skip to the next helicity
      endif

c From previous event, check if there is an update
      if (.not.all_set) then
c real emission
         if(goodhelr(0,ngood).eq.2) then
            if ( goodhelreal(1).eq.0 .and.
     &           goodhelreal(2).eq.0 .and.
     &           goodhelreal(3).eq.0 .and.
     &           goodhelreal(4).eq.0 ) then
               do j=0,4
                  goodhelr(j,ngood)=0
               enddo
            elseif( goodhelreal(1).le.1 .and.
     &              goodhelreal(2).le.1 .and.
     &              goodhelreal(3).le.1 .and.
     &              goodhelreal(4).le.1 ) then
               goodhelr(0,ngood)=1
               do j=1,4
                  goodhelr(j,ngood)=goodhelreal(j)
               enddo
            elseif (.not.(goodhelreal(1).eq.2 .and.
     &                    goodhelreal(2).eq.2 .and.
     &                    goodhelreal(2).eq.2 .and.
     &                    goodhelreal(2).eq.2) ) then
               write (*,*) 'Error #2 in get_helicities',
     &              ngood,(goodhelr(j,ngood),j=0,4)
               stop
            endif
         endif
c Born and counter events
         if(goodhelb(0,ngood).eq.2) then
            if ( goodhelborn(1).eq.0 .and.
     &           goodhelborn(2).eq.0 ) then
               do j=0,2
                  goodhelb(j,ngood)=0
               enddo
            elseif( goodhelborn(1).le.1 .and.
     &              goodhelborn(2).le.1 ) then
               goodhelb(0,ngood)=1
               do j=1,2
                  goodhelb(j,ngood)=goodhelborn(j)
               enddo
            elseif (.not.(goodhelborn(1).eq.2 .and.
     &                    goodhelborn(2).eq.2) ) then
               write (*,*) 'Error #3 in get_helicities',
     &              nexthel,(goodhelb(j,ngood),j=0,2)
               stop
            endif
         endif

c Calculate new hel_wgt
         hel_wgt=0
         do i=1,max_bhel/2
            if((chckr .and.
     &           (goodhelb(0,i).ge.1.or.goodhelr(0,i).ge.1)) .or.
     &           (.not.chckr .and. goodhelb(0,i).ge.1)) then
               hel_wgt=hel_wgt+1
            endif
         enddo
         hel_wgt_born=hel_wgt
         hel_wgt_real=hel_wgt

c check if all have been set, if so -> write to file
         all_set=.true.
         do i=1,max_bhel/2
            if ((chckr .and.
     &           (goodhelb(0,i).eq.2 .or. goodhelr(0,i).eq.2)) .or.
     &           (.not.chckr.and.goodhelb(0,i).eq.2)) all_set=.false.
         enddo
         if (all_set) then
            write (*,*) 'All good helicities have been found.',hel_wgt
            open(unit=65,file='goodhel.dat',status='unknown')
            do j=0,4
               write (65,*) (goodhelr(j,i),i=1,max_bhel/2)
            enddo
            do j=0,2
               write (65,*) (goodhelb(j,i),i=1,max_bhel/2)
            enddo
            write(65,*) hel_wgt
            close(65)
         endif
      else
         do i=1,4
            if (goodhelr(i,ngood).ne.goodhelreal(i)) then
               write (*,*)'Error #4 in get_helicities',i,ngood
               stop
            endif
         enddo
         do i=1,2
            if (goodhelb(i,ngood).ne.goodhelborn(i)) then
               write (*,*)'Error #5 in get_helicities',i,ngood
               stop
            endif
         enddo
      endif

c Get the next helicity
 534  continue
      done=.false.
      do while (.not.done)
         if (nexthel.eq.max_bhel*2) nexthel=0
         nexthel=nexthel+1
         if(nhel(i_fks,nexthel).eq.1.and.nhel(j_fks,nexthel).eq.1) then
            if (ngood.eq.max_bhel/2) ngood=0
            ngood=ngood+1
            if((chckr .and.
     &           (goodhelr(0,ngood).ge.1.or.goodhelb(0,ngood).ge.1)).or.
     &           (.not.chckr .and. goodhelb(0,ngood).ge.1)) then
c Using random number to see if we have to go to the next.
c Probably this is an overkill, but have to make sure that there is
c no bias considering the *semi*-random numbers from VEGAS.
               rnd=ran2()
               if (rnd.le.1d0/dble(hel_wgt)) then
                  done=.true.
               endif
            endif
         endif
      enddo

      do i=1,nexternal
         if (i.eq.i_fks) then
            nhelreal(i,1)=1
            nhelreal(i,2)=1
            nhelreal(i,3)=-1
            nhelreal(i,4)=-1
         elseif (i.eq.j_fks) then
            nhelreal(i,1)=1
            nhelreal(i,2)=-1
            nhelreal(i,3)=1
            nhelreal(i,4)=-1
         else
            nhelreal(i,1)=nhel(i,nexthel)
            nhelreal(i,2)=nhel(i,nexthel)
            nhelreal(i,3)=nhel(i,nexthel)
            nhelreal(i,4)=nhel(i,nexthel)
         endif
      enddo
      do j=1,4
         goodhelreal(j)=goodhelr(j,ngood)
      enddo

      do i=1,nexternal-1
         if (i.eq.min(i_fks,j_fks)) then
            nhelborn(i,1)=1
            nhelborn(i,2)=-1
         elseif(i.lt.max(i_fks,j_fks)) then
            nhelborn(i,1)=nhel(i,nexthel)
            nhelborn(i,2)=nhel(i,nexthel)
         else
            nhelborn(i,1)=nhel(i+1,nexthel)
            nhelborn(i,2)=nhel(i+1,nexthel)
         endif
      enddo
      do j=1,2
         goodhelborn(j)=goodhelb(j,ngood)
      enddo

      else !isum_hel is zero, sum explicitly over helicities

      do i=1,nexternal
         do j=1,max_bhel*2
            nhelrealall(i,j)=nhel(i,j)
         enddo
      enddo
      do i=1,nexternal-1
         k=0
         do j=1,max_bhel*2
            if (nhel(i_fks,j).eq.-1) then
               k=k+1
               if (i.lt.i_fks) then
                  nhelbornall(i,k)=nhel(i,j)                  
               elseif(i.gt.i_fks) then
                  nhelbornall(i,k)=nhel(i+1,j)
               endif
            endif
         enddo
      enddo

      endif
      return
      end


      function get_ptrel(pp,i_fks,j_fks)
      implicit none
      include 'nexternal.inc'
      double precision get_ptrel,pp(0:3,nexternal)
      integer i_fks,j_fks
      double precision tmp,psum(3)
      integer i
c
      if(j_fks.le.2)then
        tmp=sqrt(pp(1,i_fks)**2+pp(2,i_fks)**2)
      else
        do i=1,3
          psum(i)=pp(i,i_fks)+pp(i,j_fks)
        enddo
        tmp=( pp(2,i_fks)*psum(1)-pp(1,i_fks)*psum(2) )**2+
     #      ( pp(3,i_fks)*psum(1)-pp(1,i_fks)*psum(3) )**2+
     #      ( pp(3,i_fks)*psum(2)-pp(2,i_fks)*psum(3) )**2
        if(tmp.ne.0.d0)tmp=sqrt( tmp/
     #       (psum(1)**2+psum(2)**2+psum(3)**2) )
      endif
      get_ptrel=tmp
      return
      end



      FUNCTION FK88RANDOM(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      REAL*8 MINV,FK88RANDOM
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      FK88RANDOM = SEED*MINV
      END

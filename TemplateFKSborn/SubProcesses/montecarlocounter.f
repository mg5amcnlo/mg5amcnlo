      subroutine set_mc_matrices
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "born_nhel.inc"
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
c Nexternal is the number of legs (initial and final) al NLO, while max_bcol
c is the number of color flows at Born level
      integer i,j,k,l,k0,mothercol(2),i1(2)
      include "born_leshouche.inc"
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fksfather
      common/cfksfather/fksfather
      logical notagluon,found
      common/cnotagluon/notagluon

      ipartners(0)=0
      do i=1,nexternal-1
         colorflow(i,0)=0
      enddo
c ipartners(0): number of colour partners of the father, the Born-level
c   particle to which i_fks and j_fks are attached
c ipartners(i), 1<=i<=nexternal-1: the label (according to Born-level
c   labelling) of the i^th colour partner of the father
c
c colourflow(i,0), 1<=i<=nexternal-1: number of colour flows in which
c   the particle ipartners(i) is a colour partner of the father
c colourflow(i,j): the actual label (according to born_leshouche.inc)
c   of the j^th colour flow in which the father and ipartners(i) are
c   colour partners
c
c Example: in the process q(1) qbar(2) --> g(3) g(4), the two color flows are
c
c i=1    j    icolup(1)    icolup(2)       i=2    j    icolup(1)    icolup(2)
c        1      500           0                   1      500           0
c        2       0           501                  2       0           501
c        3      500          502                  3      502          501
c        4      502          501                  4      500          502
c
c and if one fixes for example fksfather=3, then ipartners = 3
c while colorflow =  0  0  0                                 1
c                    1  1  0                                 4
c                    2  1  2                                 2
c                    1  2  0
c

      fksfather=min(i_fks,j_fks)

      do i=1,max_bcol
         mothercol(1)=ICOLUP(1,fksfather,i)
         mothercol(2)=ICOLUP(2,fksfather,i)
         notagluon=(mothercol(1).eq.0 .or. mothercol(2).eq.0)

         do j=1,nexternal-1
            if (j.ne.fksfather) then
               if ( (j.le.nincoming.and.fksfather.gt.nincoming) .or.
     #              (j.gt.nincoming.and.fksfather.le.nincoming) ) then
                  i1(1)=1
                  i1(2)=2
               else
                  i1(1)=2
                  i1(2)=1
               endif
               do l=1,2
                  found=.false.
                  if( ICOLUP(i1(l),j,i).eq.mothercol(l) .and.
     &                ICOLUP(i1(l),j,i).ne.0 ) then
                     k0=-1
                     do k=1,ipartners(0)
                        if(ipartners(k).eq.j)then
                           if(found)then
                              write(*,*)'Error #1 in set_matrices'
                              write(*,*)i,j,l,k
                              stop
                           endif
                           found=.true.
                           k0=k
                        endif
                     enddo
                     if (.not.found) then
                        ipartners(0)=ipartners(0)+1
                        ipartners(ipartners(0))=j
                        k0=ipartners(0)
c Icolup (i1(l),j,i).eq.mothercol(l) means that j and fksfather are 
c color-connected: if(found), a parton color-connected to the fksfather
c has already been found, so there is an error somewhere; else, the 
c vector of color partners has to be updated
                     endif
                     if(k0.le.0)then
                        write(*,*)'Error #2 in set_matrices'
                        write(*,*)i,j,l
                        stop
                     endif
                     colorflow(k0,0)=colorflow(k0,0)+1
                     colorflow(k0,colorflow(k0,0))=i
                     if (l.eq.2 .and. colorflow(k0,0).gt.1 .and.
     &                    colorflow(k0,colorflow(k0,0)-1).eq.i )then
                         if(notagluon)then
                            write(*,*)'Error #3 in set_matrices'
                            write(*,*)i,j,l,k0
                            stop
                         endif
                         colorflow(k0,0)=colorflow(k0,0)-1
                     endif
                  endif
               enddo
            endif
         enddo
      enddo
      call check_mc_matrices
      return
      end


      subroutine check_mc_matrices
      implicit none
      include "nexternal.inc"
      include "born_nhel.inc"
      include "fks.inc"
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      integer i,j,ipart,iflow,ntot,ithere(1000)
      integer fksfather
      common/cfksfather/fksfather
      logical notagluon
      common/cnotagluon/notagluon
c
      if(ipartners(0).gt.nexternal-1)then
        write(*,*)'Error #1 in check_mc_matrices',ipartners(0)
        stop
      endif
c
      do i=1,ipartners(0)
        ipart=ipartners(i)
        if( ipart.eq.fksfather .or.
     #      ipart.le.0 .or. ipart.gt.nexternal-1 .or.
     #      ( abs(particle_type(ipart)).ne.3 .and.
     #        particle_type(ipart).ne.8 ) )then
          write(*,*)'Error #2 in check_mc_matrices',i,ipart,
     #              particle_type(ipart)
          stop
        endif
      enddo
c
      do i=1,nexternal-1
        ithere(i)=1
      enddo
      do i=1,ipartners(0)
        ipart=ipartners(i)
        ithere(ipart)=ithere(ipart)-1
        if(ithere(ipart).lt.0)then
          write(*,*)'Error #3 in check_mc_matrices',i,ipart
          stop
        endif
      enddo
c
      ntot=0
      do i=1,ipartners(0)
        ntot=ntot+colorflow(i,0)
c
        if( colorflow(i,0).le.0 .or.
     #      colorflow(i,0).gt.max_bcol )then
          write(*,*)'Error #4 in check_mc_matrices',i,colorflow(i,0)
          stop
        endif
c
        do j=1,max_bcol
          ithere(j)=1
        enddo
        do j=1,colorflow(i,0)
          iflow=colorflow(i,j)
          ithere(iflow)=ithere(iflow)-1
          if(ithere(iflow).lt.0)then
            write(*,*)'Error #5 in check_mc_matrices',i,j,iflow
            stop
          endif
        enddo
c
      enddo
c
      if( (notagluon.and.ntot.ne.max_bcol) .or.
     #    ((.not.notagluon).and.ntot.ne.(2*max_bcol)) )then
         write(*,*)'Error #6 in check_mc_matrices',
     #     notagluon,ntot,max_bcol
         stop
       endif
c
      return
      end


      subroutine xmcsubt_wrap(pp,xi_i_fks,y_ij_fks,wgt)
      implicit none
      include "nexternal.inc"

      double precision pp(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
      double precision xmc,xrealme,gfactsf,gfactcl,probne
      double precision xmcxsec(nexternal),z
      integer nofpartners
      logical lzone(nexternal),flagmc

      call xmcsubt(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   xmc,nofpartners,lzone,flagmc,z,xmcxsec)
      call xmcsubtME(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,xrealme)

      wgt=xmc+xrealme

      return
      end


      subroutine xmcsubtME(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,wgt)
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      double precision pp(0:3,nexternal),gfactsf,gfactcl,
     & wgt,wgts,wgtc,wgtsc
      double precision xi_i_fks,y_ij_fks

      double precision zero,one
      parameter (zero=0d0)
      parameter (one=1d0)

      integer izero,ione,itwo
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision pmass(nexternal)
      include "pmass.inc"
c
      if (i_type.eq.8 .and. pmass(i_fks).eq.0d0)then
c i_fks is gluon
         call set_cms_stuff(izero)
         call sreal(p1_cnt(0,1,0),zero,y_ij_fks,wgts)
         call set_cms_stuff(ione)
         call sreal(p1_cnt(0,1,1),xi_i_fks,one,wgtc)
         call set_cms_stuff(itwo)
         call sreal(p1_cnt(0,1,2),zero,one,wgtsc)
         wgt=wgts+(1-gfactcl)*(wgtc-wgtsc)
         wgt=wgt*(1-gfactsf)
      elseif (abs(i_type).eq.3)then
c i_fks is (anti-)quark
         wgt=0d0
      else
         write(*,*) 'FATAL ERROR #1 in xmcsubtME',i_type,i_fks
         stop
      endif
c
      return
      end



      subroutine xmcsubt(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
c Wrapper for different Monte Carlo showers
      implicit none
      include "nexternal.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      double precision pp(0:3,nexternal),gfactsf,gfactcl,probne,wgt,
     &z(nexternal),xi_i_fks,y_ij_fks,xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc
      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo
c
      if(MonteCarlo.eq.'HERWIG6')then
         call xmcsubt_HW6(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
      elseif(MonteCarlo.eq.'HERWIGPP')then
         call xmcsubt_HWPP(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
      elseif(MonteCarlo.eq.'PYTHIA6Q')then
         call xmcsubt_PY6Q(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
      elseif(MonteCarlo.eq.'PYTHIA6PT')then
         call xmcsubt_PY6PT(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
      elseif(MonteCarlo.eq.'PYTHIA8')then
         call xmcsubt_PY8(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
      else
         write(*,*)'Monte Carlo type ',MonteCarlo,' not implemented'
         stop
      endif

      return
      end



      subroutine xmcsubt_HW6(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
c Main routine for MC counterterms
      implicit none
      include "nexternal.inc"
      include "fks.inc"
      include "coupl.inc"
      include "born_nhel.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      include "run.inc"

      double precision pp(0:3,nexternal),gfactsf,gfactcl,probne,wgt
      double precision xi_i_fks,y_ij_fks,xm12,xm22
      double precision xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc

      double precision emsca_bare,etot,ptresc,rrnd,ref_scale,
     & scalemin,scalemax,wgt1,ptHW6,emscainv,emscafun
      double precision emscwgt(nexternal),emscav(nexternal)
      integer jpartner,mpartner
      logical emscasharp

      double precision shattmp,dot,xkern,xkernazi,born_red,
     & born_red_tilde
      double precision bornbars(max_bcol), bornbarstilde(max_bcol)

      integer i,j,npartner,cflows,ileg,N_p
      double precision tk,uk,q1q,q2q,E0sq(nexternal),dE0sqdx(nexternal),
     # dE0sqdc(nexternal),x,yi,yj,xij,z(nexternal),xi(nexternal),
     # xjac(nexternal),zHW6,xiHW6,xjacHW6_xiztoxy,ap,Q,
     # beta,xfact,prefact,kn,knbar,kn0,betae0,betad,betas,
     # gfactazi,s,gfunsoft,gfuncoll,gfunazi,bogus_probne_fun,w1,w2

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow

      integer fksfather
      common/cfksfather/fksfather

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision emsca
      common/cemsca/emsca

      double precision ran2,iseed
      external ran2

      logical isr,fsr
      logical extra

c Stuff to be written (depending on AddInfoLHE) onto the LHE file
      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe

      double precision becl,delta
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision zero,one,tiny,vtiny,ymin
      parameter (zero=0d0)
      parameter (one=1d0)
      parameter (vtiny=1.d-10)
      parameter (ymin=0.9d0)

      double precision pi
      parameter(pi=3.1415926535897932384626433d0)

      real*8 vcf,vtf,vca
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)

      double precision pmass(nexternal)
      include "pmass.inc"
c

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
        write(*,*)'Error in xmcsubt_HW6: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

c May remove UseSudakov from the definition below if ptHW6 (or similar
c variable, not yet defined) will not be needed in the computation of probne
      extra=dampMCsubt.or.AddInfoLHE.or.UseSudakov
      call xiz_driver(xi_i_fks,y_ij_fks,shat,pp,ileg,
     #                xm12,xm22,tk,uk,q1q,q2q,ptHW6,extra)
      if(extra.and.ptHW6.lt.0.d0)then
        write(*,*)'Error in xmcsubt_HW6: ptHW6=',ptHW6
        stop
      endif
      call get_mbar(pp,y_ij_fks,ileg,bornbars,bornbarstilde)

      etot=2d0*sqrt( ebeam(1)*ebeam(2) )
      emsca=etot
      if(dampMCsubt)then
        emsca=0.d0
        ref_scale=sqrt( (1-xi_i_fks)*shat )
        scalemin=max(frac_low*ref_scale,scaleMClow)
        scalemax=max(frac_upp*ref_scale,scalemin+scaleMCdelta)
        emscasharp=(scalemax-scalemin).lt.(0.001d0*scalemax)
        if(emscasharp)then
          emsca_bare=scalemax
        else
          rrnd=ran2()
          rrnd=emscainv(rrnd,one)
          emsca_bare=scalemin+rrnd*(scalemax-scalemin)
        endif
      endif

c Distinguish initial or final state radiation
      isr=.false.
      fsr=.false.
      if(ileg.le.2)then
        isr=.true.
        delta=min(1.d0,deltaI)
      elseif(ileg.eq.3.or.ileg.eq.4)then
        fsr=.true.
        delta=min(1.d0,deltaO)
      else
        write(*,*)'Error in xmcsubt_HW6: unknown ileg'
        write(*,*)ileg
        stop
      endif

c Assign fks variables
      x=1-xi_i_fks
      if(isr)then
         yj=0.d0
         yi=y_ij_fks
      elseif(fsr)then
         yj=y_ij_fks
         yi=0.d0
      else
         write(*,*)'Error in xmcsubt_HW6: isr and fsr both false'
         stop
      endif

c$$$CHECK: WHO IS SHAT HERE?
      s = shat
      xij=2*(1-xm12/shat-(1-x))/(2-(1-x)*(1-yj)) 
c
      if (abs(i_type).eq.3) then
         gfactsf=1d0
      else
         gfactsf=gfunsoft(x,s,zero,alsf,besf)
      endif
      becl=-(1.d0-ymin)
      gfactcl=gfuncoll(y_ij_fks,alsf,becl,one)
      gfactazi=gfunazi(y_ij_fks,alazi,beazi,delta)
c
c Non-emission probability. When UseSudakov=.true., the definition of
c probne may be moved later if necessary
      if(.not.UseSudakov)then
c this is standard MC@NLO
        probne=1.d0
      else
        probne=bogus_probne_fun(ptHW6)
      endif
c
      wgt=0.d0
      nofpartners=ipartners(0)
      flagmc=.false.
c
      do npartner=1,ipartners(0)
c This loop corresponds to the sum over colour lines l in the
c xmcsubt note
        E0sq(npartner)=
     &   dot(p_born(0,fksfather),p_born(0,ipartners(npartner)))
        dE0sqdx(npartner)=0.d0
        dE0sqdc(npartner)=0.d0
c With the new parametrization E0sq doesn't depend on fks variables
        if(E0sq(npartner).gt.0.d0)then
c If E0=0 the configuration is in the dead zone. This if prevents
c numerical instabilities in the calculation of shower variables
          z(npartner)=zHW6(ileg,E0sq(npartner),xm12,xm22,shat,x,yi,yj,
     &                      tk,uk,q1q,q2q)
          xi(npartner)=xiHW6(ileg,E0sq(npartner),xm12,xm22,shat,x,yi,yj,
     &                        tk,uk,q1q,q2q)
          xjac(npartner)=xjacHW6_xiztoxy(ileg,E0sq(npartner),
     &                             dE0sqdx(npartner),dE0sqdc(npartner),
     &                             xm12,xm22,shat,x,yi,yj,tk,uk,q1q,q2q)
c
c Compute deadzones
          lzone(npartner)=.false.
          if(ileg.le.2)then
            if(z(npartner).ge.0.d0.and.xi(npartner).ge.0.d0
     &          .and.z(npartner)**2.ge.xi(npartner))lzone(npartner)=.true.
          elseif(ileg.eq.3)then
            if(z(npartner).ge.0.d0.and.xi(npartner).ge.0.d0
     &          .and.E0sq(npartner)*xi(npartner)*z(npartner)**2.ge.xm12 
     &          .and.xi(npartner).le.1.d0)lzone(npartner)=.true.
          elseif(ileg.eq.4)then
            if(z(npartner).ge.0.d0.and.xi(npartner).ge.0.d0
     &          .and.xi(npartner).le.1.d0)lzone(npartner)=.true.
          else
            write(*,*)'Error 1 in xmcsubt_HW6: unknown ileg'
            write(*,*)ileg
            stop
          endif
        elseif(E0sq(npartner).eq.0.d0)then
          lzone(npartner)=.false.
        else
          write(*,*)'Error in xmcsubt_HW6: negative scale',
     &      E0sq(npartner),ileg,npartner
          stop
        endif
c
c Compute MC subtraction terms
        if(lzone(npartner))then
          if(.not.flagmc)flagmc=.true.
          if( (fsr .and. m_type.eq.8) .or.
     #        (isr .and. j_type.eq.8) )then
            if(i_type.eq.8)then
c g --> g g (icode=1) and go --> go g (SUSY) splitting 
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=2
                 if(1-x.lt.tiny)then
                    xkern=(g**2/N_p)*64*vca*E0sq(npartner)/
     &                    (s*(s*(1-yi)+4*E0sq(npartner)*(1+yi)))
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*8*vca*(1-x*(1-x))**2/(s*x**2)
                    xkernazi=-(g**2/N_p)*16*vca*(1-x)**2/(s*x**2)
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    call Qterms_reduced_spacelike(m_type,i_type,one,
     #                                            z(npartner),Q)
                    Q=Q/(1-z(npartner))
                    xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                 endif
              elseif(ileg.eq.3)then
c Works only for SUSY
                N_p=2
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced_SUSY(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                N_p=2
                if(1-x.lt.tiny)then
                  xkern=(g**2/N_p)*64*vca*E0sq(npartner)/
     &                  ( s*(s*(1-yj)+4*E0sq(npartner)*(1+yj))-
     &                    xm12*(2*s-xm12)*(1-yj) )
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*( 8*vca*
     &                  (s**2*(1-(1-x)*x)-s*(1+x)*xm12+xm12**2)**2 )/
     &                  ( s*(s-xm12)**2*(s*x-xm12)**2 )
                  xkernazi=-(g**2/N_p)*(16*vca*s*(1-x)**2)/((s-xm12)**2)
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  call Qterms_reduced_timelike(j_type,i_type,one,z(npartner),Q)
                  Q=Q/(1-z(npartner))
                  xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                endif
              else
                write(*,*)'Error 4 in xmcsubt_HW6: forbidden ileg'
                write(*,*)ileg
                stop
              endif
            elseif(abs(i_type).eq.3)then
c g --> q qbar splitting (icode=2)
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=1
                 if(1-x.lt.tiny)then
                    xkern=0.d0
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*4*vtf*(1-x)*((1-x)**2+x**2)/(s*x)
                    xkernazi=0.d0
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    xkernazi=0.d0
                 endif
              elseif(ileg.eq.4)then
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                N_p=2
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*( 4*vtf*(1-x)*
     &                  (s**2*(1-2*(1-x)*x)-2*s*x*xm12+xm12**2) )/
     &                  ( (s-xm12)**2*(s*x-xm12) )
                  xkernazi=(g**2/N_p)*(16*vtf*s*(1-x)**2)/((s-xm12)**2)
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  call Qterms_reduced_timelike(j_type,i_type,one,z(npartner),Q)
                  Q=Q/(1-z(npartner))
                  xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                endif
              else
                write(*,*)'Error 5 in xmcsubt_HW6: forbidden ileg'
                write(*,*)ileg
                stop
              endif
            else
              write(*,*)'Error 3 in xmcsubt_HW6: unknown particle type'
              write(*,*)i_type
              stop
            endif
          elseif( (fsr .and. abs(m_type).eq.3) .or.
     #            (isr .and. abs(j_type).eq.3) )then
            if(abs(i_type).eq.3)then
c q --> g q (or qbar --> g qbar) splitting (icode=3)
c the fks parton is the one associated with 1 - z: this is because its
c rescaled energy is 1 - x and in the soft limit, where x --> z --> 1,
c it has to coincide with the fraction appearing in the AP kernel
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=2
                 if(1-x.lt.tiny)then
                    xkern=0.d0
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*4*vcf*(1-x)*((1-x)**2+1)/(s*x**2)
                    xkernazi=-(g**2/N_p)*16*vcf*(1-x)**2/(s*x**2)
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    call Qterms_reduced_spacelike(m_type,i_type,one,
     #                                            z(npartner),Q)
                    Q=Q/(1-z(npartner))
                    xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                 endif
              elseif(ileg.eq.3)then
                N_p=1
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
                N_p=1
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*
     &                  ( 4*vcf*(1-x)*(s**2*(1-x)**2+(s-xm12)**2) )/
     &                  ( (s-xm12)*(s*x-xm12)**2 )
                  xkernazi=0.d0
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              else
                write(*,*)'Error 6 in xmcsubt_HW6: unknown ileg'
                write(*,*)ileg
                stop              
              endif
            elseif(i_type.eq.8)then
c q --> q g splitting (icode=4) and sq --> sq g (SUSY) splitting
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=1
                 if(1-x.lt.tiny)then
                    xkern=(g**2/N_p)*64*vcf*E0sq(npartner)/
     &                    (s*(s*(1-yi)+4*E0sq(npartner)*(1+yi)))
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*4*vcf*(1+x**2)/(s*x)
                    xkernazi=0.d0
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    xkernazi=0.d0
                 endif
              elseif(ileg.eq.3)then
                N_p=1
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
                  betae0=sqrt(1-xm12/E0sq(npartner))
                  betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
                  betas=1+(xm12-xm22)/s
                  xkern=(g**2/N_p)*(32*betae0*(1+betae0)*betad*(1-yj)*
     &                     E0sq(npartner)*vcf*(s+betad*s+xm12-xm22)) /
     &                  ( s*(betas-betad*yj)*( (s+betad*s+xm12-xm22)*
     &                    (-4*xm12+(s+xm12-xm22)*(betas-betad*yj)) + 
     &                    4*(1+betae0)*E0sq(npartner)*
     &                    (xm12-xm22+s*(1+betad-betas+betad*yj)) ) )
                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  if(abs(PDG_type(j_fks)).le.6)then
c QCD branching
                    call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  else
c Non-QCD branching, here taken to be squark->squark gluon 
                    call AP_reduced_SUSY(j_type,i_type,one,z(npartner),ap)
                  endif
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
                N_p=1
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                if(1-x.lt.tiny)then
                  xkern=(g**2/N_p)*64*vcf*E0sq(npartner)/
     &                  ( s*(s*(1-yj)+4*E0sq(npartner)*(1+yj))-
     &                    xm12*(2*s-xm12)*(1-yj) )
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*4*vcf*
     &                  ( s**2*(1+x**2)-2*xm12*(s*(1+x)-xm12) )/
     &                  ( s*(s-xm12)*(s*x-xm12) )
                  xkernazi=0.d0
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              else
                write(*,*)'Error 7 in xmcsubt_HW6: unknown ileg'
                write(*,*)ileg
                stop
              endif
            else
               write(*,*)'Error 8 in xmcsubt_HW6: unknown particle type'
               write(*,*)i_type
               stop
            endif
          else
            write(*,*)'Error 2 in xmcsubt_HW6: unknown particle type'
            write(*,*)j_type,i_type
            stop
          endif
c
          if(dampMCsubt)then
            if(emscasharp)then
              if(ptHW6.le.scalemax)then
                emscwgt(npartner)=1.d0
                emscav(npartner)=emsca_bare
              else
                emscwgt(npartner)=0.d0
                emscav(npartner)=scalemax
              endif
            else
              ptresc=(ptHW6-scalemin)/(scalemax-scalemin)
              if(ptresc.le.0.d0)then
                emscwgt(npartner)=1.d0
                emscav(npartner)=emsca_bare
              elseif(ptresc.lt.1.d0)then
                emscwgt(npartner)=1-emscafun(ptresc,one)
                emscav(npartner)=emsca_bare
              else
                emscwgt(npartner)=0.d0
                emscav(npartner)=scalemax
              endif
            endif
          endif
c
        else
c Dead zone
          xkern=0.d0
          xkernazi=0.d0
          if(dampMCsubt)then
            emscav(npartner)=etot
            emscwgt(npartner)=0.d0
          endif
        endif
        xkern=xkern*gfactsf
        xkernazi=xkernazi*gfactazi*gfactsf
        born_red=0.d0
        born_red_tilde=0.d0
        do cflows=1,colorflow(npartner,0)
c In the case of MC over colour flows, cflows will be passed from outside
          born_red=born_red+
     #             bornbars(colorflow(npartner,cflows))
          born_red_tilde=born_red_tilde+
     #                   bornbarstilde(colorflow(npartner,cflows))
        enddo
        xmcxsec(npartner) = xkern*born_red + xkernazi*born_red_tilde
        if(dampMCsubt)
     #    xmcxsec(npartner)=xmcxsec(npartner)*emscwgt(npartner)
        wgt = wgt + xmcxsec(npartner)
c
        if(xmcxsec(npartner).lt.0.d0)then
           write(*,*) 'Fatal error in xmcsubt_HW6',
     #                npartner,xmcxsec(npartner)
           do i=1,nexternal
              write(*,*) 'particle ',i,', ',(pp(j,i),j=0,3)
           enddo
           stop
        endif
c End of loop over colour partners
      enddo
c Assign emsca on statistical basis
      if(extra.and.wgt.gt.1.d-30)then
        rrnd=ran2()
        wgt1=0.d0
        jpartner=0
        do npartner=1,ipartners(0)
          if(lzone(npartner).and.jpartner.eq.0)then
            wgt1 = wgt1 + xmcxsec(npartner)
            if(wgt1.ge.rrnd*wgt)then
              jpartner=ipartners(npartner)
              mpartner=npartner
            endif
          endif
        enddo
c
        if(jpartner.eq.0)then
          write(*,*)'Error in xmcsubt_HW6: emsca unweighting failed'
          stop
        else
          emsca=emscav(mpartner)
        endif
      endif
      if(dampMCsubt.and.wgt.lt.1.d-30)emsca=etot
c Additional information for LHE
      if(AddInfoLHE)then
        fksfather_lhe=fksfather
        if(jpartner.ne.0)then
          ipartner_lhe=jpartner
        else
c min() avoids troubles if ran2()=1
          ipartner_lhe=min( int(ran2()*ipartners(0))+1,ipartners(0) )
          ipartner_lhe=ipartners(ipartner_lhe)
        endif
        scale1_lhe=ptHW6
      endif
c
      if(dampMCsubt)then
        if(emsca.lt.scalemin)then
          write(*,*)'Error in xmcsubt_HW6: emsca too small',emsca,jpartner
          if(.not.lzone(npartner))then
            write(*,*)'because configuration in dead zone '
          else 
            stop
          endif
        endif
      endif
c
      do npartner=1,ipartners(0)
        xmcxsec(npartner) = xmcxsec(npartner) * probne
      enddo
      do npartner=ipartners(0)+1,nexternal
        xmcxsec(npartner) = 0.d0
      enddo
c No need to multiply this weight by probne, because it is ignored in 
c normal running. When doing the testing (test_MC), also the other
c pieces are not multiplied by probne.
c$$$      wgt=wgt*probne
c
      return
      end





      subroutine xmcsubt_HWPP(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
c Main routine for MC counterterms
      implicit none
      include "nexternal.inc"
      include "fks.inc"
      include "coupl.inc"
      include "born_nhel.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      include "run.inc"

      double precision pp(0:3,nexternal),gfactsf,gfactcl,probne,wgt
      double precision xi_i_fks,y_ij_fks,xm12,xm22
      double precision xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc

      double precision emsca_bare,etot,ptresc,rrnd,ref_scale,
     & scalemin,scalemax,wgt1,ptHW6,emscainv,emscafun
      double precision emscwgt(nexternal),emscav(nexternal)
      integer jpartner,mpartner
      logical emscasharp

      double precision shattmp,dot,xkern,xkernazi,born_red,
     & born_red_tilde
      double precision bornbars(max_bcol), bornbarstilde(max_bcol)

      integer i,j,npartner,cflows,ileg,N_p
      double precision tk,uk,q1q,q2q,E0sq(nexternal),dE0sqdx(nexternal),
     # dE0sqdc(nexternal),x,yi,yj,xij,z(nexternal),xi(nexternal),
     # xjac(nexternal),zHW6,xiHW6,xjacHW6_xiztoxy,ap,Q,
     # beta,xfact,prefact,kn,knbar,kn0,kn_diff,betae0,betad,betas,
     # gfactazi,s,gfunsoft,gfuncoll,gfunazi,bogus_probne_fun,w1,w2

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow

      integer fksfather
      common/cfksfather/fksfather

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision emsca
      common/cemsca/emsca

      double precision ran2,iseed
      external ran2

      logical isr,fsr
      logical extra

c Stuff to be written (depending on AddInfoLHE) onto the LHE file
      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe

      double precision becl,delta
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision zero,one,tiny,vtiny,ymin
      parameter (zero=0d0)
      parameter (one=1d0)
      parameter (vtiny=1.d-10)
      parameter (ymin=0.9d0)

      double precision pi
      parameter(pi=3.1415926535897932384626433d0)

      real*8 vcf,vtf,vca
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)

      double precision pmass(nexternal)
      include "pmass.inc"


      write(*,*)'HW++ not yet implemented!'
      stop


      return
      end




      subroutine xmcsubt_PY6Q(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
c Main routine for MC counterterms
      implicit none
      include "nexternal.inc"
      include "fks.inc"
      include "coupl.inc"
      include "born_nhel.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      include "run.inc"

      double precision pp(0:3,nexternal),gfactsf,gfactcl,probne,wgt
      double precision xi_i_fks,y_ij_fks,xm12,xm22
      double precision xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc

      double precision emsca_bare,etot,ptresc,rrnd,ref_scale,
     & scalemin,scalemax,wgt1,tPY6Q,emscainv,emscafun
      double precision emscwgt(nexternal),emscav(nexternal)
      integer jpartner,mpartner
      logical emscasharp

      double precision shattmp,dot,xkern,xkernazi,born_red,
     & born_red_tilde
      double precision bornbars(max_bcol), bornbarstilde(max_bcol)

      integer i,j,npartner,cflows,ileg,N_p
      double precision tk,uk,q1q,q2q,E0sq(nexternal),dE0sqdx(nexternal),
     # dE0sqdc(nexternal),x,yi,yj,xij,z(nexternal),xi(nexternal),
     # xjac(nexternal),zPY6Q,xiPY6Q,xjacPY6Q_xiztoxy,ap,Q,
     # beta,xfact,prefact,kn,knbar,kn0,kn_diff,beta1,betad,betas,
     # gfactazi,s,gfunsoft,gfuncoll,gfunazi,bogus_probne_fun,
     # ztmp,xitmp,xjactmp,get_angle,w1,w2,
     # p_born_npartner(0:3),p_born_fksfather(0:3)

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks


      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow

      integer fksfather
      common/cfksfather/fksfather

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision emsca
      common/cemsca/emsca

      double precision ran2,iseed
      external ran2

      logical isr,fsr
      logical extra

c Stuff to be written (depending on AddInfoLHE) onto the LHE file
      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe

      double precision becl,delta
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision zero,one,tiny,vtiny,ymin
      parameter (zero=0d0)
      parameter (one=1d0)
      parameter (vtiny=1.d-10)
      parameter (ymin=0.9d0)

      double precision pi
      parameter(pi=3.1415926535897932384626433d0)

      real*8 vcf,vtf,vca
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)

      integer mstj50,mstp67,en_fks,en_mother,theta2,theta2_cc
      double precision upper_scale(nexternal-1),fff(nexternal-1)
c
      double precision pmass(nexternal)
      include "pmass.inc"
c

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
        write(*,*)'Error in xmcsubt_PY6Q: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

c May remove UseSudakov from the definition below if tPY6Q (or similar
c variable, not yet defined) will not be needed in the computation of probne
      extra=dampMCsubt.or.AddInfoLHE.or.UseSudakov
      call xiz_driver(xi_i_fks,y_ij_fks,shat,pp,ileg,
     &                   xm12,xm22,tk,uk,q1q,q2q,tPY6Q,extra)
      if(extra.and.tPY6Q.lt.0.d0)then
        write(*,*)'Error in xmcsubt_PY6Q: tPY6Q=',tPY6Q
        stop
      endif
      call get_mbar(pp,y_ij_fks,ileg,bornbars,bornbarstilde)

      etot=2d0*sqrt( ebeam(1)*ebeam(2) )
      emsca=etot
      if(dampMCsubt)then
        emsca=0.d0
        ref_scale=sqrt( (1-xi_i_fks)*shat )
        scalemin=max(frac_low*ref_scale,scaleMClow)
        scalemax=max(frac_upp*ref_scale,scalemin+scaleMCdelta)
        emscasharp=(scalemax-scalemin).lt.(0.001d0*scalemax)
        if(emscasharp)then
          emsca_bare=scalemax
        else
          rrnd=ran2()
          rrnd=emscainv(rrnd,one)
          emsca_bare=scalemin+rrnd*(scalemax-scalemin)
        endif
      endif

c Distinguish initial or final state radiation
      isr=.false.
      fsr=.false.
      if(ileg.le.2)then
        isr=.true.
        delta=min(1.d0,deltaI)
      elseif(ileg.eq.3.or.ileg.eq.4)then
        fsr=.true.
        delta=min(1.d0,deltaO)
      else
        write(*,*)'Error in xmcsubt_PY6Q: unknown ileg'
        write(*,*)ileg
        stop
      endif

c Assign fks variables
      x=1-xi_i_fks
      if(isr)then
         yj=0.d0
         yi=y_ij_fks
      elseif(fsr)then
         yj=y_ij_fks
         yi=0.d0
      else
         write(*,*)'Error in xmcsubt_PY6Q: isr and fsr both false'
         stop
      endif

      s = shat
      xij=2*(1-xm12/shat-(1-x))/(2-(1-x)*(1-yj)) 
c
      if (abs(i_type).eq.3) then
         gfactsf=1d0
      else
         gfactsf=gfunsoft(x,s,zero,alsf,besf)
      endif
      becl=-(1.d0-ymin)
      gfactcl=gfuncoll(y_ij_fks,alsf,becl,one)
      gfactazi=gfunazi(y_ij_fks,alazi,beazi,delta)
c
c Non-emission probability. When UseSudakov=.true., the definition of
c probne may be moved later if necessary
      if(.not.UseSudakov)then
c this is standard MC@NLO
        probne=1.d0
      else
        probne=bogus_probne_fun(tPY6Q)
      endif
c
      wgt=0.d0
      nofpartners=ipartners(0)
      flagmc=.false.
c
      ztmp=zPY6Q(ileg,xm12,xm22,shat,x,yi,yj,tk,uk,q1q,q2q)
      xitmp=xiPY6Q(ileg,xm12,xm22,shat,x,yi,yj,tk,uk,q1q,q2q)
      xjactmp=xjacPY6Q_xiztoxy(ileg,xm12,xm22,shat,x,yi,yj,tk,uk,
     &                       q1q,q2q)

      do npartner=1,ipartners(0)
c This loop corresponds to the sum over colour lines l in the
c xmcsubt note
            z(npartner)=ztmp
            xi(npartner)=xitmp
            xjac(npartner)=xjactmp
c
c Compute deadzones:
            lzone(npartner)=.true.
            mstj50=0
            mstp67=0
            if(mstp67.eq.2)then
               if(ileg.le.2.and.npartner.gt.2)then
                  do i=0,3
                     p_born_npartner(i)=p_born(i,npartner)
                     p_born_fksfather(i)=p_born(i,fksfather)
                  enddo
                  theta2_cc=get_angle(p_born_npartner,p_born_fksfather)
                  theta2_cc=theta2_cc**2
                  theta2=4.d0*xitmp/(s*(1-ztmp))
                  if(theta2.ge.theta2_cc)lzone(npartner)=.false.
               endif
            endif
            if(mstj50.eq.2)then
               if(ileg.eq.3.and.npartner.le.2)then
                  continue
               elseif(ileg.eq.4.and.npartner.le.2)then
                  do i=0,3
                     p_born_npartner(i)=p_born(i,npartner)
                     p_born_fksfather(i)=p_born(i,fksfather)
                  enddo
                  theta2_cc=get_angle(p_born_npartner,p_born_fksfather)
                  theta2_cc=theta2_cc**2
                  en_fks=sqrt(s)*(1-x)/2.d0
                  en_mother=en_fks/(1-ztmp)
                  theta2=max(ztmp/(1-ztmp),(1-ztmp)/ztmp)*xitmp/en_mother**2
                  if(theta2.ge.theta2_cc)lzone(npartner)=.false.
               endif
            endif
c Implementation of a maximum scale for the shower: the following scale
c simulates boson mass for VB production. It could be modified in future
        if(.not.dampMCsubt)then
           lzone(npartner)=.false.
           fff(npartner)=1.d0
           upper_scale(npartner)=sqrt(x*shat)
           upper_scale(npartner)=upper_scale(npartner)*fff(npartner)
           if(xi(npartner).le.upper_scale(npartner))lzone(npartner)=.true.
        endif
c
c Compute MC subtraction terms
        if(lzone(npartner))then
          if(.not.flagmc)flagmc=.true.
          if( (fsr .and. m_type.eq.8) .or.
     #        (isr .and. j_type.eq.8) )then
            if(i_type.eq.8)then
c g --> g g (icode=1) and go --> go g (SUSY) splitting 
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=2
                 if(1-x.lt.tiny)then
                    xkern=(g**2/N_p)*8*vca/s
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*8*vca*(1-x*(1-x))**2/(s*x**2)
                    xkernazi=-(g**2/N_p)*16*vca*(1-x)**2/(s*x**2)
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    call Qterms_reduced_spacelike(m_type,i_type,one,
     #                                            z(npartner),Q)
                    Q=Q/(1-z(npartner))
                    xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                 endif
              elseif(ileg.eq.3)then
c Works only for SUSY
                N_p=2
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced_SUSY(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                N_p=2
                if(1-x.lt.tiny)then
                  xkern=(g**2/N_p)*8*vca/s
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*( 8*vca*
     &                  (s**2*(1-(1-x)*x)-s*(1+x)*xm12+xm12**2)**2 )/
     &                  ( s*(s-xm12)**2*(s*x-xm12)**2 )
                  xkernazi=-(g**2/N_p)*(16*vca*s*(1-x)**2)/((s-xm12)**2)
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  call Qterms_reduced_timelike(j_type,i_type,one,z(npartner),Q)
                  Q=Q/(1-z(npartner))
                  xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                endif
              else
                write(*,*)'Error 4 in xmcsubt_PY6Q: forbidden ileg'
                write(*,*)ileg
                stop
              endif
            elseif(abs(i_type).eq.3)then
c g --> q qbar splitting (icode=2)
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=1
                 if(1-x.lt.tiny)then
                    xkern=0.d0
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*4*vtf*(1-x)*((1-x)**2+x**2)/(s*x)
                    xkernazi=0.d0
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    xkernazi=0.d0
                 endif
              elseif(ileg.eq.4)then
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                N_p=2
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*( 4*vtf*(1-x)*
     &                  (s**2*(1-2*(1-x)*x)-2*s*x*xm12+xm12**2) )/
     &                  ( (s-xm12)**2*(s*x-xm12) )
                  xkernazi=(g**2/N_p)*(16*vtf*s*(1-x)**2)/((s-xm12)**2)
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  call Qterms_reduced_timelike(j_type,i_type,one,z(npartner),Q)
                  Q=Q/(1-z(npartner))
                  xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                endif
              else
                write(*,*)'Error 5 in xmcsubt_PY6Q: forbidden ileg'
                write(*,*)ileg
                stop
              endif
            else
              write(*,*)'Error 3 in xmcsubt_PY6Q: unknown particle type'
              write(*,*)i_type
              stop
            endif
          elseif( (fsr .and. abs(m_type).eq.3) .or.
     #            (isr .and. abs(j_type).eq.3) )then
            if(abs(i_type).eq.3)then
c q --> g q (or qbar --> g qbar) splitting (icode=3)
c the fks parton is the one associated with 1 - z: this is because its
c rescaled energy is 1 - x and in the soft limit, where x --> z --> 1,
c it has to coincide with the fraction appearing in the AP kernel
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=2
                 if(1-x.lt.tiny)then
                    xkern=0.d0
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*4*vcf*(1-x)*((1-x)**2+1)/(s*x**2)
                    xkernazi=-(g**2/N_p)*16*vcf*(1-x)**2/(s*x**2)
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    call Qterms_reduced_spacelike(m_type,i_type,one,
     #                                            z(npartner),Q)
                    Q=Q/(1-z(npartner))
                    xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                 endif
              elseif(ileg.eq.3)then
                N_p=1
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
                N_p=1
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                if(1-x.lt.tiny)then
                  xkern=0.d0
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*
     &                  ( 4*vcf*(1-x)*(s**2*(1-x)**2+(s-xm12)**2) )/
     &                  ( (s-xm12)*(s*x-xm12)**2 )
                  xkernazi=0.d0
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              else
                write(*,*)'Error 6 in xmcsubt_PY6Q: unknown ileg'
                write(*,*)ileg
                stop              
              endif
            elseif(i_type.eq.8)then
c q --> q g splitting (icode=4) and sq --> sq g (SUSY) splitting
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=1
                 if(1-x.lt.tiny)then
                    xkern=(g**2/N_p)*8*vcf/s
                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
                    xkern=(g**2/N_p)*4*vcf*(1+x**2)/(s*x)
                    xkernazi=0.d0
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    xkernazi=0.d0
                 endif
              elseif(ileg.eq.3)then
                N_p=1
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
                  beta1=sqrt(1-4*s*xm12/(s+xm12-xm22)**2)
                  xkern=(g**2/N_p)*8*vcf*(1-yj)*beta1/(s*(1-yj*beta1))
                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  if(abs(PDG_type(j_fks)).le.6)then
c QCD branching
                    call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  else
c Non-QCD branching, here taken to be squark->squark gluon 
                    call AP_reduced_SUSY(j_type,i_type,one,z(npartner),ap)
                  endif
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
                N_p=1
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                if(1-x.lt.tiny)then
                  xkern=(g**2/N_p)*8*vcf/s
                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*4*vcf*
     &                  ( s**2*(1+x**2)-2*xm12*(s*(1+x)-xm12) )/
     &                  ( s*(s-xm12)*(s*x-xm12) )
                  xkernazi=0.d0
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              else
                write(*,*)'Error 7 in xmcsubt_PY6Q: unknown ileg'
                write(*,*)ileg
                stop
              endif
            else
              write(*,*)'Error 8 in xmcsubt_PY6Q: unknown particle type'
              write(*,*)i_type
              stop
            endif
          else
            write(*,*)'Error 2 in xmcsubt_PY6Q: unknown particle type'
            write(*,*)j_type,i_type
            stop
          endif
c
          if(dampMCsubt)then
            if(emscasharp)then
              if(tPY6Q.le.scalemax)then
                emscwgt(npartner)=1.d0
                emscav(npartner)=emsca_bare
              else
                emscwgt(npartner)=0.d0
                emscav(npartner)=scalemax
              endif
            else
              ptresc=(tPY6Q-scalemin)/(scalemax-scalemin)
              if(ptresc.le.0.d0)then
                emscwgt(npartner)=1.d0
                emscav(npartner)=emsca_bare
              elseif(ptresc.lt.1.d0)then
                emscwgt(npartner)=1-emscafun(ptresc,one)
                emscav(npartner)=emsca_bare
              else
                emscwgt(npartner)=0.d0
                emscav(npartner)=scalemax
              endif
            endif
          endif
c
        else
c Dead zone
          xkern=0.d0
          xkernazi=0.d0
          if(dampMCsubt)then
            emscav(npartner)=etot
            emscwgt(npartner)=0.d0
          endif
        endif
        xkern=xkern*gfactsf
        xkernazi=xkernazi*gfactazi*gfactsf
        born_red=0.d0
        born_red_tilde=0.d0
        do cflows=1,colorflow(npartner,0)
c In the case of MC over colour flows, cflows will be passed from outside
          born_red=born_red+
     #             bornbars(colorflow(npartner,cflows))
          born_red_tilde=born_red_tilde+
     #                   bornbarstilde(colorflow(npartner,cflows))
        enddo
        xmcxsec(npartner) = xkern*born_red + xkernazi*born_red_tilde
        if(dampMCsubt)
     #    xmcxsec(npartner)=xmcxsec(npartner)*emscwgt(npartner)
        wgt = wgt + xmcxsec(npartner)
c
        if(xmcxsec(npartner).lt.0.d0)then
           write(*,*) 'Fatal error in xmcsubt_PY6Q',
     #                npartner,xmcxsec(npartner)
           do i=1,nexternal
              write(*,*) 'particle ',i,', ',(pp(j,i),j=0,3)
           enddo
           stop
        endif
c End of loop over colour partners
      enddo
c Assign emsca on statistical basis
      if(extra.and.wgt.gt.1.d-30)then
        rrnd=ran2()
        wgt1=0.d0
        jpartner=0
        do npartner=1,ipartners(0)
          if(lzone(npartner).and.jpartner.eq.0)then
            wgt1 = wgt1 + xmcxsec(npartner)
            if(wgt1.ge.rrnd*wgt)then
              jpartner=ipartners(npartner)
              mpartner=npartner
            endif
          endif
        enddo
c
        if(jpartner.eq.0)then
          write(*,*)'Error in xmcsubt_PY6Q: emsca unweighting failed'
          stop
        else
          emsca=emscav(mpartner)
        endif
      endif
      if(dampMCsubt.and.wgt.lt.1.d-30)emsca=etot
c Additional information for LHE
      if(AddInfoLHE)then
        fksfather_lhe=fksfather
        if(jpartner.ne.0)then
          ipartner_lhe=jpartner
        else
c min() avoids troubles if ran2()=1
          ipartner_lhe=min( int(ran2()*ipartners(0))+1,ipartners(0) )
          ipartner_lhe=ipartners(ipartner_lhe)
        endif
        scale1_lhe=tPY6Q
      endif
c
      if(dampMCsubt)then
        if(emsca.lt.scalemin)then
          write(*,*)'Error in xmcsubt_PY6Q: emsca too small',emsca,jpartner
          if(.not.lzone(npartner))then
            write(*,*)'because configuration in dead zone '
          else 
            stop
          endif
        endif
      endif
c
      do npartner=1,ipartners(0)
        xmcxsec(npartner) = xmcxsec(npartner) * probne
      enddo
      do npartner=ipartners(0)+1,nexternal
        xmcxsec(npartner) = 0.d0
      enddo
c No need to multiply this weight by probne, because it is ignored in 
c normal running. When doing the testing (test_MC), also the other
c pieces are not multiplied by probne.
c$$$      wgt=wgt*probne
c
      return
      end





      subroutine xmcsubt_PY6PT(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
c Main routine for MC counterterms
      implicit none
      include "nexternal.inc"
      include "fks.inc"
      include "coupl.inc"
      include "born_nhel.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      include "run.inc"

      double precision pp(0:3,nexternal),gfactsf,gfactcl,probne,wgt
      double precision xi_i_fks,y_ij_fks,xm12,xm22
      double precision xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc

      double precision emsca_bare,etot,ptresc,rrnd,ref_scale,
     & scalemin,scalemax,wgt1,ptPY6PT,emscainv,emscafun
      double precision emscwgt(nexternal),emscav(nexternal)
      integer jpartner,mpartner
      logical emscasharp

      double precision shattmp,dot,xkern,xkernazi,born_red,
     & born_red_tilde
      double precision bornbars(max_bcol), bornbarstilde(max_bcol)

      integer i,j,npartner,cflows,ileg,N_p
      double precision tk,uk,q1q,q2q,E0sq(nexternal),dE0sqdx(nexternal),
     # dE0sqdc(nexternal),x,yi,yj,xij,z(nexternal),xi(nexternal),
     # xjac(nexternal),zPY6PT,xiPY6PT,xjacPY6PT_xiztoxy,ap,Q,
     # beta,xfact,prefact,kn,knbar,kn0,kn_diff,beta1,betad,betas,
     # gfactazi,s,gfunsoft,gfuncoll,gfunazi,bogus_probne_fun,
     # ztmp,xitmp,xjactmp,get_angle,w1,w2,
     # p_born_npartner(0:3),p_born_fksfather(0:3)

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow

      integer fksfather
      common/cfksfather/fksfather

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision emsca
      common/cemsca/emsca

      double precision ran2,iseed
      external ran2

      logical isr,fsr
      logical extra

c Stuff to be written (depending on AddInfoLHE) onto the LHE file
      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe

      double precision becl,delta
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision zero,one,tiny,vtiny,ymin
      parameter (zero=0d0)
      parameter (one=1d0)
      parameter (vtiny=1.d-10)
      parameter (ymin=0.9d0)

      double precision pi
      parameter(pi=3.1415926535897932384626433d0)

      real*8 vcf,vtf,vca
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)

      integer mstj50,mstp67,en_fks,en_mother,theta2,theta2_cc
      double precision upper_scale(nexternal-1),fff(nexternal-1)
c
      double precision pmass(nexternal)
      include "pmass.inc"
c
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
        write(*,*)'Error in xmcsubt:_PY6PT inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

c May remove UseSudakov from the definition below if ptPY6PT (or similar
c variable, not yet defined) will not be needed in the computation of probne
      extra=dampMCsubt.or.AddInfoLHE.or.UseSudakov
      call xiz_driver(xi_i_fks,y_ij_fks,shat,pp,ileg,
     &                   xm12,xm22,tk,uk,q1q,q2q,ptPY6PT,extra)
      if(extra.and.ptPY6PT.lt.0.d0)then
        write(*,*)'Error in xmcsubt_PY6PT: ptPY6PT=',ptPY6PT
        stop
      endif
      call get_mbar(pp,y_ij_fks,ileg,bornbars,bornbarstilde)

      etot=2d0*sqrt( ebeam(1)*ebeam(2) )
      emsca=etot
      if(dampMCsubt)then
        emsca=0.d0
        ref_scale=sqrt( (1-xi_i_fks)*shat )
        scalemin=max(frac_low*ref_scale,scaleMClow)
        scalemax=max(frac_upp*ref_scale,scalemin+scaleMCdelta)
        emscasharp=(scalemax-scalemin).lt.(0.001d0*scalemax)
        if(emscasharp)then
          emsca_bare=scalemax
        else
          rrnd=ran2()
          rrnd=emscainv(rrnd,one)
          emsca_bare=scalemin+rrnd*(scalemax-scalemin)
        endif
      endif

c Distinguish initial or final state radiation
      isr=.false.
      fsr=.false.
      if(ileg.le.2)then
        isr=.true.
        delta=min(1.d0,deltaI)
      elseif(ileg.eq.3.or.ileg.eq.4)then
        fsr=.true.
        delta=min(1.d0,deltaO)
      else
        write(*,*)'Error in xmcsubt_PY6PT: unknown ileg'
        write(*,*)ileg
        stop
      endif

c Assign fks variables
      x=1-xi_i_fks
      if(isr)then
         yj=0.d0
         yi=y_ij_fks
      elseif(fsr)then
         yj=y_ij_fks
         yi=0.d0
      else
         write(*,*)'Error in xmcsubt_PY6PT: isr and fsr both false'
         stop
      endif

      s = shat
      xij=2*(1-xm12/shat-(1-x))/(2-(1-x)*(1-yj)) 
c
      if (abs(i_type).eq.3) then
         gfactsf=1d0
      else
         gfactsf=gfunsoft(x,s,zero,alsf,besf)
      endif
      becl=-(1.d0-ymin)
      gfactcl=gfuncoll(y_ij_fks,alsf,becl,one)
      gfactazi=gfunazi(y_ij_fks,alazi,beazi,delta)
c
c Non-emission probability. When UseSudakov=.true., the definition of
c probne may be moved later if necessary
      if(.not.UseSudakov)then
c this is standard MC@NLO
        probne=1.d0
      else
        probne=bogus_probne_fun(ptPY6PT)
      endif
c
      wgt=0.d0
      nofpartners=ipartners(0)
      flagmc=.false.
c
      ztmp=zPY6PT(ileg,xm12,xm22,shat,x,yi,yj,tk,uk,q1q,q2q)
      xitmp=xiPY6PT(ileg,xm12,xm22,shat,x,yi,yj,tk,uk,q1q,q2q)
      xjactmp=xjacPY6PT_xiztoxy(ileg,xm12,xm22,shat,x,yi,yj,tk,uk,
     &                       q1q,q2q)

      do npartner=1,ipartners(0)
c This loop corresponds to the sum over colour lines l in the
c xmcsubt note
            z(npartner)=ztmp
            xi(npartner)=xitmp
            xjac(npartner)=xjactmp

c$$$
c$$$c Compute deadzones
c$$$
c$$$

c Implementation of a maximum scale for the shower: the following scale
c simulates boson mass for VB production. It could be modified in future
        if(.not.dampMCsubt)then
           lzone(npartner)=.false.
           fff(npartner)=1.d0
           upper_scale(npartner)=sqrt(x*shat)
           upper_scale(npartner)=upper_scale(npartner)*fff(npartner)
           if(xi(npartner).le.upper_scale(npartner))lzone(npartner)=.true.
        endif
c
c Compute MC subtraction terms
        if(lzone(npartner))then
          if(.not.flagmc)flagmc=.true.
          if( (fsr .and. m_type.eq.8) .or.
     #        (isr .and. j_type.eq.8) )then
            if(i_type.eq.8)then
c g --> g g (icode=1) and go --> go g (SUSY) splitting 
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=2
                 if(1-x.lt.tiny)then
c$$$                    xkern=(g**2/N_p)*8*vca/s
c$$$                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
c$$$                    xkern=(g**2/N_p)*8*vca*(1-x*(1-x))**2/(s*x**2)
c$$$                    xkernazi=-(g**2/N_p)*16*vca*(1-x)**2/(s*x**2)
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    call Qterms_reduced_spacelike(m_type,i_type,one,
     #                                            z(npartner),Q)
                    Q=Q/(1-z(npartner))
                    xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                 endif
              elseif(ileg.eq.3)then
c Works only for SUSY
                N_p=2
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
c$$$                  xkern=0.d0
c$$$                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced_SUSY(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                N_p=2
                if(1-x.lt.tiny)then
c$$$                  xkern=(g**2/N_p)*8*vca/s
c$$$                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
                  xkern=(g**2/N_p)*( 8*vca*
     &                  (s**2*(1-(1-x)*x)-s*(1+x)*xm12+xm12**2)**2 )/
     &                  ( s*(s-xm12)**2*(s*x-xm12)**2 )
                  xkernazi=-(g**2/N_p)*(16*vca*s*(1-x)**2)/((s-xm12)**2)
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  call Qterms_reduced_timelike(j_type,i_type,one,z(npartner),Q)
                  Q=Q/(1-z(npartner))
                  xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                endif
              else
                write(*,*)'Error 4 in xmcsubt_PY6PT: forbidden ileg'
                write(*,*)ileg
                stop
              endif
            elseif(abs(i_type).eq.3)then
c g --> q qbar splitting (icode=2)
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=1
                 if(1-x.lt.tiny)then
c$$$                    xkern=0.d0
c$$$                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
c$$$                    xkern=(g**2/N_p)*4*vtf*(1-x)*((1-x)**2+x**2)/(s*x)
c$$$                    xkernazi=0.d0
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    xkernazi=0.d0
                 endif
              elseif(ileg.eq.4)then
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                N_p=2
                if(1-x.lt.tiny)then
c$$$                  xkern=0.d0
c$$$                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
c$$$                  xkern=(g**2/N_p)*( 4*vtf*(1-x)*
c$$$     &                  (s**2*(1-2*(1-x)*x)-2*s*x*xm12+xm12**2) )/
c$$$     &                  ( (s-xm12)**2*(s*x-xm12) )
c$$$                  xkernazi=(g**2/N_p)*(16*vtf*s*(1-x)**2)/((s-xm12)**2)
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  call Qterms_reduced_timelike(j_type,i_type,one,z(npartner),Q)
                  Q=Q/(1-z(npartner))
                  xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                endif
              else
                write(*,*)'Error 5 in xmcsubt_PY6PT: forbidden ileg'
                write(*,*)ileg
                stop
              endif
            else
             write(*,*)'Error 3 in xmcsubt_PY6PT: unknown particle type'
             write(*,*)i_type
             stop
            endif
          elseif( (fsr .and. abs(m_type).eq.3) .or.
     #            (isr .and. abs(j_type).eq.3) )then
            if(abs(i_type).eq.3)then
c q --> g q (or qbar --> g qbar) splitting (icode=3)
c the fks parton is the one associated with 1 - z: this is because its
c rescaled energy is 1 - x and in the soft limit, where x --> z --> 1,
c it has to coincide with the fraction appearing in the AP kernel
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=2
                 if(1-x.lt.tiny)then
c$$$                    xkern=0.d0
c$$$                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
c$$$                    xkern=(g**2/N_p)*4*vcf*(1-x)*((1-x)**2+1)/(s*x**2)
c$$$                    xkernazi=-(g**2/N_p)*16*vcf*(1-x)**2/(s*x**2)
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    call Qterms_reduced_spacelike(m_type,i_type,one,
     #                                            z(npartner),Q)
                    Q=Q/(1-z(npartner))
                    xkernazi=prefact*xfact*xjac(npartner)*Q/xi(npartner)
                 endif
              elseif(ileg.eq.3)then
                N_p=1
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
c$$$                  xkern=0.d0
c$$$                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
                N_p=1
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                if(1-x.lt.tiny)then
c$$$                  xkern=0.d0
c$$$                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
c$$$                  xkern=(g**2/N_p)*
c$$$     &                  ( 4*vcf*(1-x)*(s**2*(1-x)**2+(s-xm12)**2) )/
c$$$     &                  ( (s-xm12)*(s*x-xm12)**2 )
c$$$                  xkernazi=0.d0
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              else
                write(*,*)'Error 6 in xmcsubt_PY6PT: unknown ileg'
                write(*,*)ileg
                stop              
              endif
            elseif(i_type.eq.8)then
c q --> q g splitting (icode=4) and sq --> sq g (SUSY) splitting
              if(ileg.eq.1.or.ileg.eq.2)then
                 N_p=1
                 if(1-x.lt.tiny)then
c$$$                    xkern=(g**2/N_p)*8*vcf/s
c$$$                    xkernazi=0.d0
                 elseif(1-yi.lt.tiny)then
c$$$                    xkern=(g**2/N_p)*4*vcf*(1+x**2)/(s*x)
c$$$                    xkernazi=0.d0
                 else
                    xfact=(1-yi)*(1-x)/x
                    prefact=4/(s*N_p)
                    call AP_reduced(m_type,i_type,one,z(npartner),ap)
                    ap=ap/(1-z(npartner))
                    xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                    xkernazi=0.d0
                 endif
              elseif(ileg.eq.3)then
                N_p=1
c ileg = 3: xm12 = squared FKS-mother and FKS-sister mass
c           xm22 = squared recoil mass
                w1=-q1q+q2q-tk
                w2=-q2q+q1q-uk
                if(1-x.lt.tiny)then
c$$$                  beta1=sqrt(1-4*s*xm12/(s+xm12-xm22)**2)
c$$$                  xkern=(g**2/N_p)*8*vcf*(1-yj)*beta1/(s*(1-yj*beta1))
c$$$                  xkernazi=0.d0
                else
                  kn=veckn_ev
                  knbar=veckbarn_ev
                  kn0=xp0jfks
                  xfact=(2-(1-x)*(1-(kn0/kn)*yj))/kn*knbar*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  if(abs(PDG_type(j_fks)).le.6)then
c QCD branching
                    call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  else
c Non-QCD branching, here taken to be squark->squark gluon 
                    call AP_reduced_SUSY(j_type,i_type,one,z(npartner),ap)
                  endif
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              elseif(ileg.eq.4)then
                N_p=1
c ileg = 4: xm12 = squared recoil mass
c           xm22 = 0 = squared FKS-mother and FKS-sister mass
                if(1-x.lt.tiny)then
c$$$                  xkern=(g**2/N_p)*8*vcf/s
c$$$                  xkernazi=0.d0
                elseif(1-yj.lt.tiny)then
c$$$                  xkern=(g**2/N_p)*4*vcf*
c$$$     &                  ( s**2*(1+x**2)-2*xm12*(s*(1+x)-xm12) )/
c$$$     &                  ( s*(s-xm12)*(s*x-xm12) )
c$$$                  xkernazi=0.d0
                else
                  beta=1-xm12/s
                  xfact=(2-(1-x)*(1-yj))/xij*beta*(1-x)*(1-yj)
                  prefact=2/(s*N_p)
                  call AP_reduced(j_type,i_type,one,z(npartner),ap)
                  ap=ap/(1-z(npartner))
                  xkern=prefact*xfact*xjac(npartner)*ap/xi(npartner)
                  xkernazi=0.d0
                endif
              else
                write(*,*)'Error 7 in xmcsubt_PY6PT: unknown ileg'
                write(*,*)ileg
                stop
              endif
            else
             write(*,*)'Error 8 in xmcsubt_PY6PT: unknown particle type'
             write(*,*)i_type
             stop
            endif
          else
            write(*,*)'Error 2 in xmcsubt_PY6PT: unknown particle type'
            write(*,*)j_type,i_type
            stop
          endif
c
          if(dampMCsubt)then
            if(emscasharp)then
              if(ptPY6PT.le.scalemax)then
                emscwgt(npartner)=1.d0
                emscav(npartner)=emsca_bare
              else
                emscwgt(npartner)=0.d0
                emscav(npartner)=scalemax
              endif
            else
              ptresc=(ptPY6PT-scalemin)/(scalemax-scalemin)
              if(ptresc.le.0.d0)then
                emscwgt(npartner)=1.d0
                emscav(npartner)=emsca_bare
              elseif(ptresc.lt.1.d0)then
                emscwgt(npartner)=1-emscafun(ptresc,one)
                emscav(npartner)=emsca_bare
              else
                emscwgt(npartner)=0.d0
                emscav(npartner)=scalemax
              endif
            endif
          endif
c
        else
c Dead zone
          xkern=0.d0
          xkernazi=0.d0
          if(dampMCsubt)then
            emscav(npartner)=etot
            emscwgt(npartner)=0.d0
          endif
        endif
        xkern=xkern*gfactsf
        xkernazi=xkernazi*gfactazi*gfactsf
        born_red=0.d0
        born_red_tilde=0.d0
        do cflows=1,colorflow(npartner,0)
c In the case of MC over colour flows, cflows will be passed from outside
          born_red=born_red+
     #             bornbars(colorflow(npartner,cflows))
          born_red_tilde=born_red_tilde+
     #                   bornbarstilde(colorflow(npartner,cflows))
        enddo
        xmcxsec(npartner) = xkern*born_red + xkernazi*born_red_tilde
        if(dampMCsubt)
     #    xmcxsec(npartner)=xmcxsec(npartner)*emscwgt(npartner)
        wgt = wgt + xmcxsec(npartner)
c
        if(xmcxsec(npartner).lt.0.d0)then
           write(*,*) 'Fatal error in xmcsubt_PY6PT',
     #                npartner,xmcxsec(npartner)
           do i=1,nexternal
              write(*,*) 'particle ',i,', ',(pp(j,i),j=0,3)
           enddo
           stop
        endif
c End of loop over colour partners
      enddo
c Assign emsca on statistical basis
      if(extra.and.wgt.gt.1.d-30)then
        rrnd=ran2()
        wgt1=0.d0
        jpartner=0
        do npartner=1,ipartners(0)
          if(lzone(npartner).and.jpartner.eq.0)then
            wgt1 = wgt1 + xmcxsec(npartner)
            if(wgt1.ge.rrnd*wgt)then
              jpartner=ipartners(npartner)
              mpartner=npartner
            endif
          endif
        enddo
c
        if(jpartner.eq.0)then
          write(*,*)'Error in xmcsubt_PY6PT: emsca unweighting failed'
          stop
        else
          emsca=emscav(mpartner)
        endif
      endif
      if(dampMCsubt.and.wgt.lt.1.d-30)emsca=etot
c Additional information for LHE
      if(AddInfoLHE)then
        fksfather_lhe=fksfather
        if(jpartner.ne.0)then
          ipartner_lhe=jpartner
        else
c min() avoids troubles if ran2()=1
          ipartner_lhe=min( int(ran2()*ipartners(0))+1,ipartners(0) )
          ipartner_lhe=ipartners(ipartner_lhe)
        endif
        scale1_lhe=ptPY6PT
      endif
c
      if(dampMCsubt)then
        if(emsca.lt.scalemin)then
          write(*,*)'Error in xmcsubt_PY6PT: emsca too small',emsca,jpartner
          if(.not.lzone(npartner))then
            write(*,*)'because configuration in dead zone '
          else 
            stop
          endif
        endif
      endif
c
      do npartner=1,ipartners(0)
        xmcxsec(npartner) = xmcxsec(npartner) * probne
      enddo
      do npartner=ipartners(0)+1,nexternal
        xmcxsec(npartner) = 0.d0
      enddo
c No need to multiply this weight by probne, because it is ignored in 
c normal running. When doing the testing (test_MC), also the other
c pieces are not multiplied by probne.
c$$$      wgt=wgt*probne
c
      return
      end






      subroutine xmcsubt_PY8(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     #                   wgt,nofpartners,lzone,flagmc,z,xmcxsec)
c Main routine for MC counterterms
      implicit none
      include "nexternal.inc"
      include "fks.inc"
      include "coupl.inc"
      include "born_nhel.inc"
      include "fks_powers.inc"
      include "madfks_mcatnlo.inc"
      include "run.inc"

      double precision pp(0:3,nexternal),gfactsf,gfactcl,probne,wgt
      double precision xi_i_fks,y_ij_fks,xm12,xm22
      double precision xmcxsec(nexternal)
      integer nofpartners
      logical lzone(nexternal),flagmc

      double precision emsca_bare,etot,ptresc,rrnd,ref_scale,
     & scalemin,scalemax,wgt1,tPY6Q,emscainv,emscafun
      double precision emscwgt(nexternal),emscav(nexternal)
      integer jpartner,mpartner

      double precision shattmp,dot,xkern,xkernazi,born_red,
     & born_red_tilde
      double precision bornbars(max_bcol), bornbarstilde(max_bcol)

      integer i,j,npartner,cflows,ileg,N_p
      double precision tk,uk,q1q,q2q,E0sq(nexternal),dE0sqdx(nexternal),
     # dE0sqdc(nexternal),x,yi,yj,xij,z(nexternal),xi(nexternal),
     # xjac(nexternal),zPY6Q,xiPY6Q,xjacPY6Q_xiztoxy,ap,Q,
     # beta,xfact,prefact,kn,knbar,kn0,kn_diff,beta1,betad,betas,
     # gfactazi,s,gfunsoft,gfuncoll,gfunazi,bogus_probne_fun,
     # ztmp,xitmp,xjactmp,w1,w2

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks


      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow

      integer fksfather
      common/cfksfather/fksfather

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision emsca
      common/cemsca/emsca

      double precision ran2,iseed
      external ran2

      logical isr,fsr
      logical extra

c Stuff to be written (depending on AddInfoLHE) onto the LHE file
      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe

      double precision becl,delta
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision zero,one,tiny,vtiny,ymin
      parameter (zero=0d0)
      parameter (one=1d0)
      parameter (vtiny=1.d-10)
      parameter (ymin=0.9d0)

      double precision pi
      parameter(pi=3.1415926535897932384626433d0)

      real*8 vcf,vtf,vca
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)

      double precision upper_scale(nexternal-1),fff(nexternal-1)
c
      double precision pmass(nexternal)
      include "pmass.inc"
c

      write(*,*)'PY8 not yet implemented!'
      stop


      return
      end




      subroutine get_mbar(p,y_ij_fks,ileg,bornbars,bornbarstilde)
c Computes barred amplitudes (bornbars) squared according
c to Odagiri's prescription (hep-ph/9806531).
c Computes barred azimuthal amplitudes (bornbarstilde) with
c the same method 
      implicit none

      include "genps.inc"
      include "nexternal.inc"
      include "born_nhel.inc"

      double precision p(0:3,nexternal)
      double precision y_ij_fks,bornbars(max_bcol),bornbarstilde(max_bcol)

      double precision zero
      parameter (zero=0.d0)
      double precision p_born_rot(0:3,nexternal-1)

      integer imother_fks,ileg

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double complex wgt1(2),W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

      double complex xij_aor
      common/cxij_aor/xij_aor

      double precision born,sumborn,borntilde
      integer i

      double precision vtiny,pi(0:3),pj(0:3),cphi_mother,sphi_mother
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))

      double precision xi_i_fks_ev,y_ij_fks_ev,t
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      logical rotategranny
      common/crotategranny/rotategranny

      double precision cthbe,sthbe,cphibe,sphibe
      common/cbeangles/cthbe,sthbe,cphibe,sphibe

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

c
c BORN
      call sborn(p_born,wgt1)
      born=dble(wgt1(1))
c born is the total born amplitude squared
      sumborn=0.d0
      do i=1,max_bcol
        sumborn=sumborn+jamp2(i)
c sumborn is the sum of the leading-color amplitudes squared
      enddo
      

c BORN TILDE
      if(ileg.eq.1.or.ileg.eq.2)then
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
            calculatedBorn=.false.
         else
            call sborn(p_born,wgt1)
         endif
         if (abs(m_type).eq.3) then
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
     #                wgt1(2) * dconjg(azifact)
         endif
      elseif(ileg.eq.3.or.ileg.eq.4)then
         if(abs(j_type).eq.3.and.i_type.eq.8)then
            wgt1(2)=0.d0
         elseif(m_type.eq.8)then
c Insert <ij>/[ij] which is not included by sborn()
            if(1.d0-y_ij_fks.lt.vtiny)then
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
     #                           cphi_mother,sphi_mother)
            else
               call getaziangles(p_born(0,imother_fks),
     #                           cphi_mother,sphi_mother)
            endif
            wgt1(2) = -(cphi_mother-ximag*sphi_mother)**2 *
     #                  wgt1(2) * azifact
         else
            write(*,*)'FATAL ERROR in get_mbar',
     #           i_type,j_type,i_fks,j_fks
            stop
         endif
      else
         write(*,*)'unknown ileg in get_mbar'
         stop
      endif

      borntilde=dble(wgt1(2))


c BARRED AMPLITUDES
      do i=1,max_bcol
         if (sumborn.ne.0d0) then
            bornbars(i)=jamp2(i)/sumborn * born
         elseif (born.eq.0d0 .or. jamp2(i).eq.0d0) then
            bornbars(i)=0d0
         else
            write (*,*) 'ERROR #1, dividing by zero'
            stop
         endif
         if (sumborn.ne.0d0) then
            bornbarstilde(i)=jamp2(i)/sumborn * borntilde
         elseif (borntilde.eq.0d0 .or. jamp2(i).eq.0d0) then
            bornbarstilde(i)=0d0
         else
            write (*,*) 'ERROR #2, dividing by zero'
            stop
         endif      
c bornbars(i) is the i-th leading-color amplitude squared re-weighted
c in such a way that the sum of bornbars(i) is born rather than sumborn.
c the same holds for bornbarstilde(i).
      enddo

      return
      end




      function gfuncoll(yy,alcl,becl,delta)
c Gets smoothly to 0 in the collinear limits; the function gfunsoft
c must be called before this function. The functional form is given
c in eq.(A.86) of FW, with alpha==alcl. tilde{x}_{DZ} is replaced here
c by ygcoll, and x_{DZ} by ymincl. The function is different from 1
c in the region ygcoll<|y|<1. Call with
c  becl<0  ==> ymincl=0
c  becl>0  ==> ymincl=Max(0,1-delta) for standard subtraction
c              ymincl=0 for zeta-subtraction
c  |becl|-->0 ==> ygcoll-->1
c  |becl|-->1 ==> ygcoll-->ymincl
c This function differs from the original one of the QQ code; 
c alcl, becl and delta are now given in input rather than in common;
c the dependence on gacl has been eliminated (was only useful for testing
c purposes), and as a consequence the entry xx has been removed
      implicit none
      real * 8 gfuncoll,yy,alcl,becl,delta,y,ymincl,ygcoll,tt,tmp
      integer isubttype
      parameter (isubttype=0)
c
      y=yy
      if(becl.lt.0.d0)then
        ymincl=0.d0
      else
        if(isubttype.eq.0)then
          ymincl=max(0.d0,1.d0-delta)
        elseif(isubttype.eq.1)then
          write(*,*)'No such option in gfuncoll',isubttype
          stop
        else
          write(*,*)'Fatal error #1 in gfuncoll',isubttype
          stop
        endif
      endif
      ygcoll=1.d0-(1-ymincl)*abs(becl)
      if(ygcoll.gt.0.99d0)ygcoll=0.99d0
      tt=(abs(y)-ygcoll)/(1.d0-ygcoll)
      if(tt.gt.1.d0)then
        write(6,*)'Fatal error #2 in gfuncoll',tt
        stop
      endif
      tmp=1.d0
      if(alcl.gt.0.d0)then
        if(tt.gt.0.d0.and.abs(y).lt.0.99d0)
     #    tmp=(1-tt)**(2*alcl)/(tt**(2*alcl)+(1-tt)**(2*alcl))
        if(abs(y).ge.0.99d0)tmp=0.d0
      endif
      gfuncoll=tmp
      return
      end


      function gfunazi(y,alazi,beazi,delta)
c This function multiplies the azimuthal correlation term in the MC 
c subtraction kernel; it is not the same as in the old QQ code. We have
c   alazi<0  ==>  gfunazi=1-gfuncoll(|alazi|)
c   alazi>0  ==>  gfunazi=0
c ie in the testing phase (alazi<0) we include an azimuthal-dependent
c contribution in the MC subtraction terms
      implicit none
      real*8 gfunazi,y,alazi,beazi,delta,aalazi,tmp,gfuncoll
c
      tmp=0.d0
      if(alazi.lt.0.d0)then
        aalazi=abs(alazi)
        tmp=1.d0-gfuncoll(y,aalazi,beazi,delta)
      endif
      gfunazi=tmp
      return
      end



      function gfunsoft(xx,xs,xxm12,alsf,besf)
c Gets smoothly to 0 in the soft limit. The functional form is given
c in eq.(A.86) of FW, with alpha==alsf. tilde{x}_{DZ} is replaced here
c by xgsoft, and x_{DZ} by xminsf. The function is different from 1
c in the region xgsoft<x<1. Call with
c  besf<0  ==> xminsf=4*m2/S_{hadr}
c  besf>0  ==> xminsf=tilde{rho} for standard subtraction
c              xminsf=1-sqrt{zeta} for zeta-subtraction
c  |besf|-->0 ==> xgsoft-->1
c  |besf|-->1 ==> xgsoft-->xminsf
c This function has been derived from the analogous function in the
c QQ code; alsf and besf are now given in input rather than in common;
c xm12 replaced xmq2; the functional form of rho has been modified to
c render it consistent with that relevant to single top production;
c the definition of xminsf for besf>0 doesn't depend on the (soft)
c subtraction parameter any longer.
c If alsf<0, gfunsoft equals 1 everywhere. This option should be used
c for testing purposes only
      implicit none
      real * 8 gfunsoft,xx,xs,xxm12,alsf,besf,x,s,xm12,xminsf,xgsoft,
     # tt,tmp
      integer isubttype
      parameter (isubttype=0)
c
      x=xx
      s=xs
      xm12=xxm12
      if(besf.lt.0.d0)then
c was m**2/sh
        xminsf=0.d0
      else
        if(isubttype.eq.0)then
c was m**2/s
          xminsf=0.d0
        elseif(isubttype.eq.1)then
          write(*,*)'No such option in gfunsoft',isubttype
          stop
        else
          write(*,*)'Fatal error #1 in gfunsoft',isubttype
          stop
        endif
      endif
      xgsoft=1.d0-(1-xminsf)*abs(besf)
      if(xgsoft.gt.0.99d0)xgsoft=0.99d0
      tt=(x-xgsoft)/(1.d0-xgsoft)
      if(tt.gt.1.d0)then
        write(6,*)'Fatal error #2 in gfunsoft',x
        stop
      endif
      tmp=1.d0
      if(alsf.gt.0.d0)then
        if(tt.gt.0.d0.and.x.lt.0.99d0)
     #    tmp=(1-tt)**(2*alsf)/(tt**(2*alsf)+(1-tt)**(2*alsf))
        if(x.ge.0.99d0)tmp=0.d0
      endif
      gfunsoft=tmp
      return
      end



      subroutine xiz_driver(xi_i_fks,y_ij_fks,sh,pp,ileg,
     #     xm12,xm22,xtk,xuk,xq1q,xq2q,qMC,extra)
c Determines the arguments entering the definition of z, x and xjacHW6_xiztoxy
c INPUTS:   pp,sh,xi_i_fks,y_ij_fks,extra
c OUTPUTS:  ileg,xm12,xm22,xtk,xuk,xq1q,xq2q,qMC
      implicit none
      include "nexternal.inc"
      include "coupl.inc"

      double precision zero
      parameter (zero=0.d0)

      double precision pp(0:3,nexternal)
      double precision sh,shtmp,xi_i_fks,y_ij_fks,yitmp,xij
      double precision xm12,xm22,xtk,xuk,xq1q,xq2q,qMC,tPY6Q,ptHW6
      double precision beta1,beta2,eps1,eps2,w1,w2,zeta1,zeta2
      integer ileg,j,i,nfinal
      logical extra

      double precision xs,xs2,xq1c,xq2c,xw1,xw2,dot

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision xmass_j_fks,pp_sister(0:3),
     &pp_rec(0:3),xp1(0:3),xp2(0:3),xk1(0:3),xk2(0:3),xk3(0:3)
      integer fksfather
      common/cfksfather/fksfather

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision tiny,tiny2
      parameter(tiny=1.d-6)

      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo

      double precision pmass(nexternal)
      include "pmass.inc"

c Determine ileg
      xmass_j_fks=pmass(j_fks)

      if(fksfather.le.2)then
        ileg=fksfather
      elseif(xmass_j_fks.ne.0.d0)then
        ileg=3
      elseif(xmass_j_fks.eq.0.d0)then
        ileg=4
      else
        write(*,*)'Error 1 in xiz_driver: unknown ileg'
        write(*,*)ileg,fksfather,xmass_j_fks
        stop
      endif
c NOTE: instead of using pmass(j_fks), one should determine ileg with
c pmass(fksfather).This is equivalent for QCD splittings, and one may
c wonder if it is so for SQCD ones. All kernels are summarized in the
c following table (m_sq = X, m_go = Y, m_q = W):
c
c  QCD
c  MOTHER    SISTER  FKS-PARTON
c    W   -->   W         0       SINGULAR
c    W   -->   0         W       SINGULAR IF W = 0
c    0   -->   W         W       SINGULAR IF W = 0
c    0   -->   0         0       SINGULAR
c
c  SQCD    
c  MOTHER    SISTER  FKS-PARTON
c    X   -->   X         0       SINGULAR
c    Y   -->   Y         0       SINGULAR
c    X   -->   Y         W       NOT SINGULAR
c    Y   -->   X         W       NOT SINGULAR 
c    X   -->   0         X       NOT SINGULAR
c    0   -->   X         X       NOT SINGULAR
c    Y   -->   0         Y       NOT SINGULAR
c    0   -->   Y         Y       NOT SINGULAR
c    X   -->   W         Y       NOT SINGULAR
c    Y   -->   W         X       NOT SINGULAR
c    W   -->   X         Y       NOT SINGULAR
c    W   -->   Y         X       NOT SINGULAR
c
c All singular kernels have m_mother = m_sister. In particular, kernels
c 7 and 8 are not singular even if there is a massless particle: their
c collinear limit is finite because the sister is massive, and their
c soft limit is finite as well, because the massless particle emitted
c is a quark.

c Determine and assign momenta
      do j=0,3
         pp_rec(j)=0.d0
         pp_sister(j)=0.d0
         xk1(j)=0.d0
         xk2(j)=0.d0
c xk1 and xk2 are never used for isr
         xp1(j)=pp(j,1)
         xp2(j)=pp(j,2)
         xk3(j)=pp(j,i_fks)
         if(ileg.gt.2)then
            pp_rec(j)=pp(j,1)+pp(j,2)-pp(j,i_fks)-pp(j,j_fks)
            pp_sister(j)=pp(j,j_fks)
            if(ileg.eq.3)then
               xk1(j)=pp_sister(j)
               xk2(j)=pp_rec(j)
            elseif(ileg.eq.4)then
               xk1(j)=pp_rec(j)
               xk2(j)=pp_sister(j)
            endif
         endif
      enddo
c
      nfinal=nexternal-2
      xm12=0.d0
      xm22=0.d0
      xq1q=0.d0
      xq2q=0.d0
      ptHW6=-1.d0
      tPY6Q=-1.d0
      qMC=-1.d0

c Determine invariants needed in MC functions in terms of fks
c variables: the invariants argument of MC functions are (p-k)^2,
c NOT - 2 p k.
c
c variables xm12, xm22, xq1q and xq2q are not computed when ileg=1,2
c since they never enter isr formulae in MC functions
      if(ileg.eq.1)then
         xtk=-sh*xi_i_fks*(1-y_ij_fks)/2
         xuk=-sh*xi_i_fks*(1+y_ij_fks)/2
         if(extra)then
            if(MonteCarlo.eq.'HERWIG6')then
               ptHW6=xi_i_fks/2.d0*sqrt(sh*(1-y_ij_fks**2))
               qMC=ptHW6
            elseif(MonteCarlo.eq.'HERWIGPP')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA6Q')then
               tPY6Q=sqrt(abs(-xtk))
               qMC=tPY6Q
            elseif(MonteCarlo.eq.'PYTHIA6PT')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA8')then
               write(*,*)'no such MonteCarlo yet'
               stop
            endif
         endif
      elseif(ileg.eq.2)then
         xtk=-sh*xi_i_fks*(1+y_ij_fks)/2
         xuk=-sh*xi_i_fks*(1-y_ij_fks)/2
         if(extra)then
            if(MonteCarlo.eq.'HERWIG6')then
               ptHW6=xi_i_fks/2.d0*sqrt(sh*(1-y_ij_fks**2))
               qMC=ptHW6
            elseif(MonteCarlo.eq.'HERWIGPP')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA6Q')then
               tPY6Q=sqrt(abs(-xuk))
               qMC=tPY6Q
            elseif(MonteCarlo.eq.'PYTHIA6PT')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA8')then
               write(*,*)'no such MonteCarlo yet'
               stop
            endif
         endif
      elseif(ileg.eq.3)then
         xm12=xmass_j_fks**2
         xm22=dot(pp_rec,pp_rec)
         xtk=-2*dot(xp1,xk3)
         xuk=-2*dot(xp2,xk3)
         xq1q=-2*dot(xp1,xk1)+xm12
         xq2q=-2*dot(xp2,xk2)+xm22
         if(extra)then
            if(MonteCarlo.eq.'HERWIG6')then
               w1=-xq1q+xq2q-xtk
               w2=-xq2q+xq1q-xuk
               eps2=1-(xm12-xm22)/(sh-w1)
               beta2=sqrt(eps2**2-4*sh*xm22/((sh-w1)**2))
               zeta1=( (2*sh-(sh-w1)*eps2)*w2+
     #                 (sh-w1)*((w1+w2)*beta2-eps2*w1) )/
     #                 ( (sh-w1)*beta2*(2*sh-(sh-w1)*eps2+(sh-w1)*beta2) )
               ptHW6=zeta1*((1-zeta1)*w1-zeta1*xm12)
               ptHW6=sqrt(ptHW6)
               qMC=ptHW6
            elseif(MonteCarlo.eq.'HERWIGPP')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA6Q')then
               tPY6Q=abs(-xq1q+xq2q-xtk)
               tPY6Q=sqrt(tPY6Q)
               qMC=tPY6Q
            elseif(MonteCarlo.eq.'PYTHIA6PT')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA8')then
               write(*,*)'no such MonteCarlo yet'
               stop
            endif
         endif
      elseif(ileg.eq.4)then
         xm12=dot(pp_rec,pp_rec)
         xm22=0.d0
         xtk=-2*dot(xp1,xk3)
         xuk=-2*dot(xp2,xk3)
         xij=2*(1-xm12/sh-xi_i_fks)/(2-xi_i_fks*(1-y_ij_fks))
         yitmp=1-dot(xp1,xk2)*4.d0/(sh*xij)
         xw2=sh*xi_i_fks*xij*(1-y_ij_fks)/2.d0
         xq2q=-sh*xij*(1+yitmp)/2.d0
         xq1q=xuk+xq2q+xw2
         if(extra)then
            if(MonteCarlo.eq.'HERWIG6')then
               w1=-xq1q+xq2q-xtk
               w2=xw2
               eps1=1+xm12/(sh-w2)
               beta1=sqrt(eps1**2-4*sh*xm12/(sh-w2)**2)
               zeta2=( (2*sh-(sh-w2)*eps1)*w1+
     #                 (sh-w2)*((w1+w2)*beta1-eps1*w2) )/
     #                 ( (sh-w2)*beta1*(2*sh-(sh-w2)*eps1+(sh-w2)*beta1) )
               ptHW6=zeta2*(1-zeta2)*w2
               ptHW6=sqrt(ptHW6)
               qMC=ptHW6
            elseif(MonteCarlo.eq.'HERWIGPP')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA6Q')then
               tPY6Q=abs(xw2)
               tPY6Q=sqrt(tPY6Q)
               qMC=tPY6Q
            elseif(MonteCarlo.eq.'PYTHIA6PT')then
               write(*,*)'no such MonteCarlo yet'
               stop
            elseif(MonteCarlo.eq.'PYTHIA8')then
               write(*,*)'no such MonteCarlo yet'
               stop
            endif
         endif
      else
         write(*,*)'Error: xiz_driver assigned wrong ileg'
         stop
      endif

c
c Checks
      if(sh.ge.1d0)then
         shtmp=sh
         tiny2=tiny
      elseif(sh.lt.1d0)then
         shtmp=1d0
         tiny2=1d1*tiny
      endif
      if(ileg.le.2)then
         if((abs(xtk+2*dot(xp1,xk3))/shtmp.ge.tiny2).or.
     &      (abs(xuk+2*dot(xp2,xk3))/shtmp.ge.tiny2))then
            write(*,*)'Imprecision 1 in xiz_driver'
            write(*,*)abs(xtk+2*dot(xp1,xk3))/shtmp,abs(xuk+2*dot(xp2,xk3))/shtmp
            stop
         endif
      elseif(ileg.eq.3)then
      elseif(ileg.eq.4)then
         if(((abs(xw2-2*dot(xk2,xk3))/shtmp.ge.tiny2)).or.
     &      ((abs(xq2q+2*dot(xp2,xk2))/shtmp.ge.tiny2)).or.
     &      ((abs(xq1q+2*dot(xp1,xk1)-xm12)/shtmp.ge.tiny2)))then
            write(*,*)'Imprecision 2 in xiz_driver'
            write(*,*)abs(xw2-2*dot(xk2,xk3))/shtmp,
     &                abs(xq2q+2*dot(xp2,xk2))/shtmp,
     &                abs(xq1q+2*dot(xp1,xk1)-xm12)/shtmp
            stop
         endif
      endif


      return
      end



C
C--MONTE CARLO FUNCTIONS
C

c
c
c Begin of HW6 stuff
c
c
c This functions are the analogue of the ones in xiz_st, except for the
c fact that it's generalized for a 2 --> N process.
c Most is formally unchanged with respect to single-top, with only few
c specifications:
c ileg=3 represents now emission of a gluon (or massless quark) from a
c generic MASSIVE final state leg, thus in this case k1 and xm12 are
c momentum and mass of the emitting leg, while k2 and xm22 are the ones
c of the recoiling system (xm22.ne.0);
c ileg=4 represents emission from a MASSLESS final state leg, thus k1 and
c xm12 belong to the recoiling system, k2 is the emitting momentum and
c xm22=0.
c Finally, E0 can now be chosen in N+1 ways as the dot product of the
c emitting momentum with any other one.
c
c
c         **********   CONVENTIONS ABOUT INVARIANTS   **********
c
c
c The invariants given in input to the routines follow FNR conventions
c (i.e., are defined as (p-k)^2). 
c
c Those used within these routines follow the notes, and therefore follow
c MNR conventions (i.e., are defined as -2p.k). Using eq.(2.7) of FNR
c and the table of the draft we obtain
c
c  MNR   FNR
c  q1c = m12-s-tk-q1q
c  q2c = m22-s-uk-q2q
c  w1  = -q1q+q2q-tk
c  w2  = q1q-q2q-uk
c
c
c
      function zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Returns Herwig shower variable z; xm12 is the emitting mass squared if
c ileg=3, it's the recoiling mass squared for ileg=4, and viceversa for
c xm22 (xm22=0 for ileg=4)
      implicit none
      integer ileg
      real*8 zHW6,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,v1,
     #v2,xa,xb,ss,w1,w2,beta2,tbeta1,eps2,zeta1,beta1,tbeta2,eps1,zeta2,
     #beta,betae0,betad,betas
      parameter (tiny=1.d-5)
c
c incoming parton #1 (left)
c
c momenta for ileg=1:
c
c p1  =  sqrt(s)/2*(1,0,0,1)
c p2  =  sqrt(s)/2*(1,0,0,-1)
c k3  =  sqrt(s)*(1-x)/2*(1,0,sqrt(1-yi**2),yi)
c k1  =  it doesn't matter
c k2  =  it doesn't matter
c
c yi is the cosine of the angle between p1 and k3
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          zHW6=1-(1-x)*(s*(1-yi)+4*e0sq*(1+yi))/(8*e0sq)
        elseif(1-yi.lt.tiny)then
          zHW6=x-(1-yi)*(1-x)*(s*x**2-4*e0sq)/(8*e0sq)
        else
          v1=xtk
          v2=xuk
          xa=e0sq/v1
          xb=v2/s
          ss=1-(1+xb)/xa
          if(ss.ge.0.d0)then
            zHW6=2*xa*(1-sqrt(ss))
          else
            zHW6=-1.d0
          endif
        endif
c incoming parton #2 (right)
c
c momenta for ileg=2:
c
c p1  =  sqrt(s)/2*(1,0,0,1)
c p2  =  sqrt(s)/2*(1,0,0,-1)
c k3  =  sqrt(s)*(1-x)/2*(1,0,sqrt(1-yi**2),-yi)
c k1  =  it doesn't matter
c k2  =  it doesn't matter
c
c yi is the cosine of the angle between p1 and k3
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          zHW6=1-(1-x)*(s*(1-yi)+4*e0sq*(1+yi))/(8*e0sq)
        elseif(1-yi.lt.tiny)then
          zHW6=x-(1-yi)*(1-x)*(s*x**2-4*e0sq)/(8*e0sq)
c notice that actually the functional dependence upon yi and x
c for ileg = 1 and for ileg = 2 is the same for all Herwig
c functions: this is because yi is always the fks cosine 
c (see k3(4) above), both for ileg = 1 and for ileg = 2,
c which entails the following relations:
c
c  v1(ileg = 1) = v2(ileg = 2)
c  v2(ileg = 1) = v1(ileg = 2)
c  xtx(ileg = 1) = xuk(ileg = 2) 
c  xux(ileg = 1) = xtk(ileg = 2).
c
c since ileg = 2 is obtained from ileg = 1 by replacing
c (v1(ileg = 1), v2(ileg = 1)) <--> (v2(ileg = 2), v1(ileg = 2)),
c then everything remains the same.
        else
          v1=xtk
          v2=xuk
          xa=e0sq/v2
          xb=v1/s
          ss=1-(1+xb)/xa
          if(ss.ge.0.d0)then
            zHW6=2*xa*(1-sqrt(ss))
          else
            zHW6=-1.d0
          endif
        endif
c outgoing parton #3 (massive)
c
c here xm12 and k1 are mass and momentum of the emitting leg
c while xm22 and k2 are the quantities of the recoiling system: for
c this reason, the formulae (A.12) and (A.17)-(A.23) of the single-
c top paper hold formally unchanged.
c
c momenta for ileg=3 (up to an azimuth, here = 0):
c
c p1  =  sqrt(s)/2*(1,0,sqrt(1-yi**2),yi)
c p2  =  sqrt(s)/2*(1,0,-sqrt(1-yi**2),-yi)
c k1  =  (sqrt(mom_fks_sister**2+xm12),0,0,mom_fks_sister)
c k2  =  p1+p2-k1-k3
c k3  =  B*(1,0,sqrt(1-yj**2),yj)
c
c where B = sqrt(s)/2*(1-x) and mom_fks_sister is such that k2**2 = xm22
c
c yj is cosine of the angle between k1 and k3
      elseif(ileg.eq.3)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.(w1+xm12))then
          zHW6=-1.d0
          return
        endif

        if(1-x.lt.tiny)then
          beta=1-xm12/s
          betae0=sqrt(1-xm12/e0sq)
          betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
          betas=(s+xm12-xm22)/s
          zHW6=1+(1-x)*( s*(yj*betad-betas)/(4*e0sq*(1+betae0))-
     #          betae0*(xm12-xm22+s*(1+(1+yj)*betad-betas))
     #          /(betad*(xm12-xm22+s*(1+betad))) )
c the collinear limit is useless because the emitting leg is massive
c and then you don't have collinear divergences
        else
          tbeta1=sqrt(1-(w1+xm12)/e0sq)
          eps2=1-(xm12-xm22)/(s-w1)
          beta2=sqrt(eps2**2-4*s*xm22/((s-w1)**2))
          zeta1=( (2*s-(s-w1)*eps2)*w2+
     #          (s-w1)*((w1+w2)*beta2-eps2*w1) )/
     #          ( (s-w1)*beta2*(2*s-(s-w1)*eps2+(s-w1)*beta2) )
          zHW6=1-tbeta1*zeta1-w1/(2*(1+tbeta1)*e0sq)
        endif
c outgoing parton #4 (massless)
c
c here xm12 and k1 are mass and momentum of the recoiling system
c while xm22=0 and k2 are the quantities of the emitting leg: for
c this reason, the formulae (A.12) and (A.17)-(A.23) of the single
c top paper hold interchanging 1<-->2 and putting xm22=0
c
c momenta for ileg=4 (up to an azimuth, here = 0):
c
c p1  =  sqrt(s)/2*(1,0,sqrt(1-yi**2),yi)
c p2  =  sqrt(s)/2*(1,0,-sqrt(1-yi**2),-yi)
c k1  =  p1+p2-k2-k3
c k2  =  A*(1,0,0,1) 
c k3  =  B*(1,0,sqrt(1-yj**2),yj)
c
c where B = sqrt(s)/2*(1-x) and A = (s*x-xm12)/(sqrt(s)*(2-(1-x)*(1-y)))
c in such a way that k1**2 = xm12
c
c yj is the cosine of the angle between k2 and k3
      elseif(ileg.eq.4)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.w2)then
          zHW6=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          zHW6=1-(1-x)*( (s-xm12)*(1-yj)/(8*e0sq)+
     #                    s*(1+yj)/(2*(s-xm12)) )
        elseif(1-yj.lt.tiny)then
          zHW6=(s*x-xm12)/(s-xm12)+(1-yj)*(1-x)*(s*x-xm12)*
     #          ( (s-xm12)**2*(s*(1-2*x)+xm12)+
     #            4*e0sq*s*(s*x-xm12*(2-x)) )/
     #          ( 8*e0sq*(s-xm12)**3 )
        else
          beta1=sqrt((1+xm12/(s-w2))**2-4*xm12*s/(s-w2)**2)
          tbeta2=sqrt(1-w2/e0sq)
          eps1=1+xm12/(s-w2)
          zeta2=( (2*s-(s-w2)*eps1)*w1+
     #            (s-w2)*((w1+w2)*beta1-eps1*w2) )/
     #          ( (s-w2)*beta1*(2*s-(s-w2)*eps1+(s-w2)*beta1) )
          zHW6=1-tbeta2*zeta2-w2/(2*(1+tbeta2)*e0sq)
        endif
      else

        write(6,*)'zHW6: unknown parton number'
        stop
      endif

      return
      end



      function xiHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Returns Herwig shower variable xi; xm12 is the emitting mass squared if
c ileg=3, it's the recoiling mass squared for ileg=4 and viceversa for xm22.
      implicit none
      integer ileg
      real*8 xiHW6,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,z,
     #zHW6,v2,v1,w1,w2,beta,betae0,betad,betas
      parameter (tiny=1.d-5)
c
c incoming parton #1 (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          xiHW6=2*s*(1-yi)/(s*(1-yi)+4*e0sq*(1+yi))
        elseif(1-yi.lt.tiny)then
          xiHW6=(1-yi)*s*x**2/(4*e0sq)
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            v2=xuk
            xiHW6=2*(1+v2/(s*(1-z)))
          else
            xiHW6=-1.d0
          endif
        endif
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          xiHW6=2*s*(1-yi)/(s*(1-yi)+4*e0sq*(1+yi))
        elseif(1-yi.lt.tiny)then
          xiHW6=(1-yi)*s*x**2/(4*e0sq)
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            v1=xtk
            xiHW6=2*(1+v1/(s*(1-z)))
          else
            xiHW6=-1.d0
          endif
        endif
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.(w1+xm12))then
          xiHW6=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=1-xm12/s
          betae0=sqrt(1-xm12/e0sq)
          betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
          betas=(s+xm12-xm22)/s
          xiHW6=
     # ( s*(1+betae0)*betad*(xm12-xm22+s*(1+betad))*(yj*betad-betas) )/
     # ( -4*e0sq*betae0*(1+betae0)*(xm12-xm22+s*(1+(1+yj)*betad-betas))+
     #   (s*betad*(xm12-xm22+s*(1+betad))*(yj*betad-betas)) )
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xiHW6=w1/(2*z*(1-z)*e0sq)
          else
            xiHW6=-1.d0
          endif
        endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.w2)then
          xiHW6=-1.d0
          return
        endif
        if(1-x.lt.tiny)then
          xiHW6=2*(s-xm12)**2*(1-yj)/
     #           ( (s-xm12)**2*(1-yj)+4*e0sq*s*(1+yj) )
        elseif(1-yj.lt.tiny)then
          xiHW6=(s-xm12)**2*(1-yj)/(4*e0sq*s)
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xiHW6=w2/(2*z*(1-z)*e0sq)
          else
            xiHW6=-1.d0
          endif
        endif
      else
        write(6,*)'xiHW6: unknown parton number'
        stop
      endif
      return
      end


      function xjacHW6_xiztoxy(ileg,e0sq,de0sqdx,de0sqdc,xm12,xm22,s,x,yi,
     #                      yj,xtk,xuk,xq1q,xq2q)
c Returns the jacobian d(z,xi)/d(x,c), where z and xi are Herwig shower 
c variables, and x and c are FKS variables. In the case of initial-state
c emissions, we have x=1-xii and c=yi; in the case of final-state emissions,
c we have x=1-xii and c=yj. e0sq is the shower scale squared, and
c de0sqdx=d(e0sq)/dx, de0sqdc=F*d(e0sq)/dc, with F=(1-yi^2) for legs 1 and 2,
c F=1 for leg 3, and F=(1-yj) for leg 4
      implicit none
      integer ileg
      real*8 xjacHW6_xiztoxy,e0sq,de0sqdx,de0sqdc,xm12,xm22,s,x,yi,yj,
     &xtk,xuk,xq1q,xq2q,tiny,tmp,z,zHW6,xi,xiHW6,w1,w2,beta2,tbeta1,
     &eps2,zeta1,dw1dy,dw2dx,dw1dx,dw2dy,beta1,tbeta2,eps1,dq1cdx,
     &dq2qdx,dq1cdy,dq2qdy,beta,betae0,betad,betas,zmo,dw2dxred,w2red,
     &mom_fks_sister,denfkssisdx,denfkssisdy,afun,bfun,cfun,signfac,
     &dadx,dady,dbdx,dbdy,dcdx,dcdy,coeffw1w2,coeffe0w1,coeffe0w2,
     &dzdw1,dzdtbeta1,dzde0sq,dxidw1,dxide0sq,dtbeta1dw1,dtbeta1de0sq,
     &deps2dw1,dbeta2dw1,zdenom,kappa,dzeta1deps2,dzeta1dbeta2,dzeta1dw1,
     &dtotzeta1dw1,get_sign,en_fks_sister,mom_fks_sister_m,dmomfkssisdx,dmomfkssisdy

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      parameter (tiny=1.d-5)
c
      tmp=0.d0
c incoming parton #1 (left)
      if(ileg.eq.1)then
        if(1-x.lt.tiny)then
          tmp=-2*s/(s*(1-yi)+4*(1+yi)*e0sq)
        elseif(1-yi.lt.tiny)then
          tmp=-s*x**2/(4*e0sq)
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.lt.0.d0)then
            xjacHW6_xiztoxy=0.d0
            return
          endif
          xi=xiHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #              xtk,xuk,xq1q,xq2q)
          tmp=-s*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))*
     #         ( 1+de0sqdx*(1-x)*(1-yi)/(2*e0sq)+
     #             de0sqdc/(2*e0sq) )
        endif
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
        if(1-x.lt.tiny)then
          tmp=-2*s/(s*(1-yi)+4*(1+yi)*e0sq)
        elseif(1-yi.lt.tiny)then
          tmp=-s*x**2/(4*e0sq)
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.lt.0.d0)then
            xjacHW6_xiztoxy=0.d0
            return
          endif
          xi=xiHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #              xtk,xuk,xq1q,xq2q)
          tmp=-s*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))*
     #         ( 1+de0sqdx*(1-x)*(1-yi)/(2*e0sq)+
     #             de0sqdc/(2*e0sq) )
        endif
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk 
        if(e0sq.le.(w1+xm12))then
          xjacHW6_xiztoxy=0.d0
          return
        endif
        if(1-x.lt.tiny)then
          beta=1-xm12/s
          betae0=sqrt(1-xm12/e0sq)
          betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
          betas=(s+xm12-xm22)/s
          tmp=( s*betae0*(1+betae0)*betad*(xm12-xm22+s*(1+betad)) )/
     #    ( (-4*e0sq*(1+betae0)*(xm12-xm22+s*(1+betad*(1+yj)-betas)))+
     #    (xm12-xm22+s*(1+betad))*( xm12*(4+yj*betad-betas)-
     #    (xm22-s)*(yj*betad-betas) ) )
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xi=xiHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #                xtk,xuk,xq1q,xq2q)
            tbeta1=sqrt(1-(w1+xm12)/e0sq)
            eps2=1-(xm12-xm22)/(s-w1)
            beta2=sqrt((eps2**2)-(4*s*xm22)/((s-w1)**2))
            zeta1=( (2*s-(s-w1)*eps2)*w2+
     #              (s-w1)*((w1+w2)*beta2-eps2*w1) )/
     #            ( (s-w1)*beta2*(2*s-(s-w1)*eps2+(s-w1)*beta2) )
            afun=sqrt(s)*(1-x)*(xm12-xm22+s*x)*yj
            bfun=s*( (1+x)**2*(xm12**2+(xm22-s*x)**2-
     &               xm12*(2*xm22+s*(1+x**2)))+
     &               xm12*s*(1-x**2)**2*yj**2 )
            cfun=s*(-(1+x)**2+(1-x)**2*yj**2)
            signfac=get_sign(w1,w2,s,x,yj,xm12,xm22,afun,bfun,cfun)
            mom_fks_sister  =(afun+signfac*sqrt(bfun))/cfun
            mom_fks_sister_m=(afun-signfac*sqrt(bfun))/cfun
            if(min(abs(mom_fks_sister-veckn_ev),abs(mom_fks_sister_m-veckn_ev))
     &         .eq.abs(mom_fks_sister_m-veckn_ev))then
              write(*,*)'Get_sign assigned wrong solution'
              signfac=-signfac
            endif
            if(min(abs(mom_fks_sister-veckn_ev),abs(mom_fks_sister_m-veckn_ev))/
     &             abs(veckn_ev).ge.tiny)then
              write(*,*)'Numerical imprecision #1 in xjacHW6_xiztoxy'
            elseif(min(abs(mom_fks_sister-veckn_ev),abs(mom_fks_sister_m-veckn_ev))/
     &                 abs(veckn_ev).ge.1.d-3)then
              write(*,*)'Fatal imprecision #1 in xjacHW6_xiztoxy'
              stop
            endif
            mom_fks_sister=veckn_ev
            dadx=sqrt(s)*yj*(xm22-xm12+s*(1-2*x))
            dady=sqrt(s)*(1-x)*(xm12-xm22+s*x)
            dbdx=2*s*(1+x)*( xm12**2+(xm22-s*x)*(xm22-s*(1+2*x))
     #           -xm12*(2*xm22+s*(1+x+2*(x**2)+2*(1-x)*x*(yj**2))) )
            dbdy=2*xm12*(s**2)*((1-x**2)**2)*yj
            dcdx=-2*s*(1+x+(yj**2)*(1-x))
            dcdy=2*s*((1-x)**2)*yj
            dmomfkssisdx=(dadx+signfac*dbdx/(2*sqrt(bfun))-dcdx*mom_fks_sister)/cfun
            dmomfkssisdy=(dady+signfac*dbdy/(2*sqrt(bfun))-dcdy*mom_fks_sister)/cfun
            en_fks_sister=sqrt(xm12+mom_fks_sister**2)
            dw1dx=sqrt(s)*(yj*mom_fks_sister-en_fks_sister+(1-x)*(mom_fks_sister/en_fks_sister-yj)*dmomfkssisdx)
            dw2dx=-dw1dx-s
            dw1dy=-sqrt(s)*(1-x)*(mom_fks_sister+(yj-mom_fks_sister/en_fks_sister)*dmomfkssisdy)
            dw2dy=-dw1dy
c derivatives with respect to invariants
            dzdw1=-1/(2*e0sq*(1+tbeta1))
            dzdtbeta1=-zeta1+w1/(2*e0sq*((1+tbeta1)**2))
            dzde0sq=w1/(2*(e0sq**2)*(1+tbeta1))
            dxidw1=1/(2*e0sq*z*(1-z))
            dxide0sq=-w1/(2*(e0sq**2)*z*(1-z))
            dtbeta1dw1=-1/(2*tbeta1*e0sq)
            dtbeta1de0sq=(w1+xm12)/(2*tbeta1*(e0sq**2))
            deps2dw1=(eps2-1)/(s-w1)
            dbeta2dw1=1/(2*beta2)*(2*eps2*deps2dw1-8*s*xm22/(s-w1)**3)
            zdenom=2*s+(s-w1)*(beta2-eps2)
            kappa=(s-w1)*beta2*zdenom
            dzeta1deps2=-2*s*w1/(beta2*(zdenom**2))
            dzeta1dbeta2=-(dzeta1deps2+zeta1/beta2)

            dzeta1dw1=(1/kappa)*( (beta2-eps2)*(s-2*w1-w2) +
     #                     2*zeta1*beta2*(s+(s-w1)*(beta2-eps2)) )
            dtotzeta1dw1= dzeta1deps2*deps2dw1+dzeta1dbeta2*dbeta2dw1+
     #                    dzeta1dw1
c coefficients of the jacobian
            coeffw1w2=tbeta1/(2*e0sq*z*(1-z)*(s-w1)*beta2)
            coeffe0w2=coeffw1w2*w1/e0sq
            coeffe0w1=dxide0sq*
     #                (dzdw1-tbeta1*dtotzeta1dw1+dzdtbeta1*dtbeta1dw1)-
     #                dxidw1*
     #                (dzde0sq+dzdtbeta1*dtbeta1de0sq)
            tmp=-( 
     #       (dw1dy*dw2dx-dw1dx*dw2dy)*coeffw1w2 +
     #       (de0sqdx*dw2dy-de0sqdc*dw2dx)*coeffe0w2+
     #       (de0sqdx*dw1dy-de0sqdc*dw1dx)*coeffe0w1 )
          endif
        endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
        if(e0sq.le.w2)then
          xjacHW6_xiztoxy=0.d0
          return
        endif
        if(1-x.lt.tiny)then
          zmo=(s-xm12)*(1-yj)/(8*e0sq)+s*(1+yj)/(2*(s-xm12))
          tmp=-s/(4*e0sq*zmo)
        elseif(1-yj.lt.tiny)then
          tmp=-(s-xm12)/(4*e0sq)
        else
          z=zHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #            xtk,xuk,xq1q,xq2q)
          if(z.gt.0.d0)then
            xi=xiHW6(ileg,e0sq,xm12,xm22,s,x,yi,yj,
     #                xtk,xuk,xq1q,xq2q)
            beta1=sqrt((1+xm12/(s-w2))**2-4*xm12*s/(s-w2)**2)
            tbeta2=sqrt(1-w2/e0sq)
            eps1=1+xm12/(s-w2)
            dq1cdx=-(1-yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
            dq2qdx=-(1+yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
            dw2dxred=(s*(1+yj-x*(2*(1+yj)+x*(1-yj)))+2*xm12)/
     #               (1+yj+x*(1-yj))**2
            dw2dx=(1-yj)*dw2dxred
            dw1dx=dq1cdx+dq2qdx
            dq1cdy=(1-x)*(1-yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dq2qdy=(1-x)*(1+yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dw2dy=-2*(1-x)*(s*x-xm12)/(1+yj+x*(1-yj))**2
            dw1dy=dq1cdy+dq2qdy
            w2red=s*(1-x)*(x-xm12/s)/(2-(1-x)*(1-yj))
            tmp=-( 
     #       -(dw1dy*dw2dx-dw1dx*dw2dy)*
     #           (1+tbeta2)/(beta1*(s-w2))+
     #        (de0sqdx*dw1dy*w2-de0sqdc*dw1dx*w2red)*
     #           (1+tbeta2)/(e0sq*beta1*(s-w2))+
     #        (de0sqdx*dw2dy-de0sqdc*dw2dxred)*(1+tbeta2)*w2*(
     #           beta1**4*(s-w2)**4*(s+w1)-
     #           2*beta1**3*(s-w2)**3*(xm12*(s+w1)-w1*(s+w2))-
     #           beta1**2*(s-w2)**2*( (s*(s+xm12)+w2**2)*(s-w1-xm12)-
     #                                w2*(2*s-xm12)*(s+w1+xm12) )-
     #          (2*beta1*(s-w2)+s+w2-xm12)*xm12*(3*s+w2-xm12)*
     #          (s*(w2-w1)-(w2-xm12)*(w1+w2)) )/
     #        ( e0sq*beta1**3*(s-w2)**4*
     #          (beta1*(s-w2)+s+w2-xm12)**2 ) )*
     #        tbeta2/( 2*e0sq*(1+tbeta2)*z*(1-z) )
          endif
        endif
      else
        write(6,*)'xjacHW6_xiztoxy: unknown parton number'
        stop
      endif
      xjacHW6_xiztoxy=abs(tmp)

      return
      end






c
c
c Begin of PY6Q stuff
c
c ileg=1: emission from the initial parton incoming from the left
c ileg=2: emission from the initial parton incoming from the right
c ileg=3: emission from a massive final parton
c ileg=4: emission from a massless final parton
c xm12 is the fks-sister mass squared for ileg=3
c xm12 is the recoiler   mass squared for ileg=4
c xm22 is the recoiler   mass squared for ileg=3
c xm22 is the fks-sister mass squared for ileg=4 (i.e. it is =0)
c
c
c The invariants given in input to the routines follow FNR conventions
c (i.e., are defined as (p-k)^2). 
c
c Those used within these routines follow the notes, and therefore follow
c MNR conventions (i.e., are defined as -2p.k). Using eq.(2.7) of FNR
c and the table of the draft we obtain
c
c  MNR   FNR
c  q1c = m12-s-tk-q1q
c  q2c = m22-s-uk-q2q
c  w1  = -q1q+q2q-tk
c  w2  = q1q-q2q-uk
c
c Phase space parametrization (up to an azimuth)
c
c ileg=1
c p1  =  sqrt(s)/2*(1,0,0,1)
c p2  =  sqrt(s)/2*(1,0,0,-1)
c k3  =  sqrt(s)*(1-x)/2*(1,0,sqrt(1-yi**2),yi)
c k1  =  it doesn't matter
c k2  =  it doesn't matter
c
c ileg=2:
c p1  =  sqrt(s)/2*(1,0,0,1)
c p2  =  sqrt(s)/2*(1,0,0,-1)
c k3  =  sqrt(s)*(1-x)/2*(1,0,sqrt(1-yi**2),-yi)
c k1  =  it doesn't matter
c k2  =  it doesn't matter
c
c ileg=3
c here xm12 and k1 are mass and momentum of the emitting leg
c while xm22 and k2 are the quantities of the recoiling system: for
c this reason, the formulae (A.12) and (A.17)-(A.23) of the single-
c top paper hold formally unchanged.
c Define B = sqrt(s)/2*(1-x) and mom_fks_sister is such that k2**2 = xm22.
c p1  =  sqrt(s)/2*(1,0,sqrt(1-yi**2),yi)
c p2  =  sqrt(s)/2*(1,0,-sqrt(1-yi**2),-yi)
c k1  =  (sqrt(mom_fks_sister**2+xm12),0,0,mom_fks_sister)
c k2  =  p1+p2-k1-k3
c k3  =  B*(1,0,sqrt(1-yj**2),yj)
c
c ileg=4
c here xm12 and k1 are mass and momentum of the recoiling system
c while xm22=0 and k2 are the quantities of the emitting leg: for
c this reason, the formulae (A.12) and (A.17)-(A.23) of the single
c top paper hold interchanging 1<-->2 and putting xm22=0
c Define B = sqrt(s)/2*(1-x) and A = (s*x-xm12)/(sqrt(s)*(2-(1-x)*(1-y)))
c in such a way that k1**2 = xm12.
c p1  =  sqrt(s)/2*(1,0,sqrt(1-yi**2),yi)
c p2  =  sqrt(s)/2*(1,0,-sqrt(1-yi**2),-yi)
c k1  =  p1+p2-k2-k3
c k2  =  A*(1,0,0,1) 
c k3  =  B*(1,0,sqrt(1-yj**2),yj)
c
c
c
      function zPY6Q(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Returns PYTHIA energy shower variable
      implicit none
      integer ileg
      real*8 zPY6Q,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,w1,w2,
     &en_fks,en_fks_sister,mom_fks_sister,tiny

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      parameter(tiny=1.d-5)
c
      if(ileg.gt.4.or.ileg.le.0)then
         write(*,*)'error #1 in zPY6Q, unknown ileg ',ileg
         stop
      endif

c incoming parton #1 (left)
      if(ileg.eq.1)then
         zPY6Q=x
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
         zPY6Q=x
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
         w1=-xq1q+xq2q-xtk
         w2=-xq2q+xq1q-xuk
         if(1-x.lt.tiny)then
            zPY6Q=1-(1-x)*s/(s+xm12-xm22)
         else
            en_fks=sqrt(s)*(1-x)/2.d0
            mom_fks_sister=veckn_ev
            en_fks_sister=sqrt(mom_fks_sister**2+xm12)
            zPY6Q=en_fks_sister/(en_fks + en_fks_sister)
         endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            zPY6Q=1-s*(1-x)/(s-xm12)
         elseif(1-yj.lt.tiny)then
            zPY6Q=(s*x-xm12)/(s-xm12)+(1-yj)*(1-x)**2*s*(s*x-xm12)/
     &                                 ( 2*(s-xm12)**2 )
         else
            en_fks=sqrt(s)*(1-x)/2.d0
            en_fks_sister=sqrt(s)*(x-xm12/s)/(2-(1-x)*(1-yj))
            zPY6Q=en_fks_sister/(en_fks + en_fks_sister)
         endif
      endif

      return
      end



      function xiPY6Q(ileg,xm12,xm22,s,x,yi,yj,
     &                                  xtk,xuk,xq1q,xq2q)
c Returns PYTHIA evolution shower variable
      implicit none
      integer ileg
      real*8 xiPY6Q,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,z,zPY6Q,
     &en_fks,en_fks_sister,mom_fks_sister,xt,tiny,w1,w2,beta,betas,beta1

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      parameter(tiny=1.d-5)
c
      if(ileg.gt.4.or.ileg.le.0)then
         write(*,*)'error #1 in xiPY6Q, unknown ileg ',ileg
         stop
      endif
c incoming parton #1 (left)
      if(ileg.eq.1)then
         xiPY6Q=s*(1-x)*(1-yi)/2.d0
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
         xiPY6Q=s*(1-x)*(1-yi)/2.d0
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
         w1=-xq1q+xq2q-xtk
         w2=-xq2q+xq1q-xuk
         if(1-x.lt.tiny)then
            betas=1+(xm12-xm22)/s
            beta1=sqrt(1-4*s*xm12/(s+xm12-xm22)**2)
            xiPY6Q=s*(1-x)*betas*(1-yj*beta1)/2
         else
            en_fks=sqrt(s)*(1-x)/2.d0
            mom_fks_sister=veckn_ev
            en_fks_sister=sqrt(mom_fks_sister**2+xm12)
            beta=sqrt(1-xm12/en_fks_sister**2)
            xt=xm12+2*en_fks*en_fks_sister*(1-beta*yj)
            xiPY6Q=xt-xm12
c see pagg. 352 AND 361 of hep-ph/0603175
         endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiPY6Q=(1-yj)*(1-x)*(s-xm12)/2.d0
         elseif(1-yj.lt.tiny)then
            xiPY6Q=(1-yj)*(1-x)*(s*x-xm12)/2.d0
         else
            en_fks=sqrt(s)*(1-x)/2.d0
            en_fks_sister=sqrt(s)*(x-xm12/s)/(2-(1-x)*(1-yj))
            xt=2*en_fks*en_fks_sister*(1-yj)
            xiPY6Q=xt
         endif
      endif

      return
      end




      function xjacPY6Q_xiztoxy(ileg,xm12,xm22,s,x,yi,yj,
     &                                            xtk,xuk,xq1q,xq2q)
c Returns PYTHIA jacobian |d(xi_PY6Q,z_PY6Q)/d(x,y)|
      implicit none
      integer ileg
      real*8 xjacPY6Q_xiztoxy,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,
     &z,zPY6Q,en_fks,en_fks_sister,mom_fks_sister,mom_fks_sister_m,xt,tiny,tmp,w1,w2,
     &afun,bfun,cfun,signfac,get_sign,beta,dzPY6Qdenfks,dzPY6Qdenfkssis,dxiPY6Qdenfks,
     &dxiPY6Qdenfkssis,dadx,dady,dbdx,dbdy,dcdx,dcdy,denfksdx,denfksdy,
     &denfkssisdx,denfkssisdy,dmomfkssisdx,dmomfkssisdy,dzPY6Qdx,dzPY6Qdy,
     &dxiPY6Qdx,dxiPY6Qdy,beta1

      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      parameter(tiny=1.d-5)
c
      if(ileg.gt.4.or.ileg.le.0)then
         write(*,*)'error #1 in xjacPY6Q_xiztoxy, unknown ileg ',ileg
         stop
      endif
c incoming parton #1 (left)
      if(ileg.eq.1)then
         tmp=-s*(1-x)/2.d0
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
         tmp=-s*(1-x)/2.d0
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
         w1=-xq1q+xq2q-xtk
         w2=-xq2q+xq1q-xuk
         if(1-x.lt.tiny)then
            beta1=sqrt(1-4*s*xm12/(s+xm12-xm22)**2)
            tmp=s*(1-x)*beta1/2
         else
            afun=sqrt(s)*(1-x)*(xm12-xm22+s*x)*yj
            bfun=s*( (1+x)**2*(xm12**2+(xm22-s*x)**2-
     &                  xm12*(2*xm22+s*(1+x**2)))+
     &                  xm12*s*(1-x**2)**2*yj**2 )
            cfun=s*(-(1+x)**2+(1-x)**2*yj**2)
            signfac=get_sign(w1,w2,s,x,yj,xm12,xm22,afun,bfun,cfun)
            en_fks=sqrt(s)*(1-x)/2.d0
            mom_fks_sister  =(afun+signfac*sqrt(bfun))/cfun
            mom_fks_sister_m=(afun-signfac*sqrt(bfun))/cfun
            if(min(abs(mom_fks_sister-veckn_ev),abs(mom_fks_sister_m-veckn_ev))
     &         .eq.abs(mom_fks_sister_m-veckn_ev))then
              write(*,*)'Get_sign assigned wrong solution'
              signfac=-signfac
            endif
            if(min(abs(mom_fks_sister-veckn_ev),abs(mom_fks_sister_m-veckn_ev))/
     &             abs(veckn_ev).ge.tiny)then
              write(*,*)'Numerical imprecision #1 in xjacHW6_xiztoxy'
            elseif(min(abs(mom_fks_sister-veckn_ev),abs(mom_fks_sister_m-veckn_ev))/
     &                 abs(veckn_ev).ge.1.d-3)then
              write(*,*)'Fatal imprecision #1 in xjacHW6_xiztoxy'
              stop
            endif
            mom_fks_sister=veckn_ev
            en_fks_sister=sqrt(mom_fks_sister**2+xm12)
            beta=sqrt(1-xm12/en_fks_sister**2)
            dzPY6Qdenfks=-en_fks_sister/(en_fks + en_fks_sister)**2
            dzPY6Qdenfkssis=en_fks/(en_fks + en_fks_sister)**2
            dxiPY6Qdenfks=2*en_fks_sister*(1-beta*yj)
            dxiPY6Qdenfkssis=2*en_fks*(1-yj/beta)
            dadx=sqrt(s)*yj*(xm22-xm12+s*(1-2*x))
            dady=sqrt(s)*(1-x)*(xm12-xm22+s*x)
            dbdx=2*s*(1+x)*( xm12**2+(xm22-s*x)*(xm22-s*(1+2*x))
     &              -xm12*(2*xm22+s*(1+x+2*(x**2)+2*(1-x)*x*(yj**2))) )
            dbdy=2*xm12*s**2*(1-x**2)**2*yj
            dcdx=-2*s*(1+x+(yj**2)*(1-x))
            dcdy=2*s*(1-x)**2*yj
            denfksdx=-sqrt(s)/2.d0
            denfksdy=0.d0
            dmomfkssisdx=( dadx+signfac*dbdx/(2*sqrt(bfun))-
     &                        dcdx*mom_fks_sister )/cfun
            dmomfkssisdy=( dady+signfac*dbdy/(2*sqrt(bfun))-
     &                        dcdy*mom_fks_sister )/cfun
            denfkssisdx=(mom_fks_sister/en_fks_sister)*dmomfkssisdx
            denfkssisdy=(mom_fks_sister/en_fks_sister)*dmomfkssisdy
            dzPY6Qdx=dzPY6Qdenfks*denfksdx+dzPY6Qdenfkssis*denfkssisdx
            dzPY6Qdy=dzPY6Qdenfks*denfksdy+dzPY6Qdenfkssis*denfkssisdy
            dxiPY6Qdx=dxiPY6Qdenfks*denfksdx+dxiPY6Qdenfkssis*denfkssisdx
            dxiPY6Qdy=dxiPY6Qdenfks*denfksdy+dxiPY6Qdenfkssis*denfkssisdy-
     &                 2*beta*en_fks*en_fks_sister
            tmp=dzPY6Qdx*dxiPY6Qdy-dzPY6Qdy*dxiPY6Qdx
         endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            tmp=s*(1-x)/2
         elseif(1-yj.lt.tiny)then
            tmp=-s*(1-x)*(s*x-xm12)/( 2*(s-xm12) )
         else
            tmp=( 2*s*(1-x)*(s*x-xm12) )/
     &           ( (x*(yj-1)-(yj+1))*(s*((x-2)*x*(yj-1)+yj+1)-2*xm12) )
         endif
      endif

      xjacPY6Q_xiztoxy=abs(tmp)

      return
      end



c Begin of PY6PT stuff
      function zPY6PT(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Returns PYTHIA energy shower variable
      implicit none
      integer ileg
      real*8 zPY6PT,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny
      parameter(tiny=1.d-5)
c
      if(ileg.gt.4.or.ileg.le.0)then
         write(*,*)'error #1 in zPY6PT, unknown ileg ',ileg
         stop
      endif

c incoming parton #1 (left)
      if(ileg.eq.1)then
         zPY6PT=x
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
         zPY6PT=x
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
         else
         endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
         elseif(1-yj.lt.tiny)then
         else
c z here is the energy sharing in the radiator + recoiler cm
         endif
      endif

      return
      end



      function xiPY6PT(ileg,xm12,xm22,s,x,yi,yj,
     &                                  xtk,xuk,xq1q,xq2q)
c Returns PYTHIA evolution shower variable
      implicit none
      integer ileg
      real*8 xiPY6PT,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,z,zPY6PT,
     &en_fks,en_fks_sister,mom_fks_sister,xt,tiny,w1,w2,beta
      parameter(tiny=1.d-5)
c
      if(ileg.gt.4.or.ileg.le.0)then
         write(*,*)'error #1 in xiPY6PT, unknown ileg ',ileg
         stop
      endif

c incoming parton #1 (left)
      if(ileg.eq.1)then
         xiPY6PT=(s*(1-x)*(1-yi)/2.d0)*(1-x)
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
         xiPY6PT=(s*(1-x)*(1-yi)/2.d0)*(1-x)
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
         else
         endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
c it is expected to be 0
         elseif(1-yj.le.tiny)then
c it is expected to be 0
         else
c     z=zPY6PT(.....)
            en_fks=sqrt(s)*(1-x)/2.d0
            en_fks_sister=sqrt(s)*(x-xm12/s)/(2-(1-x)*(1-yj))
            xt=2*en_fks*en_fks_sister*(1-yj)
            xiPY6PT=z*(1-z)*xt
         endif
      endif

      return
      end




      function xjacPY6PT_xiztoxy(ileg,xm12,xm22,s,x,yi,yj,
     &                                            xtk,xuk,xq1q,xq2q)
c Returns PYTHIA jacobian |d(xi_PY6PT,z_PY6PT)/d(x,y)|
      implicit none
      integer ileg
      real*8 xjacPY6PT_xiztoxy,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,
     &z,zPY6PT,en_fks,en_fks_sister,mom_fks_sister,xt,tiny,tmp,w1,w2,
     &afun,bfun,cfun,signfac,get_sign,beta,dzPY6PTdenfks,dzPY6PTdenfkssis,dxiPY6PTdenfks,
     &dxiPY6PTdenfkssis,dadx,dady,dbdx,dbdy,dcdx,dcdy,denfksdx,denfksdy,
     &denfkssisdx,denfkssisdy,dmomfkssisdx,dmomfkssisdy,dzPY6PTdx,dzPY6PTdy,
     &dxiPY6PTdx,dxiPY6PTdy
      parameter(tiny=1.d-5)
c
      if(ileg.gt.4.or.ileg.le.0)then
         write(*,*)'error #1 in xjacPY6PT_xiztoxy, unknown ileg ',ileg
         stop
      endif
c incoming parton #1 (left)
      if(ileg.eq.1)then
         tmp=-s*(1-x)**2/2.d0
c incoming parton #2 (right)
      elseif(ileg.eq.2)then
         tmp=-s*(1-x)**2/2.d0
c outgoing parton #3 (massive)
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
         else
         endif
c outgoing parton #4 (massless)
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
         elseif(1-yj.le.tiny)then
         else
         endif
      endif

      xjacPY6PT_xiztoxy=abs(tmp)

      return
      end







      function emscafun(x,alpha)
      implicit none
      real*8 emscafun,x,alpha
c
      if(x.lt.0.d0.or.x.gt.1.d0)then
        write(6,*)'Fatal error in emscafun'
        stop
      endif
      emscafun=x**(2*alpha)/(x**(2*alpha)+(1-x)**(2*alpha))
      return
      end


      function emscainv(r,alpha)
c Inverse of emscafun; implemented only for alpha=1 for the moment
      implicit none
      real*8 emscainv,r,alpha
c
      if(r.lt.0.d0.or.r.gt.1.d0.or.alpha.ne.1.d0)then
        write(6,*)'Fatal error in emscafun'
        stop
      endif
      if(r.ne.0.5d0)then
        emscainv=(r-sqrt(r-r**2))/(2*r-1)
      else
        emscainv=0.5d0
      endif
      return
      end
c
c
c No-emission probability stuff
c
c 
      function bogus_probne_fun(qMC)
      implicit none
      double precision bogus_probne_fun,qMC
      double precision one,two,x,tmp,emscafun
      parameter (one=1.d0)
      parameter (two=2.d0)
      integer itype
      data itype/2/
c
      if(itype.eq.1)then
c Theta function
        if(qMC.le.2.d0)then
          tmp=0.d0
        else
          tmp=1.d0
        endif
      elseif(itype.eq.2)then
c Smooth function
        if(qMC.le.0.5d0)then
          tmp=0.d0
        elseif(qMC.le.10.d0)then
          x=(10.d0-qMC)/(10.d0-0.5d0)
          tmp=1-emscafun(x,two)
        else
          tmp=1.d0
        endif
      else
        write(*,*)'Error in bogus_probne_fun: unknown option',itype
        stop
      endif
      bogus_probne_fun=tmp
      return
      end



      function get_sign(w1,w2,s,x,yj,xm12,xm22,afun,bfun,cfun)
c establish if en_fks_sister in this event corresponds
c to the solution with the plus or with the minus
      implicit none
      integer i
      double precision get_sign,w1,w2,s,x,yj,afun,bfun,cfun,
     &mom_fks_sister_p,mom_fks_sister_m,dot,tiny,tiny2,xm12,xm22,xw1p,xw2p,xw1m,
     &xw2m,r1p,r2p,r1m,r2m,sign,
     &xq(0:3),xk1m(0:3),xk1p(0:3),xk2m(0:3),xk2p(0:3),xk3(0:3)
      logical iregp,iregm
      parameter (tiny=1.d-5)
c
      mom_fks_sister_p=(afun+sqrt(bfun))/cfun
      mom_fks_sister_m=(afun-sqrt(bfun))/cfun

      do i=0,3
         xq(i)=0.d0
         xk1m(i)=0.d0
         xk1p(i)=0.d0
         xk2m(i)=0.d0
         xk2p(i)=0.d0
         xk3(i)=0.d0
      enddo
c
      xq(0)=sqrt(s)
      xk3(2)=(sqrt(s)*(1-x)/2)*sqrt(1.d0-yj**2)
      xk3(3)=(sqrt(s)*(1-x)/2)*yj
      xk3(0)=sqrt(s)*(1-x)/2
      xk1p(3)=mom_fks_sister_p
      xk1m(3)=mom_fks_sister_m    
      xk1p(0)=sqrt(xm12+mom_fks_sister_p**2)
      xk1m(0)=sqrt(xm12+mom_fks_sister_m**2)
      do i=0,3
         xk2p(i)=xq(i)-xk1p(i)-xk3(i)
         xk2m(i)=xq(i)-xk1m(i)-xk3(i)
      enddo
      xw1p=2*dot(xk1p,xk3)
      xw2p=2*dot(xk2p,xk3)
      xw1m=2*dot(xk1m,xk3)
      xw2m=2*dot(xk2m,xk3)
      if(abs(max(w1,xw1p)).ge.1.d0)then
         r1p=abs(xw1p-w1)/abs(max(w1,xw1p))
         tiny2=tiny
      else
         r1p=abs(xw1p-w1)
         tiny2=tiny*1.d2
      endif
      if(abs(max(w2,xw2p)).ge.1.d0)then
         r2p=abs(xw2p-w2)/abs(max(w2,xw2p))
         tiny2=tiny
      else
         r2p=abs(xw2p-w2)
         tiny2=tiny*1.d2
      endif
      if(abs(max(w1,xw1m)).ge.1.d0)then
         r1m=abs(xw1m-w1)/abs(max(w1,xw1m))
         tiny2=tiny
      else
         r1m=abs(xw1m-w1)
         tiny2=tiny*1.d2
      endif
      if(abs(max(w2,xw2m)).ge.1.d0)then
         r2m=abs(xw2m-w2)/abs(max(w2,xw2m))
         tiny2=tiny
      else
         r2m=abs(xw2m-w2)
         tiny2=tiny*1.d2
      endif
      if(r1p.le.tiny2.and.r2p.le.tiny2)then
         iregp=.true.
         iregm=.false.
      elseif(r1m.le.tiny2.and.r2m.le.tiny2)then
         iregm=.true.
         iregp=.false.
      else
         write(*,*)'imprecision in get_sign',min(r1p,r1m),
     &                                       min(r2p,r2m),tiny2
      endif 
c
      if(iregp)sign=1.d0
      if(iregm)sign=-1.d0
      get_sign=sign

      return
      end


      function get_angle(p1,p2)
      implicit none
      double precision get_angle,p1(0:3),p2(0:3)
      double precision tiny,mod1,mod2,cosine
      parameter (tiny=1.d-5)
c
      mod1=sqrt(p1(1)**2+p1(2)**2+p1(3)**2)
      mod2=sqrt(p2(1)**2+p2(2)**2+p2(3)**2)

      if(mod1.eq.0.d0.or.mod2.eq.0.d0)then
        write(*,*)'Undefined angle in get_angle',mod1,mod2
        stop
      endif

      cosine=p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3)
      cosine=cosine/(mod1*mod2)

      if(abs(cosine).gt.1.d0+tiny)then
        write(*,*)'cosine larger than 1 in get_angle',cosine,p1,p2
        stop
      elseif(abs(cosine).ge.1.d0)then
        cosine=sign(1.d0,cosine)
      endif

      get_angle=acos(cosine)

      return
      end

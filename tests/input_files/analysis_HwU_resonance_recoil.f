c Differential validation for u b > d b e+ ve QCD=0 [QCD].
c Use together with resonance_recoil_cuts.f and explicitly labelled
c flavours. Jets use the same definition as the generation cuts.
c The leading positive-net-bottom jet reconstructs the top with e+ and
c nu_e. The recoil jet is the hardest separate positive-net-down jet.
c All spectra include underflow/overflow in their first/last bins.
c HwU combines correlated real/counterevent weights before estimating
c uncertainties. Do not call HwU_add_points inside this analysis.
      subroutine analysis_begin(nwgt,weights_info)
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      call HwU_inithist(nwgt,weights_info)
      call HwU_book(1,'total rate',1,0d0,1d0)
      call HwU_book(2,'total rate Born',1,0d0,1d0)
      call HwU_book(3,'m bjet e nu broad',40,0d0,400d0)
      call HwU_book(4,'m bjet e nu peak',50,150d0,200d0)
      call HwU_book(5,'m e nu',40,70d0,90d0)
      call HwU_book(6,'pt bjet',20,20d0,220d0)
      call HwU_book(7,'eta bjet',20,-5d0,5d0)
      call HwU_book(8,'pt recoil jet',20,20d0,220d0)
      call HwU_book(9,'eta recoil jet',24,-6d0,6d0)
      call HwU_book(10,'pt positron',25,0d0,200d0)
      call HwU_book(11,'eta positron',20,-5d0,5d0)
      call HwU_book(12,'pt bjet e nu',30,0d0,300d0)
      call HwU_book(13,'delta R bjet positron',24,0d0,6d0)
      call HwU_book(14,'jet multiplicity',2,1.5d0,3.5d0)
      end

      subroutine analysis_end(dummy)
      implicit none
      double precision dummy
      call HwU_write_file
      end

      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
      implicit none
      include 'nexternal.inc'
      include 'cuts.inc'
      double precision p(0:4,nexternal),wgts(*),pq(0:3,nexternal),
     $     pj(0:3,nexternal),pw(0:3),ptop(0:3),ptmax,ptnow,dphi,dr,
     $     rr_pt,rr_mass,rr_eta,pi
      parameter (pi=3.14159265358979323846d0)
      integer istatus(nexternal),ipdg(nexternal),ibody,
     $     jet(nexternal),flavour(nexternal,2),jf(nexternal,2),
     $     nq,njet,i,ib,ir,ilep,inu
      external rr_pt,rr_mass,rr_eta
      if(wgts(1).eq.0d0)return
      nq=0
      ilep=0
      inu=0
      do i=nincoming+1,nexternal
         if(istatus(i).ne.1.or.p(0,i).le.0d0)cycle
         if(ipdg(i).eq.-11)ilep=i
         if(ipdg(i).eq.12)inu=i
         if(abs(ipdg(i)).gt.5.and.ipdg(i).ne.21)cycle
         nq=nq+1
         pq(:,nq)=p(0:3,i)
         flavour(nq,:)=0
         if(abs(ipdg(i)).eq.5)flavour(nq,1)=sign(1,ipdg(i))
         if(abs(ipdg(i)).eq.1)flavour(nq,2)=sign(1,ipdg(i))
      enddo
      if(ilep.eq.0.or.inu.eq.0.or.nq.lt.2)then
         write(*,*) 'Invalid external state in recoil analysis'
         stop 1
      endif
      call amcatnlo_fastjetppgenkt_etamax_timed(pq,nq,jetradius,
     $     ptj,etaj,jetalgo,pj,njet,jet)
      jf=0
      do i=1,nq
         if(jet(i).gt.0.and.jet(i).le.njet)
     $        jf(jet(i),:)=jf(jet(i),:)+flavour(i,:)
      enddo
      ib=0
      ptmax=-1d0
      do i=1,njet
         if(jf(i,1).le.0)cycle
         ptnow=rr_pt(pj(:,i))
         if(ptnow.le.ptmax)cycle
         ib=i
         ptmax=ptnow
      enddo
      ir=0
      ptmax=-1d0
      do i=1,njet
         if(i.eq.ib)cycle
         if(jf(i,2).le.0)cycle
         ptnow=rr_pt(pj(:,i))
         if(ptnow.le.ptmax)cycle
         ir=i
         ptmax=ptnow
      enddo
      if(njet.lt.2.or.ib.eq.0.or.ir.eq.0)then
         write(*,*) 'Recoil analysis requires validation jet cuts'
         stop 1
      endif
      pw=p(0:3,ilep)+p(0:3,inu)
      ptop=pw+pj(:,ib)
      dphi=abs(atan2(pj(2,ib),pj(1,ib))-
     $          atan2(p(2,ilep),p(1,ilep)))
      dphi=min(dphi,2d0*pi-dphi)
      dr=sqrt((rr_eta(pj(:,ib))-rr_eta(p(0:3,ilep)))**2+
     $        dphi**2)
      call HwU_fill(1,0.5d0,wgts)
      if(ibody.eq.3)call HwU_fill(2,0.5d0,wgts)
      call rr_fill(3,rr_mass(ptop),0d0,400d0,wgts)
      call rr_fill(4,rr_mass(ptop),150d0,200d0,wgts)
      call rr_fill(5,rr_mass(pw),70d0,90d0,wgts)
      call rr_fill(6,rr_pt(pj(:,ib)),20d0,220d0,wgts)
      call rr_fill(7,rr_eta(pj(:,ib)),-5d0,5d0,wgts)
      call rr_fill(8,rr_pt(pj(:,ir)),20d0,220d0,wgts)
      call rr_fill(9,rr_eta(pj(:,ir)),-6d0,6d0,wgts)
      call rr_fill(10,rr_pt(p(0:3,ilep)),0d0,200d0,wgts)
      call rr_fill(11,rr_eta(p(0:3,ilep)),-5d0,5d0,wgts)
      call rr_fill(12,rr_pt(ptop),0d0,300d0,wgts)
      call rr_fill(13,dr,0d0,6d0,wgts)
      call rr_fill(14,dble(njet),1.5d0,3.5d0,wgts)
      end

      subroutine rr_fill(label,x,xmin,xmax,wgts)
      implicit none
      integer label
      double precision x,xmin,xmax,wgts(*),value,eps
c A fixed small offset avoids the excluded upper edge in HwU_fill.
      eps=1d-10*(xmax-xmin)
      value=max(xmin+eps,min(xmax-eps,x))
      call HwU_fill(label,value,wgts)
      end

      double precision function rr_pt(p)
      implicit none
      double precision p(0:3)
      rr_pt=sqrt(p(1)**2+p(2)**2)
      end

      double precision function rr_mass(p)
      implicit none
      double precision p(0:3)
      rr_mass=sqrt(max(0d0,p(0)**2-sum(p(1:3)**2)))
      end

      double precision function rr_eta(p)
      implicit none
      double precision p(0:3),pt
      pt=sqrt(p(1)**2+p(2)**2)
      rr_eta=asinh(p(3)/max(pt,1d-100))
      end

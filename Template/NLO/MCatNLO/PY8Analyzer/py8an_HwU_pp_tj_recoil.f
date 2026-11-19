c Stable-top, parton-level analysis for the standard MG5_aMC/Pythia8 driver.
c anti-kt R=0.4; final quarks/gluons only, excluding the stable top.
c Jets: pT>20 GeV, |y|<4.5; leading jet: pT>30 GeV.
c GAP additionally requires |y_t-y_j1|>3 and m(t,j1)>500 GeV.
c j1 is the leading accepted jet, with no flavour or ancestry assignment.
c j2 is the VBF third-jet analogue. zstar is oriented towards j1:
c sign(y_j1-y_t)*(y_j2-(y_t+y_j1)/2)/|y_t-y_j1|, filled for gap>1.
c Histogram errors retain squared signed event weights. Call add_points
c once per event, including events failing the analysis selections.

      module st_recoil_settings
      implicit none
      integer,parameter :: st_maxpart=4000,st_maxwgt=1024
      double precision,parameter :: st_radius=0.4d0,st_ptmin=20d0,
     $     st_tagpt=30d0,st_ymax=4.5d0,st_gapmin=3d0,
     $     st_mmin=500d0,st_pi=3.14159265358979323846d0
      integer :: st_nweights=0,st_nevents=0
      contains

      double precision function st_pt(p)
      double precision,intent(in) :: p(4)
      st_pt=sqrt(sum(p(1:2)**2))
      end function

      double precision function st_rap(p)
      double precision,intent(in) :: p(4)
      if(p(4).gt.abs(p(3)))then
         st_rap=0.5d0*log((p(4)+p(3))/(p(4)-p(3)))
      else
         st_rap=sign(20d0,p(3))
      endif
      end function

      double precision function st_eta(p)
      double precision,intent(in) :: p(4)
      if(st_pt(p).gt.0d0)then
         st_eta=asinh(p(3)/st_pt(p))
      else
         st_eta=sign(20d0,p(3))
      endif
      end function

      double precision function st_mass(p)
      double precision,intent(in) :: p(4)
      st_mass=sqrt(max(0d0,p(4)**2-sum(p(1:3)**2)))
      end function

      double precision function st_dphi(p,q)
      double precision,intent(in) :: p(4),q(4)
      double precision den
      den=st_pt(p)*st_pt(q)
      st_dphi=0d0
      if(den.gt.0d0)st_dphi=acos(max(-1d0,min(1d0,
     $     sum(p(1:2)*q(1:2))/den)))
      end function

      subroutine st_book_selection(l,suffix)
      integer,intent(in) :: l
      character(*),intent(in) :: suffix
      call HwU_book(l+1,'fiducial rate [pb]'//suffix,1,0d0,1d0)
      call HwU_book(l+2,'top pT [GeV]'//suffix,60,0d0,600d0)
      call HwU_book(l+3,'top y'//suffix,50,-5d0,5d0)
      call HwU_book(l+4,'top abs y'//suffix,25,0d0,5d0)
      call HwU_book(l+5,'j1 pT [GeV]'//suffix,60,0d0,600d0)
      call HwU_book(l+6,'j1 y'//suffix,50,-5d0,5d0)
      call HwU_book(l+7,'j1 eta'//suffix,50,-5d0,5d0)
      call HwU_book(l+8,'j1 abs eta'//suffix,25,0d0,5d0)
      call HwU_book(l+9,'abs dy(t,j1)'//suffix,40,0d0,10d0)
      call HwU_book(l+10,'dy(t,j1)'//suffix,50,-10d0,10d0)
      call HwU_book(l+11,'dphi(t,j1) [rad]'//suffix,
     $     32,0d0,st_pi+1d-10)
      call HwU_book(l+12,'pT(t+j1) [GeV]'//suffix,60,0d0,300d0)
      call HwU_book(l+13,'pT(t+j1) low [GeV]'//suffix,50,0d0,50d0)
      call HwU_book(l+14,'m(t,j1) [GeV]'//suffix,60,150d0,1650d0)
      call HwU_book(l+15,'y(t+j1)'//suffix,50,-5d0,5d0)
      call HwU_book(l+16,'Njets20 (last >=8)'//suffix,9,-0.5d0,8.5d0)
      call HwU_book(l+17,'j2 pT [GeV]'//suffix,50,0d0,300d0)
      call HwU_book(l+18,'j2 y'//suffix,50,-5d0,5d0)
      call HwU_book(l+19,'j2 eta'//suffix,50,-5d0,5d0)
      call HwU_book(l+20,'j2 y minus midpoint'//suffix,50,-5d0,5d0)
      call HwU_book(l+21,'j2 zstar (dy>1)'//suffix,40,-2d0,2d0)
      call HwU_book(l+22,'j2 zstar pT>30 (dy>1)'//suffix,
     $     40,-2d0,2d0)
      call HwU_book(l+23,'j2 signed gap distance'//suffix,
     $     50,-5d0,5d0)
      call HwU_book(l+24,'j2 dy from j1 (oriented)'//suffix,
     $     50,-6d0,6d0)
      call HwU_book(l+25,'j2 dy from top (oriented)'//suffix,
     $     50,-6d0,6d0)
      call HwU_book(l+26,'pT(t+j1+j2) [GeV]'//suffix,
     $     60,0d0,300d0)
      call HwU_book(l+27,'Ngap jets20 (last >=5)'//suffix,
     $     6,-0.5d0,5.5d0)
      call HwU_book(l+28,'hardest gap jet pT [GeV]'//suffix,
     $     50,0d0,200d0)
      call HwU_book(l+29,'gap jets HT [GeV]'//suffix,50,0d0,300d0)
      call HwU_book(l+30,'midpoint extra jets HT [GeV]'//suffix,
     $     50,0d0,300d0)
c These are cumulative cross sections, not differential distributions.
c Divide the two scans bin by bin to obtain the gap-veto efficiency.
      call HwU_book(l+31,'gap veto survived Q0 [GeV]'//suffix,
     $     40,20d0,100d0)
      call HwU_book(l+32,'gap veto reference Q0 [GeV]'//suffix,
     $     40,20d0,100d0)
      call HwU_book(l+33,'at least two jets rate [pb]'//suffix,
     $     1,0d0,1d0)
      end subroutine
      end module

      subroutine RCLOS()
      end

      subroutine PYABEG(nnn,wwwi)
      use st_recoil_settings
      use HwU_wgts_info_len
      include 'HEPMC_ST_RECOIL.INC'
      integer nnn
      character*(wgts_info_len) wwwi(*)
      if(nnn.lt.1.or.nnn.gt.st_maxwgt)stop 'Invalid analysis weights'
c Match HepMC's array strides to this analysis's larger Fortran common.
      call hepmc_set_max_number_entries(NMXHEP)
      st_nweights=nnn
      st_nevents=0
      call HwU_inithist(nnn,wwwi)
      call set_error_estimation(0)
      call HwU_book(1,'total rate [pb]|T@ALL',1,0d0,1d0)
      call HwU_book(2,'top pT [GeV]|T@ALL',60,0d0,600d0)
      call HwU_book(3,'top y|T@ALL',50,-5d0,5d0)
      call HwU_book(4,'top abs y|T@ALL',25,0d0,5d0)
      call HwU_book(5,'Njets20 (last >=8)|T@ALL',9,-0.5d0,8.5d0)
      call HwU_book(6,'jets HT [GeV]|T@ALL',50,0d0,1000d0)
      call HwU_book(7,'top plus jets HT [GeV]|T@ALL',50,0d0,1200d0)
      call HwU_book(8,'top pT high [GeV]|T@ALL',50,0d0,1500d0)
      call HwU_book(9,'j1 pT [GeV]|T@ALL',60,0d0,600d0)
      call st_book_selection(20,'|T@TAG')
      call st_book_selection(60,'|T@GAP')
      end

      subroutine PYAEND(ievttot)
      use st_recoil_settings
      implicit none
      double precision ievttot
      if(st_nevents.eq.0)return
c Use the number of analysis calls, not the HepMC event serial number.
      call finalize_histograms(st_nevents)
      open(unit=99,file='MADatNLO.HwU',status='replace')
      call HwU_output(99,ievttot)
      close(99)
      end

      subroutine PYANAL(nnn,xww)
      use st_recoil_settings
      include 'HEPMC_ST_RECOIL.INC'
      integer nnn,i,id,ntop,ntrack,njet,nacc,j,l,selection,ngap
      integer assignment(st_maxpart)
      double precision xww(*),weights(st_maxwgt),tracks(4,st_maxpart),
     $     jets(4,st_maxpart),accepted(4,st_maxpart),top(4),pair(4),
     $     yjet(st_maxpart),ptjet(st_maxpart),pttop,ytop,ht,gap,
     $     midpoint,orientation,mtj,rel,zstar,nearest,htgap,htmid,
     $     maxgap,q0
      logical separated,inside

      if(nnn.ne.st_nweights)stop 'Analysis weight count changed'
      if(NHEP.ge.NMXHEP)stop 'HEPEVT capacity reached: refusing truncated event'
      st_nevents=st_nevents+1
      if(xww(1).eq.0d0)then
         if(any(xww(1:nnn).ne.0d0))stop 'Nonzero variation at zero weight'
         goto 999
      endif
      weights(1:nnn)=EVWGT*xww(1:nnn)/xww(1)
      ntop=0
      ntrack=0
      top=0d0
      do i=1,NHEP
         if(ISTHEP(i).ne.1)cycle
         id=abs(IDHEP(i))
         if(id.eq.6)then
            ntop=ntop+1
            top=PHEP(1:4,i)
         elseif((id.ge.1.and.id.le.5).or.id.eq.21)then
            if(ntrack.eq.st_maxpart)stop 'Too many final partons'
            ntrack=ntrack+1
            tracks(:,ntrack)=PHEP(1:4,i)
         elseif(id.ge.100.or.id.eq.22.or.
     $          (id.ge.11.and.id.le.16))then
            stop 'Non-partonic final state: check shower/decay settings'
         endif
      enddo
      if(ntop.ne.1)stop 'Expected exactly one stable final top'
      njet=0
      if(ntrack.gt.0)call fastjetppgenkt(tracks,ntrack,st_radius,
     $     st_ptmin,-1d0,jets,njet,assignment)
      nacc=0
      ht=0d0
      do i=1,njet
         if(abs(st_rap(jets(:,i))).ge.st_ymax)cycle
         nacc=nacc+1
         accepted(:,nacc)=jets(:,i)
         ptjet(nacc)=st_pt(jets(:,i))
         yjet(nacc)=st_rap(jets(:,i))
         ht=ht+ptjet(nacc)
         if(nacc.gt.1)then
            if(ptjet(nacc).gt.ptjet(nacc-1)+1d-8)
     $           stop 'Jet clustering did not return pT ordering'
         endif
      enddo
      pttop=st_pt(top)
      ytop=st_rap(top)
      call HwU_fill(1,0.5d0,weights)
      call HwU_fill(2,pttop,weights)
      call HwU_fill(3,ytop,weights)
      call HwU_fill(4,abs(ytop),weights)
      call HwU_fill(5,dble(min(nacc,8)),weights)
      call HwU_fill(6,ht,weights)
      call HwU_fill(7,pttop+ht,weights)
      call HwU_fill(8,pttop,weights)
      if(nacc.eq.0)goto 999
      call HwU_fill(9,ptjet(1),weights)
      if(ptjet(1).le.st_tagpt)goto 999
      pair=top+accepted(:,1)
      mtj=st_mass(pair)
      gap=abs(yjet(1)-ytop)
      midpoint=(yjet(1)+ytop)/2d0
      orientation=sign(1d0,yjet(1)-ytop)
      separated=gap.gt.st_gapmin.and.mtj.gt.st_mmin
      ngap=0
      htgap=0d0
      htmid=0d0
      maxgap=0d0
      do j=2,nacc
         inside=yjet(j).gt.min(ytop,yjet(1)).and.
     $        yjet(j).lt.max(ytop,yjet(1))
         if(inside)then
            ngap=ngap+1
            htgap=htgap+ptjet(j)
            maxgap=max(maxgap,ptjet(j))
         endif
         if(abs(yjet(j)-midpoint).lt.0.5d0)htmid=htmid+ptjet(j)
      enddo
      do selection=1,2
         if(selection.eq.2.and..not.separated)cycle
         l=20+40*(selection-1)
         call HwU_fill(l+1,0.5d0,weights)
         call HwU_fill(l+2,pttop,weights)
         call HwU_fill(l+3,ytop,weights)
         call HwU_fill(l+4,abs(ytop),weights)
         call HwU_fill(l+5,ptjet(1),weights)
         call HwU_fill(l+6,yjet(1),weights)
         call HwU_fill(l+7,st_eta(accepted(:,1)),weights)
         call HwU_fill(l+8,abs(st_eta(accepted(:,1))),weights)
         call HwU_fill(l+9,gap,weights)
         call HwU_fill(l+10,ytop-yjet(1),weights)
         call HwU_fill(l+11,st_dphi(top,accepted(:,1)),weights)
         call HwU_fill(l+12,st_pt(pair),weights)
         call HwU_fill(l+13,st_pt(pair),weights)
         call HwU_fill(l+14,mtj,weights)
         call HwU_fill(l+15,st_rap(pair),weights)
         call HwU_fill(l+16,dble(min(nacc,8)),weights)
         call HwU_fill(l+27,dble(min(ngap,5)),weights)
         call HwU_fill(l+28,maxgap,weights)
         call HwU_fill(l+29,htgap,weights)
         call HwU_fill(l+30,htmid,weights)
         do i=1,40
            q0=20d0+2d0*(dble(i)-0.5d0)
            call HwU_fill(l+32,q0,weights)
            if(maxgap.lt.q0)call HwU_fill(l+31,q0,weights)
         enddo
         if(nacc.lt.2)cycle
         call HwU_fill(l+33,0.5d0,weights)
         rel=yjet(2)-midpoint
         nearest=min(abs(yjet(2)-ytop),abs(yjet(2)-yjet(1)))
         inside=yjet(2).gt.min(ytop,yjet(1)).and.
     $        yjet(2).lt.max(ytop,yjet(1))
         if(.not.inside)nearest=-nearest
         call HwU_fill(l+17,ptjet(2),weights)
         call HwU_fill(l+18,yjet(2),weights)
         call HwU_fill(l+19,st_eta(accepted(:,2)),weights)
         call HwU_fill(l+20,rel,weights)
         if(gap.gt.1d0)then
            zstar=orientation*rel/gap
            call HwU_fill(l+21,zstar,weights)
            if(ptjet(2).gt.30d0)call HwU_fill(l+22,zstar,weights)
         endif
         call HwU_fill(l+23,nearest,weights)
         call HwU_fill(l+24,orientation*(yjet(2)-yjet(1)),weights)
         call HwU_fill(l+25,orientation*(yjet(2)-ytop),weights)
         call HwU_fill(l+26,st_pt(pair+accepted(:,2)),weights)
      enddo
 999  call HwU_add_points
      end

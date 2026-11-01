c No-cut W+ b bbar campaign analysis; shared helper defines all observables.
      subroutine RCLOS()
      end

      subroutine PYABEG(nnn,wwwi)
      use HwU_wgts_info_len
      include 'HEPMC.INC'
      include 'reweight0.inc'
      integer nnn,i,nwgt_analysis
      integer,parameter :: nobs=19
      common/c_analysis/nwgt_analysis
      character*(wgts_info_len) wwwi(*),names(max_weight_shower)
      do i=1,nnn
         names(i)=wwwi(i)
      enddo
      nwgt_analysis=nnn
      call HwU_inithist(nnn,names)
      call set_error_estimation(0)
      call HwU_book(1,'total raw |T@NOCUT',1,0.00000000000000000d+00,1.00000000000000000d+00)
      call HwU_book(2,'primary scale 0-100 |T@NOCUT',100,0.00000000000000000d+00,1.00000000000000000d+02)
      call HwU_book(3,'primary log10 scale |T@NOCUT',140,-4.00000000000000000d+00,3.00000000000000000d+00)
      call HwU_book(4,'system pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(5,'Wbb invariant mass |T@NOCUT',100,8.00000000000000000d+01,2.08000000000000000d+03)
      call HwU_book(6,'Wbb rapidity |T@NOCUT',80,-8.00000000000000000d+00,8.00000000000000000d+00)
      call HwU_book(7,'bb pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(8,'bb invariant mass |T@NOCUT',100,0.00000000000000000d+00,1.00000000000000000d+03)
      call HwU_book(9,'bb rapidity |T@NOCUT',80,-8.00000000000000000d+00,8.00000000000000000d+00)
      call HwU_book(10,'W pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(11,'W rapidity |T@NOCUT',80,-8.00000000000000000d+00,8.00000000000000000d+00)
      call HwU_book(12,'bottom pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(13,'bottom rapidity |T@NOCUT',80,-8.00000000000000000d+00,8.00000000000000000d+00)
      call HwU_book(14,'antibottom pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(15,'antibottom rapidity |T@NOCUT',80,-8.00000000000000000d+00,8.00000000000000000d+00)
      call HwU_book(16,'bb delta phi |T@NOCUT',64,0.00000000000000000d+00,3.14159265358979312d+00)
      call HwU_book(17,'bb delta rapidity |T@NOCUT',100,-1.00000000000000000d+01,1.00000000000000000d+01)
      call HwU_book(18,'bb delta R |T@NOCUT',120,0.00000000000000000d+00,1.20000000000000000d+01)
      call HwU_book(19,'radiation jet1 pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(20,'radiation jet1 pT 0-50 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+01)
      call HwU_book(21,'radiation jet2 pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(22,'all jet1 pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      call HwU_book(23,'all jet2 pT 0-500 |T@NOCUT',100,0.00000000000000000d+00,5.00000000000000000d+02)
      open(unit=97,file='wbb_observables.bin',status='replace',
     $     form='unformatted',access='stream')
      write(97) nobs,nnn
      open(unit=96,file='wbb_weight_names.txt',status='replace')
      do i=1,nnn
         write(96,'(a)') trim(names(i))
      enddo
      close(96)
      end

      subroutine PYAEND(ievttot)
      include 'HEPMC.INC'
      double precision ievttot
      call finalize_histograms(nevhep)
      open(unit=99,file='MADatNLO.HwU',status='replace')
      call HwU_output(99,ievttot)
      close(99)
      flush(97)
      end

      subroutine PYANAL(nnn,xww)
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      include 'HEPMC.INC'
      include 'reweight0.inc'
      integer nnn,i,nfinal,nwgt_analysis,status
      integer,parameter :: nobs=19
      common/c_analysis/nwgt_analysis
      double precision xww(*),www(max_weight_shower)
      integer ids(nmxhep)
      double precision momenta(4,nmxhep),v(nobs)
      if(xww(1).eq.0d0)stop 'Zero nominal diagnostic weight'
      do i=1,nwgt_analysis
         www(i)=evwgt*xww(i)/xww(1)
      enddo
      nfinal=0
      do i=1,nhep
         if(isthep(i).ne.1)cycle
         nfinal=nfinal+1
         ids(nfinal)=idhep(i)
         momenta(:,nfinal)=phep(1:4,i)
      enddo
      call wbb_observables(nfinal,ids,momenta,v,status)
      if(status.ne.0)stop 'No-cut Wbb observables failed'
      write(97) v,www(1:nwgt_analysis)
      call HwU_fill(1,0.5d0,www)
      if(ieee_is_finite(v(1)))then
         call HwU_fill(2,v(1),www)
      endif
      if(ieee_is_finite(v(1)))then
         if(v(1).gt.0d0)call HwU_fill(3,log10(v(1)),www)
      endif
      if(ieee_is_finite(v(1)))then
         call HwU_fill(4,v(1),www)
      endif
      if(ieee_is_finite(v(2)))then
         call HwU_fill(5,v(2),www)
      endif
      if(ieee_is_finite(v(3)))then
         call HwU_fill(6,v(3),www)
      endif
      if(ieee_is_finite(v(4)))then
         call HwU_fill(7,v(4),www)
      endif
      if(ieee_is_finite(v(5)))then
         call HwU_fill(8,v(5),www)
      endif
      if(ieee_is_finite(v(6)))then
         call HwU_fill(9,v(6),www)
      endif
      if(ieee_is_finite(v(7)))then
         call HwU_fill(10,v(7),www)
      endif
      if(ieee_is_finite(v(8)))then
         call HwU_fill(11,v(8),www)
      endif
      if(ieee_is_finite(v(9)))then
         call HwU_fill(12,v(9),www)
      endif
      if(ieee_is_finite(v(10)))then
         call HwU_fill(13,v(10),www)
      endif
      if(ieee_is_finite(v(11)))then
         call HwU_fill(14,v(11),www)
      endif
      if(ieee_is_finite(v(12)))then
         call HwU_fill(15,v(12),www)
      endif
      if(ieee_is_finite(v(13)))then
         call HwU_fill(16,v(13),www)
      endif
      if(ieee_is_finite(v(14)))then
         call HwU_fill(17,v(14),www)
      endif
      if(ieee_is_finite(v(15)))then
         call HwU_fill(18,v(15),www)
      endif
      if(ieee_is_finite(v(16)))then
         call HwU_fill(19,v(16),www)
      endif
      if(ieee_is_finite(v(16)))then
         call HwU_fill(20,v(16),www)
      endif
      if(ieee_is_finite(v(17)))then
         call HwU_fill(21,v(17),www)
      endif
      if(ieee_is_finite(v(18)))then
         call HwU_fill(22,v(18),www)
      endif
      if(ieee_is_finite(v(19)))then
         call HwU_fill(23,v(19),www)
      endif
      call HwU_add_points
      end

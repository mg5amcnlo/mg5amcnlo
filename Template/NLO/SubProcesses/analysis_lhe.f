c LHE analysis routines used when having fo_analysis_format=lhe in the
c FO_analyse_card.
c that card also defines FO_LHE_WEIGHT which is a minimal weight to write the 
c event (event with lower weights will be un-weighted)

      subroutine analysis_begin(nwgt,weights_info)
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer nwgts,nevents
      double precision sum_of_wgts
      common/to_lhe_analysis/sum_of_wgts,nwgts,nevents
      logical lopen
      nwgts=nwgt
      sum_of_wgts=0d0
      nevents=0
      inquire(41,OPENED=lopen) ! in case run is reset due to 2 bad iterations...
      if (lopen) close(41)
      open(41,file= 'events.lhe',status='UNKNOWN')
      write (41,'(a)') '<eventgroup>'
      end

      subroutine analysis_end(xnorm)
      implicit none
      double precision xnorm
      integer nwgts,nevents
      double precision sum_of_wgts
      common/to_lhe_analysis/sum_of_wgts,nwgts,nevents
      logical lopen
      integer                                   npoints
      double precision            cross_section
      common /for_FixedOrder_lhe/ cross_section,npoints
      character*10 MonteCarlo
      inquire(41,OPENED=lopen)  ! safety
      if (lopen) then
         backspace(41)          ! overwrite the final </eventgroup> tag
         write (41,*) nevents,sum_of_wgts,cross_section
         close(41)
         open(41, file='header.txt')
         MonteCarlo='FO'
         call write_lhef_header(41, 0, MonteCarlo)
         close(41)
      endif
      end

      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      integer nwgt
c      include 'genps.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody
      integer nwgts,nevents
      double precision sum_of_wgts
      common/to_lhe_analysis/sum_of_wgts,nwgts,nevents
c
      integer i,j,npart
      double precision ran2
      external ran2
      double precision R,twgt
      double precision zero
      integer izero 
      parameter (zero=0.d0)
      parameter (izero=0)
      include 'nFKSconfigs.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iSorH_lhe,ifks_lhe(fks_configs) ,jfks_lhe(fks_configs)
     &     ,fksfather_lhe(fks_configs) ,ipartner_lhe(fks_configs)
      double precision scale1_lhe(fks_configs),scale2_lhe(fks_configs)
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe, fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe
c
c     forbid unweighting close to the fks pole (should not happen but better safe than sorry)
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      double precision xi_i_fks_ev,y_ij_fks_ev
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
c
c Auxiliary quantities used when writing events
      integer kwgtinfo
      integer i_wgt, kk, ii, jj, n, nn
      character*140 buff
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     & MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,AQEDUP,AQCDUP,
     & PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
c maximum weight ratio for the partial unweighting
      integer                                   npoints
      double precision            cross_section
      common /for_FixedOrder_lhe/ cross_section,npoints
c --- do the partial unweighting (don't do it when to close to singular
c --- region)
      twgt=abs(cross_section)*FO_LHE_weight_ratio !/npoints
      if(abs(wgts(1)).lt.abs(twgt) .and. xi_i_fks_ev.gt.1d-3 .and.
     $     1d0-y_ij_fks_ev.gt.1d-2)then
         R = ran2()*abs(twgt)
         if (R.gt.abs(wgts(1)))then
            return
         else
            do i=2,nwgts
               wgts(i) = wgts(i)*abs(twgt/wgts(1))
            enddo
            wgts(1) = sign(twgt,wgts(1))
         endif
      endif
c --- accumulate the total weights of all the events in the event file
      sum_of_wgts=sum_of_wgts+wgts(1)

c --- fill the multi-weight common blocks with the scale and PDF
c --- variation weights
      i_wgt=1
      if (do_rwgt_scale) then
         do kk=1,dyn_scale(0)
            if (lscalevar(kk)) then
               do ii=1,nint(scalevarF(0))
                  do jj=1,nint(scalevarR(0))
                     i_wgt=i_wgt+1
                     wgtxsecmu(jj,ii,kk)= wgts(i_wgt)
                  enddo
               enddo
            else
               i_wgt=i_wgt+1
               wgtxsecmu(1,1,kk)= wgts(i_wgt)
            endif
         enddo
      endif
      if (do_rwgt_pdf) then
         do nn=1,lhaPDFid(0)
            if (lpdfvar(nn)) then
               do n=0,nmemPDF(nn)
                  i_wgt=i_wgt+1
                  wgtxsecPDF(n,nn) = wgts(i_wgt)
               enddo
            else
               i_wgt=i_wgt+1
               wgtxsecPDF(0,nn) = wgts(i_wgt)
            endif
         enddo
      endif

c --- prepare the event file info according to the LesHouches standard
      npart = nexternal
      do i=1,nexternal
        IDUP(i)= ipdg(i)
        ISTUP(i)= istatus(i)
        MOTHUP(1,i)=0
        MOTHUP(2,i)=0
        ICOLUP(1,i)=599
        ICOLUP(2,i)=599
        PUP(1,i)=p(1,i)
        PUP(2,i)=p(2,i)
        PUP(3,i)=p(3,i)
        PUP(4,i)=p(0,i)
        PUP(5,i)=p(4,i)
        VTIMUP(i)=0.d0
        SPINUP(i)=9
      enddo

c --- prepare the buffer information
      if(.not.doreweight)then
         write(buff,201)'#aMCatNLO',iSorH_lhe,ifks_lhe(nFKSprocess)
     &        ,jfks_lhe(nFKSprocess),fksfather_lhe(nFKSprocess)
     &        ,ipartner_lhe(nFKSprocess),scale1_lhe(nFKSprocess)
     &        ,scale2_lhe(nFKSprocess),izero,izero,izero,zero,zero
     &        ,zero,zero,ibody*1d0
      else
         if(iwgtinfo.ne.-5)then
            write(*,*)'Error in write_events_lhe'
            write(*,*)'  Inconsistency in reweight parameters'
            write(*,*)doreweight,iwgtinfo
            stop
         endif
         kwgtinfo= 9
         write(buff,201)'#aMCatNLO',iSorH_lhe,ifks_lhe(nFKSprocess)
     &        ,jfks_lhe(nFKSprocess),fksfather_lhe(nFKSprocess)
     &        ,ipartner_lhe(nFKSprocess),scale1_lhe(nFKSprocess)
     &        ,scale2_lhe(nFKSprocess),kwgtinfo,nexternal,iwgtnumpartn
     &        ,zero,zero,zero,zero,ibody*1d0
      endif

c --- write the event
      call write_lhef_event(41,
     &     npart,IDPRUP,wgts(1),0d0,0d0,0d0,
     &     IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)

 201  format(a9,1x,i1,4(1x,i2),2(1x,d14.8),2x,i2,2(1x,i2),5(1x,d14.8))
      end

c This we can use for the event grouping!      
      subroutine HwU_add_points
      implicit none
      integer nwgts,nevents
      double precision sum_of_wgts
      common/to_lhe_analysis/sum_of_wgts,nwgts,nevents
      nevents=nevents+1
      write (41,'(a)') '</eventgroup>'
      write (41,'(a)') '<eventgroup>'
      end

c Dummy routines
      subroutine HwU_accum_iter(ldummy,idummy)
      logical ldummy
      integer idummy
      end
      subroutine HwU_output(idummy,dummy)
      integer idummy
      double precision dummy
      write (*,*) 'HwU_output should not be called',idummy
      stop 1
      end
      subroutine addfil(string)
      character*(*) string
      end
      subroutine accum(ldummy)
      logical ldummy
      end

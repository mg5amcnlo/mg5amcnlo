c LHE analysis routines used when having fo_analysis_format=lhe in the
c FO_analyse_card.
c that card also defines FO_LHE_WEIGHT which is a minimal weight to write the 
c event (event with lower weights will be un-weighted)

      subroutine analysis_begin(nwgt,weights_info)
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      double precision test
      integer iteration
      common/to_lhe_analysis/iteration
      double precision cross, twgt
      logical opened_file
      common /FO_LHE_CROSS/ cross, twgt, opened_file
      iteration = 0
      cross = 0d0
      twgt = 0d0
      
c      call inihist
      open(41,file= 'events.lhe',status='UNKNOWN')
      opened_file=.true.
      end

      subroutine analysis_end(xnorm)
      implicit none
      double precision xnorm
      integer iteration
      common/to_lhe_analysis/iteration
      double precision cross, twgt
      logical opened_file
      common /FO_LHE_CROSS/ cross, twgt, opened_file
      if (.not.opened_file)then
           open(41,file= 'events.lhe',status='OLD',POSITION='APPEND')
      endif
      write(41,*) '!', xnorm
      close(41)
      opened_file = .false.
      end

      subroutine accum(to_accum)
      logical to_accum
      integer iteration
      common/to_lhe_analysis/iteration
      logical opened_file
      double precision cross, twgt
      common /FO_LHE_CROSS/ cross, twgt, opened_file
      integer itmax,ncall
      common/citmax/itmax,ncall
      twgt = cross/ncall/1d2
      iteration = iteration +1
      close(41)
      opened_file= .false.
      end

c dummy subroutine
      subroutine addfil(string)
      character*(*) string
      end

      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
      implicit none
      include 'nexternal.inc'
      include 'reweight.inc'
      include 'run.inc'
      integer nwgt,max_weight
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
c      include 'genps.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(max_weight)
      integer ibody
c
      integer i,j
c     fixed value
c      integer ickkw
      double precision shower_scale
c     to write_lhe event information
      integer ic(7,nexternal)
      integer npart
c
      double precision cross, twgt
      logical opened_file
      common /FO_LHE_CROSS/ cross, twgt, opened_file
c      double precision twgt, maxwgt,swgt(maxevents)
c      integer                             lun, nw
c      common/to_unwgt/twgt, maxwgt, swgt, lun, nw
      double precision ran2
      logical to_write
      double precision R

c Auxiliary quantities used when writing events
c      integer jwgtinfo,mexternal
c      common/cwgtaux0/jwgtinfo,mexternal

      integer i_wgt, kk, ii, jj, n,nn
c********************************************************************
c     Writes one event from data file #lun according to LesHouches
c     ic(1,*) = Particle ID
c     ic(2.*) = Mothup(1)
c     ic(3,*) = Mothup(2)
c     ic(4,*) = ICOLUP(1)
c     ic(5,*) = ICOLUP(2)
c     ic(6,*) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(7,*) = Helicity
c********************************************************************

      if (.not.opened_file)then
           open(41,file= 'events.lhe',status='replace',position='APPEND')
           opened_file = .true.
      endif

      if (twgt.eq.0d0)then
         cross = cross + wgts(1)
         return ! not written file for the first iteration
      elseif(abs(wgts(1)).lt.abs(twgt))then
         R = ran2()*abs(twgt)
         if (R.gt.abs(wgts(1)))then
            return
         endif
         do i =2, max_weight
            wgts(i) = wgts(i)*abs(twgt/wgts(1))
         enddo
         wgts(1) = sign(twgt,wgts(1))
      endif

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

      
      ickkw=0
      shower_scale = 0d0
      npart = nexternal

      do i=1,nexternal
         ic(1,i) = ipdg(i)
         ic(2,i) = 0 ! invalid to make PS to fail
         ic(3,i) = 0
         ic(4,i) = 501
         ic(5,i) = 501 ! invalid to make PS to fail
         ic(6,i) = istatus(i)
         ic(7,i) = 9
      enddo
         
      jwgtinfo = 9
      call write_events_lhe(p,wgts(1),ic,npart,41,shower_scale
     $     ,ickkw)

      end

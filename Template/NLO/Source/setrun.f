      subroutine setrun
c----------------------------------------------------------------------
c     Sets the run parameters reading them from the run_card.dat
c
c 1. PDF set
c 2. Collider parameters
c 3. cuts
c---------------------------------------------------------------------- 
      use extra_weights
      implicit none
      include 'PDF/pdf.inc'
      include 'run.inc'
      include 'alfas.inc'
      include 'MODEL/coupl.inc'
      integer k,i
      character*132 buff
      include 'cuts.inc'
c Les Houches init block (for the <init> info)
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
c
      integer          iseed
      common /to_seed/ iseed
      integer nevents
      character*7 event_norm
      common /event_normalisation/event_norm
      integer iappl
      common /for_applgrid/ iappl
      integer idum
      logical              fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps
c jet-rate distance. To be set to 1 for FxFx
      double precision D
      common/to_dj/D
c Include all the parameters set in the run_card.dat
      include 'run_card.inc'
c Change shower_MC string to upper case
      call to_upper(shower_MC)
c Determine if there is a need to do scale and/or PDF reweighting
      do_rwgt_scale=.false.
      do i=1,dyn_scale(0)
         if (lscalevar(i) .or. dyn_scale(0).gt.1) then
            do_rwgt_scale=.true.
            exit
         endif
      enddo
      do_rwgt_pdf=.false.
      do i=1,lhaPDFid(0)
         if (lpdfvar(i) .or. lhaPDFid(0).gt.1) then
            do_rwgt_pdf=.true.
            exit
         endif
      enddo
c Default scale and PDF choice used for the actual run
      dynamical_scale_choice=dyn_scale(1)
      lhaid=lhaPDFid(1)
c For backward compatibility
      scale = muR_ref_fixed
      q2fact(1) = muF1_ref_fixed**2      ! fact scale**2 for pdf1
      q2fact(2) = muF2_ref_fixed**2      ! fact scale**2 for pdf2     
      scalefact=muR_over_ref
      ellissextonfact=QES_over_ref
c check that the event normalization input is reasoble
      buff = event_norm 
      call case_trap2(buff) ! requires a string of length 20 at least
      event_norm=buff 
      if ( event_norm(1:7).ne.'average' .and.
     $     event_norm(1:3).ne.'sum' .and.
     $     event_norm(1:5).ne.'unity'.and.
     $     event_norm(1:4).ne.'bias')then
         write (*,*) 'Do not understand the event_norm parameter'/
     &        /' in the run_card.dat. Possible options are'/
     &        /' "average", "sum", "unity" or "bias". '/
     &        /'Current input is: ', event_norm
         stop 1
      endif
c Check parameters for FxFx/UNLOPS/NNLL-veto
      if ( ickkw.ne.0 .and. ickkw.ne.4 .and. ickkw.ne.3 .and.
     &     ickkw.ne.-1) then
         write (*,*) 'ickkw parameter not known. ickkw=',ickkw
         stop
      endif
      chcluster=.false.
      ktscheme=1
      xqcut=0d0
      xmtc=0d0
      D=1d0
c Set alphaS(mZ)      
      if(lpp(1).ne.0.or.lpp(2).ne.0) then
         write(*,*) 'A PDF is used, so alpha_s(MZ)'/
     &        /' is going to be modified'
          call setpara('param_card.dat')
          asmz=G**2/(16d0*atan(1d0))
          write(*,*) 'Old value of alpha_s from param_card: ',asmz
          call pdfwrap
          write(*,*) 'New value of alpha_s from PDF ',pdlabel,':',asmz
      else
          call setpara('param_card.dat')
          asmz=G**2/(16d0*atan(1d0))
          nloop=2
          pdlabel='none'
          write(*,*)
     &         'No PDF is used, alpha_s(MZ) from param_card is used'
          write(*,*) 'Value of alpha_s from param_card: ',asmz
          write(*,*) 'The default order of alpha_s running is fixed to '
     &         ,nloop
      endif
      if (nlo_ps) then
C Fill common block for Les Houches init info
         do i=1,2
            if(lpp(i).eq.1.or.lpp(i).eq.2) then
               idbmup(i)=2212
            elseif(lpp(i).eq.-1.or.lpp(i).eq.-2) then
               idbmup(i)=-2212
            elseif(lpp(i).eq.3) then
               idbmup(i)=11
            elseif(lpp(i).eq.-3) then
               idbmup(i)=-11
            elseif(lpp(i).eq.0) then
               open (unit=71,status='old',file='initial_states_map.dat')
               read (71,*,err=100)idum,idum,idbmup(1),idbmup(2)
               close (71)
            else
               idbmup(i)=lpp(i)
            endif
            ebmup(i)=ebeam(i)
         enddo
         call get_pdfup(pdlabel,pdfgup,pdfsup,lhaid)
      endif
c Fill the nmemPDF(i) array with the number of PDF error set. This we
c get from LHAPDF.
      if (lpdfvar(1) .and. (lpp(1).ne.0.or.lpp(2).ne.0) ) then
         call numberPDFm(1,nmemPDF(1))
         if (nmemPDF(1).eq.1) then
            nmemPDF(1)=0
            lpdfvar(1)=.False.
         endif
      else
         nmemPDF(1)=0
      endif
      return
 100  write(*,*) '"initial_states_map.dat" not found (or incorrect'/
     $     /' format) by "Source/setrun"'
      stop 1
      end

      subroutine get_pdfup(pdfin,pdfgup,pdfsup,lhaid)
C     Convert internal pdf name to LHAPDF number
      implicit none
      character*(*) pdfin
      integer mpdf
      integer npdfs,i,pdfgup(2),pdfsup(2),lhaid
      parameter (npdfs=16)
      character*7 pdflabs(npdfs)
      data pdflabs/ 'none', 'mrs02nl', 'mrs02nn', 'cteq4_m', 'cteq4_l',
     $     'cteq4_d', 'cteq5_m', 'cteq5_d', 'cteq5_l', 'cteq5m1',
     $     'cteq6_m', 'cteq6_l', 'cteq6l1', 'nn23lo', 'nn23lo1',
     $     'nn23nlo'/
      integer numspdf(npdfs)
      data numspdf/ 00000, 20250, 20270, 19150, 19170, 19160, 19050,
     $     19060, 19070, 19051, 10000, 10041, 10042, 246800, 247000,
     $     244800/
      if(pdfin.eq."lhapdf") then
         write(*,*)'using LHAPDF'
         do i=1,2
            pdfgup(i)=-1
            pdfsup(i)=lhaid
         enddo
         return
      endif
      mpdf=-1
      do i=1,npdfs
         if(pdfin(1:len_trim(pdfin)) .eq. pdflabs(i))then
            mpdf=numspdf(i)
         endif
      enddo
      if(mpdf.eq.-1) then
         write(*,*)'ERROR: pdf ',pdfin,' not implemented in get_pdfup.'
         write(*,*)'known pdfs are'
         write(*,*) pdflabs
         stop 1
      endif
      do i=1,2
         pdfgup(i)=-1
         pdfsup(i)=mpdf
      enddo
      return
      end

      program reweight_xsec_events
c Given a LH file that contains an <rwgt> part, computes the scale 
c and/or PDF dependence through reweighting. A new file is created,
c which does not contain the <rwgt> part, but retains only the 
c information on the maximum and minimum weights due to scale
c and PDF variations
c Compile with makefile_rwgt
      implicit none
      include "run.inc"
      include "reweight_all.inc"
      integer i,ii,jj,kk,isave,idpdf(0:maxPDFs),itmp,lef,ifile,maxevt
     $     ,iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
     $     ,kwgtinfo,kexternal,jwgtnumpartn,ofile,kf,kr,n,nn
      double precision yfactR(maxscales),yfactF(maxscales),value(20)
     $     ,scale1_lhe,scale2_lhe,wgtcentral,wgtmumin,wgtmumax,wgtpdfmin
     $     ,wgtpdfmax,saved_weight,xsecPDFr_acc(0:maxPDFs,maxPDFsets)
     $     ,xsecScale_acc(maxscales,maxscales,maxdynscales)
      logical AddInfoLHE,unweighted
      character*9 ch1
      character*10 MonteCarlo
      character*20 parm(20)
      character*80 event_file,fname1
      character*140 buff
c Parameters
      integer    izero
      parameter (izero=0)
c Common blocks
      character*7         pdlabel,epa_label
      integer       lhaid
      common/to_pdf/lhaid,pdlabel,epa_label
c Les Houches Event File info:
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP)
     $     ,ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP(5,MAXNUP)
     $     ,VTIMUP(MAXNUP),SPINUP(MAXNUP)
c
      call setrun                !Sets up run parameters

      write(*,*) 'Enter event file name'
      read(*,*) event_file

      write(*,*)'Enter 1 to save all cross sections on tape'
      write(*,*)'      0 otherwise'
      read(*,*)isave
      if(isave.eq.1)then
        if (store_rwgt_info)then
           isave = -9
        else
           isave=9
        endif
      else
        isave=0
      endif

      if(do_rwgt_scale)then
         do nn=1,dyn_scale(0)
            if (lscalevar(nn)) then
               write (*,*) "Including scale uncertainties for"/
     $              /" dynamical_scale_choice",dyn_scale(nn)
               write (*,*) "  - renormalisation scale variation:"
     $              ,(scalevarR(i),i=1,nint(scalevarR(0)))
               write (*,*) "  - factorisation scale variation:"
     $              ,(scalevarF(i),i=1,nint(scalevarF(0)))
            else
               write (*,*) "Including central value for"/
     $              /" dynamical_scale_choice",dyn_scale(nn)
            endif
         enddo
      endif

      if(do_rwgt_pdf)then
         do nn=1,lhaPDFid(0)
            if (lpdfvar(nn)) then
               write (*,*) "Including central PDF with "/
     $              /"uncertainties for "//lhaPDFsetname(nn)
            else
               write (*,*) "Including central PDF for "
     $              //lhaPDFsetname(nn)
            endif
c Load all the PDF sets (the 1st one has already by loaded by the call
c to "setrun")
            if (nn.gt.1) then
               call initpdfsetbynamem(nn,lhaPDFsetname(nn))
               if (lpdfvar(nn)) then
                  call numberPDFm(nn,nmemPDF(nn))
                  if (nmemPDF(nn).eq.1) then
                     nmemPDF(nn)=0
                     lpdfvar(nn)=.false.
                  endif
               else
                  nmemPDF(nn)=0
               endif
            endif
            if(nmemPDF(nn)+1.gt.maxPDFs)then
               write(*,*)'Too many PDFs: increase maxPDFs in '/
     $              /'reweight0.inc to ',numPDFs+1
               stop
            endif
         enddo
c start with central member of the first set
         call InitPDFm(1,0)
      endif

      lef=index(event_file,' ')-1
      fname1=event_file(1:lef)//'.rwgt'

      ifile=34
      open (unit=ifile,file=event_file,status='old')
      AddInfoLHE=.true.
      unweighted=.true.
      call read_lhef_header(ifile,maxevt,MonteCarlo)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)

      do i=1,min(10,maxevt)
         call read_lhef_event(ifile,
     &        NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &        IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         if(buff(1:1).ne.'#')then
            write (*,*) 'This event file cannot be reweighted [1]',i
            stop
         endif
         read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe
     $        ,ipartner_lhe,scale1_lhe,scale2_lhe,kwgtinfo,kexternal
     $        ,jwgtnumpartn,wgtcentral,wgtmumin,wgtmumax,wgtpdfmin
     $        ,wgtpdfmax
        if(kwgtinfo.ne.-5)then
           write (*,*) 'This event file cannot be reweighted [2]',i
           write (*,*) kwgtinfo
           stop 1
        endif
        if(i.eq.1)then
          saved_weight=abs(XWGTUP)
        else
          unweighted=unweighted.and.
     #               abs(1.d0-abs(XWGTUP)/saved_weight).lt.1.d-5
        endif
      enddo

      write(*,*)'  '
      if(unweighted)then
        write(*,*)'The events appear to be unweighted'
        write(*,*)' Will store the ratios of recomputed weights'
        write(*,*)' over reference weights'
      else
        write(*,*)'The events appear to be weighted'
        write(*,*)' Will store recomputed weights'
      endif

      rewind(34)

      ofile=35
      open(unit=ofile,file=fname1,status='unknown')

      call read_lhef_header(ifile,maxevt,MonteCarlo)
      call write_lhef_header(ofile,maxevt,MonteCarlo)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
      call write_lhef_init(ofile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)

c To keep track of the accumulated results:
      do kk=1,dyn_scale(0)
         if (lscalevar(kk)) then
            do ii=1,nint(scalevarF(0))
               do jj=1,nint(scalevarR(0))
                  xsecScale_acc(jj,ii,kk)=0d0
               enddo
            enddo
         else
            xsecScale_acc(1,1,kk)=0d0
         endif
      enddo
      do nn=1,lhaPDFid(0)
         if (lpdfvar(nn)) then
            do n=0,nmemPDF(nn)
               xsecPDFr_acc(n,nn)=0d0
            enddo
         else
            xsecPDFr_acc(0,nn)=0d0
         endif
      enddo
      nScontributions=1

c Determine the flavor map between the NLO and Born
      call find_iproc_map()
      do i=1,maxevt
         call read_lhef_event(ifile,
     &       NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &       IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         if(buff(1:1).ne.'#')then
            write(*,*)'This event file cannot be reweighted [3]',i
            stop
         endif
         read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe
     $        ,ipartner_lhe,scale1_lhe,scale2_lhe,kwgtinfo,kexternal
     $        ,jwgtnumpartn,wgtcentral,wgtmumin,wgtmumax,wgtpdfmin
     $        ,wgtpdfmax

c Do the actual reweighting.
         call fill_wgt_info_from_rwgt_lines
         if (do_rwgt_scale)call reweight_scale_ext
         if (do_rwgt_pdf)  call reweight_pdf_ext
         call fill_rwgt_arrays

         write(buff,201)'#aMCatNLO',iSorH_lhe,ifks_lhe,jfks_lhe,
     $        fksfather_lhe,ipartner_lhe, scale1_lhe,scale2_lhe, isave
     $        ,mexternal,izero, wgtcentral,wgtmumin,wgtmumax,wgtpdfmin
     $        ,wgtpdfmax

c renormalize all the scale & PDF weights to have the same normalization
c as XWGTUP
         if(do_rwgt_scale)then
            do kk=1,dyn_scale(0)
               if (lscalevar(kk)) then
                  do ii=1,nint(scalevarF(0))
                     do jj=1,nint(scalevarR(0))
                        wgtxsecmu(jj,ii,kk)=
     &                       wgtxsecmu(jj,ii,kk)/wgtref*XWGTUP
                     enddo
                  enddo
               else
                  wgtxsecmu(1,1,kk)=wgtxsecmu(1,1,kk)/wgtref*XWGTUP
               endif
            enddo
         endif
         if (do_rwgt_pdf) then
            do nn=1,lhaPDFid(0)
               if (lpdfvar(nn)) then
                  do n=0,nmemPDF(nn)
                     wgtxsecPDF(n,nn)=wgtxsecPDF(n,nn)/wgtref*XWGTUP
                  enddo
               else
                  wgtxsecPDF(0,nn)=wgtxsecPDF(0,nn)/wgtref*XWGTUP
               endif
            enddo
         endif

c Keep track of the accumulated results:
         if (do_rwgt_scale) then
            do kk=1,dyn_scale(0)
               if (lscalevar(kk)) then
                  do ii=1,nint(scalevarF(0))
                     do jj=1,nint(scalevarR(0))
                        xsecScale_acc(jj,ii,kk)=xsecScale_acc(jj,ii,kk)
     $                       +wgtxsecmu(jj,ii,kk)
                        
                     enddo
                  enddo
               else
                  xsecScale_acc(1,1,kk)=xsecScale_acc(1,1,kk)
     $                 +wgtxsecmu(1,1,kk)
               endif
            enddo
         endif
         if (do_rwgt_pdf) then
            do nn=1,lhaPDFid(0)
               if (lpdfvar(nn)) then
                  do n=0,nmemPDF(nn)
                     xsecPDFr_acc(n,nn)=xsecPDFr_acc(n,nn)+wgtxsecPDF(n
     $                    ,nn)
                  enddo
               else
                  xsecPDFr_acc(0,nn)=xsecPDFr_acc(0,nn)+wgtxsecPDF(0,nn)
               endif
            enddo
         endif
c Write event to disk:
         call write_lhef_event(ofile,
     &        NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &        IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         
      enddo

      write(ofile,'(a)')'</LesHouchesEvents>'
      close(34)
      close(35)

c Write the accumulated results to a file
      open (unit=34,file='scale_pdf_dependence.dat',status='unknown')
      if (do_rwgt_scale) then
         write (34,*) "scale variations:"
         do kk=1,dyn_scale(0)
            if (lscalevar(kk)) then
               write (34,*) dyn_scale(kk),nint(scalevarR(0))
     $              ,nint(scalevarF(0))
               write (34,*) ((xsecScale_acc(jj,ii,kk),jj=1
     $              ,nint(scalevarR(0))),ii=1,nint(scalevarF(0)))
            else
               write (34,*) dyn_scale(kk),1,1
               write (34,*) xsecScale_acc(1,1,kk)
            endif
         enddo
      endif
      if (do_rwgt_pdf) then
         write (34,*) "pdf variations:"
         do nn=1,lhaPDFid(0)
            if (lpdfvar(nn)) then
               write (34,*) trim(adjustl(lhaPDFsetname(nn))),
     $              nmemPDF(nn)+1
               write (34,*) (xsecPDFr_acc(n,nn),n=0,nmemPDF(nn))
            else
               write(34,*) lhaPDFsetname(nn),nmemPDF(nn) + 1
               write (34,*) xsecPDFr_acc(0,nn)
            endif
         enddo
      endif
      close(34)

 201  format(a9,1x,i1,4(1x,i2),2(1x,e14.8),1x,i2,2(1x,i2),5(1x,e14.8))

      end

c Dummy subroutine (normally used with vegas/mint when resuming plots)
      subroutine resume()
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
      return
      end


      subroutine fill_wgt_info_from_rwgt_lines
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight0.inc'
      integer i,idum,j,k,momenta_conf(2),ii
      icontr=n_ctr_found
      iwgt=1
      do i=1,icontr
         read(n_ctr_str(i),*)(wgt(j,i),j=1,3),(wgt_ME_tree(j,i),j=1,2)
     &        ,idum,(pdg(j,i),j=1,nexternal),QCDpower(i),(bjx(j,i),j=1
     &        ,2),(scales2(j,i),j=1,3),g_strong(i),(momenta_conf(j),j=1
     &        ,2),itype(i),nFKS(i),idum,idum,idum,wgts(1,i)
         do ii=1,2
            do j=1,nexternal
               do k=0,3
                  if (momenta_conf(ii).gt.0) then
                     momenta_m(k,j,ii,i)=momenta_str(k,j
     $                                               ,momenta_conf(ii))
                  else
                     momenta_m(k,j,ii,i)=-99d0
                     exit
                  endif
               enddo
            enddo
         enddo
      enddo
      end
      
      subroutine reweight_scale_ext
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'run.inc'
      include 'reweight0.inc'
      integer i,pd,lp,iwgt_save,kr,kf,dd
      double precision mu2_f(maxscales),mu2_r(maxscales),xlum(maxscales)
     $     ,pdg2pdf,mu2_q,rwgt_muR_dep_fac,g(maxscales),alphas,pi
     $     ,c_mu2_r,c_mu2_f
      parameter (pi=3.14159265358979323846d0)
      external pdg2pdf,rwgt_muR_dep_fac,alphas
      iwgt_save=iwgt
      do i=1,icontr
         iwgt=iwgt_save
         mu2_q=scales2(1,i)
         do dd=1,dyn_scale(0)
            call set_mu_central(i,dd,c_mu2_r,c_mu2_f)
            do kr=1,nint(scalevarR(0))
               if ((.not. lscalevar(dd)) .and. kr.ne.1) exit
               mu2_r(kr)=c_mu2_r*scalevarR(kr)**2
c Update the strong coupling
               g(kr)=sqrt(4d0*pi*alphas(sqrt(mu2_r(kr))))
            enddo
            do kf=1,nint(scalevarF(0))
               if ((.not. lscalevar(dd)) .and. kf.ne.1) exit
               mu2_f(kf)=c_mu2_f*scalevarF(kf)**2
c call the PDFs
               xlum(kf)=1d0
               LP=SIGN(1,LPP(1))
               pd=pdg(1,i)
               if (pd.eq.21) pd=0
               xlum(kf)=xlum(kf)*PDG2PDF(ABS(LPP(1)),pd*LP,bjx(1,i)
     &              ,DSQRT(mu2_f(kf)))
               LP=SIGN(1,LPP(2))
               pd=pdg(2,i)
               if (pd.eq.21) pd=0
               xlum(kf)=xlum(kf)*PDG2PDF(ABS(LPP(2)),pd*LP,bjx(2,i)
     &              ,DSQRT(mu2_f(kf)))
            enddo
            do kf=1,nint(scalevarF(0))
               if ((.not. lscalevar(dd)) .and. kf.ne.1) exit
               do kr=1,nint(scalevarR(0))
                  if ((.not. lscalevar(dd)) .and. kr.ne.1) exit
                  iwgt=iwgt+1   ! increment the iwgt for the wgts() array
                  if (iwgt.gt.max_wgt) then
                     write (*,*) 'ERROR too many weights in '/
     $                    /'reweight_scale',iwgt,max_wgt
                     stop 1
                  endif
c add the weights to the array
                  wgts(iwgt,i)=xlum(kf) * (wgt(1,i)+wgt(2,i)
     $                 *log(mu2_r(kr)/mu2_q)+wgt(3,i)*log(mu2_f(kf)
     $                 /mu2_q))*g(kr)**QCDpower(i)
                  wgts(iwgt,i)=wgts(iwgt,i)*rwgt_muR_dep_fac(
     &                          sqrt(mu2_r(kr)),sqrt(scales2(2,i)))
               enddo
            enddo
         enddo
      enddo
      return
      end

      
      subroutine reweight_pdf_ext
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'run.inc'
      include 'reweight0.inc'
      integer i,pd,lp,iwgt_save,izero,n,nn,iset,imem
      parameter (izero=0)
      double precision mu2_f,mu2_r,pdg2pdf,mu2_q,rwgt_muR_dep_fac
     &     ,xlum,alphas,g,pi
      parameter (pi=3.14159265358979323846d0)
      external pdg2pdf,rwgt_muR_dep_fac,alphas
      do nn=1,lhaPDFid(0)
         do n=0,nmemPDF(nn)
            if ((.not. lpdfvar(nn)) .and. n.ne.0) exit
            iwgt=iwgt+1
            if (iwgt.gt.max_wgt) then
               write (*,*) 'ERROR too many weights in reweight_pdf',iwgt
     &              ,max_wgt
               stop 1
            endif
            call InitPDFm(nn,n)
            do i=1,icontr
               mu2_q=scales2(1,i)
               mu2_r=scales2(2,i)
               mu2_f=scales2(3,i)
c alpha_s
               g=sqrt(4d0*pi*alphas(sqrt(mu2_r)))
c call the PDFs
               xlum=1d0
               LP=SIGN(1,LPP(1))
               pd=pdg(1,i)
               if (pd.eq.21) pd=0
               xlum=xlum*
     &            PDG2PDF(ABS(LPP(1)),pd*LP,bjx(1,i),DSQRT(mu2_f))
               LP=SIGN(1,LPP(2))
               pd=pdg(2,i)
               if (pd.eq.21) pd=0
               xlum=xlum*
     &             PDG2PDF(ABS(LPP(2)),pd*LP,bjx(2,i),DSQRT(mu2_f))
c add the weights to the array
               wgts(iwgt,i)=xlum * (wgt(1,i) + wgt(2,i)*log(mu2_r/mu2_q)
     &              +wgt(3,i)*log(mu2_f/mu2_q))*g**QCDpower(i)
               wgts(iwgt,i)=wgts(iwgt,i)
     &              *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r))
            enddo
         enddo
      enddo
c reset to the 0th member of the 1st set
      call InitPDFm(1,0)
      return
      end
      

      subroutine fill_rwgt_arrays
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight0.inc'
      include 'run.inc'
      integer ii,jj,kk,nn,n,iw,i
      do kk=1,dyn_scale(0)
         if (lscalevar(kk)) then
            do ii=1,nint(scalevarF(0))
               do jj=1,nint(scalevarR(0))
                  wgtxsecmu(jj,ii,kk)=0d0
               enddo
            enddo
         else
            wgtxsecmu(1,1,kk)=0d0
         endif
      enddo
      do nn=1,lhaPDFid(0)
         if (lpdfvar(nn)) then
            do n=0,nmemPDF(nn)
               wgtxsecPDF(n,nn)=0d0
            enddo
         else
            wgtxsecPDF(0,nn)=0d0
         endif
      enddo
      do i=1,icontr
         iw=2
         if (do_rwgt_scale) then
            do kk=1,dyn_scale(0)
               if (lscalevar(kk)) then
                  do ii=1,nint(scalevarF(0))
                     do jj=1,nint(scalevarR(0))
                        wgtxsecmu(jj,ii,kk)=wgtxsecmu(jj,ii,kk)+wgts(iw,i)
                        iw=iw+1
                     enddo
                  enddo
               else
                  wgtxsecmu(1,1,kk)=wgtxsecmu(1,1,kk)+wgts(iw,i)
                  iw=iw+1
               endif
            enddo
         endif
         if (do_rwgt_pdf) then
            do nn=1,lhaPDFid(0)
               if (lpdfvar(nn)) then
                  do n=0,nmemPDF(nn)
                     wgtxsecPDF(n,nn)=wgtxsecPDF(n,nn)+wgts(iw,i)
                     iw=iw+1
                  enddo
               else
                  wgtxsecPDF(0,nn)=wgtxsecPDF(0,nn)+wgts(iw,i)
                  iw=iw+1
               endif
            enddo
         endif
      enddo
      return
      end

      
      subroutine set_mu_central(ic,dd,c_mu2_r,c_mu2_f)
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight0.inc'
      include 'run.inc'
      integer ic,dd,i,j
      double precision c_mu2_r,c_mu2_f,muR,muF,pp(0:3,nexternal)
      if (dd.eq.1) then
         c_mu2_r=scales2(2,ic)
         c_mu2_f=scales2(3,ic)
      else
c need to recompute the scales using the momenta
         dynamical_scale_choice=dyn_scale(dd)
         do i=1,nexternal
            do j=0,3
               pp(j,i)=momenta(j,i,ic)
            enddo
         enddo
         call set_ren_scale(pp,muR)
         c_mu2_r=muR**2
         call set_fac_scale(pp,muF)
         c_mu2_f=muF**2
c     reset the default dynamical_scale_choice
         dynamical_scale_choice=dyn_scale(1)
      endif
      return
      end

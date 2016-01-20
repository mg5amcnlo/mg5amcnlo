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
      integer i,ii,jj,isave,idpdf(0:maxPDFs),itmp,lef,ifile,maxevt
     $     ,iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
     $     ,kwgtinfo,kexternal,jwgtnumpartn,ofile,kf,kr,n
      double precision yfactR(maxscales),yfactF(maxscales),value(20)
     $     ,scale1_lhe,scale2_lhe,wgtcentral,wgtmumin,wgtmumax,wgtpdfmin
     $     ,wgtpdfmax,saved_weight,xsecPDFr_acc(0:maxPDFs)
     $     ,xsecScale_acc(maxscales,maxscales)
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
        isave=9
      else
        isave=0
      endif

      if(do_rwgt_scale)then
        yfactR(1)=1.d0
        yfactR(2)=rw_Rscale_up
        yfactR(3)=rw_Rscale_down
        yfactF(1)=1.d0
        yfactF(2)=rw_Fscale_up
        yfactF(3)=rw_Fscale_down
        write(*,*) 'Doing scale reweight:'
        write(*,*) rw_Fscale_down, ' < mu_F < ', rw_Fscale_up
        write(*,*) rw_Rscale_down, ' < mu_R < ', rw_Rscale_up
        numscales=3
        if(numscales.gt.maxscales)then
           write(*,*)'Too many scales: '/
     $          /'increase maxscales in reweight0.inc'
           stop
        endif
      else
        numscales=0
      endif

c Note: when ipdf#0, the central PDF set will be used also as a reference
c for the scale uncertainty
      if(do_rwgt_pdf)then
         idpdf(0)=lhaid
         idpdf(1)=pdf_set_min
         itmp=pdf_set_max
         numPDFs=itmp-idpdf(1)+1
         if(numPDFs.gt.maxPDFs)then
            write(*,*)'Too many PDFs: increase maxPDFs in reweight0.inc'
            stop
         endif
         write(*,*) 'Doing PDF reweight:'
         write(*,*) 'Central set id: ', idpdf(0)
         write(*,*) 'Min error set id: ', idpdf(1)
         write(*,*) 'Max error set id: ', itmp
         do i=2,numPDFs
            idpdf(i)=idpdf(1)+i-1
         enddo
         value(1)=idpdf(0)
         parm(1)='DEFAULT'
         call pdfset(parm,value)
      else
         numPDFs=0
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
      do ii=1,numscales
         do jj=1,numscales
            xsecScale_acc(jj,ii)=0d0
         enddo
      enddo
      do n=0,numPDFs
         xsecPDFr_acc(n)=0d0
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
         if (do_rwgt_scale)call reweight_scale_ext(yfactR,yfactF)
         if (do_rwgt_pdf)  call reweight_pdf_ext
         call fill_rwgt_arrays

         write(buff,201)'#aMCatNLO',iSorH_lhe,ifks_lhe,jfks_lhe,
     $        fksfather_lhe,ipartner_lhe, scale1_lhe,scale2_lhe, isave
     $        ,izero,izero, wgtcentral,wgtmumin,wgtmumax,wgtpdfmin
     $        ,wgtpdfmax

c renormalize all the scale & PDF weights to have the same normalization
c as XWGTUP
         if(do_rwgt_scale)then
            do kr=1,numscales
               do kf=1,numscales
                  wgtxsecmu(kr,kf)=wgtxsecmu(kr,kf)/wgtref*XWGTUP
               enddo
            enddo
         endif
         if(do_rwgt_pdf)then
            do n=0,numPDFs
               wgtxsecPDF(n)=wgtxsecPDF(n)/wgtref*XWGTUP
            enddo
         endif

c Keep track of the accumulated results:
         if (numscales.gt.0) then
            do ii=1,numscales
               do jj=1,numscales
                  xsecScale_acc(ii,jj)=xsecScale_acc(ii,jj)+wgtxsecmu(ii
     $                 ,jj)
               enddo
            enddo
         endif
         if (numPDFs.gt.0) then
            do n=0,numPDFs
               xsecPDFr_acc(n)=xsecPDFr_acc(n)+wgtxsecPDF(n)
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
      write (34,*) numscales**2
      if (numscales.gt.0) then
         write (34,*) ((xsecScale_acc(ii,jj),ii=1,numscales),jj=1
     $        ,numscales)
      else
         write (34,*) ''
      endif
      if (numPDFs.gt.0) then
         write (34,*) numPDFs + 1
         write (34,*) (xsecPDFr_acc(n),n=0,numPDFs)
      else
         write(34,*) numPDFs
         write (34,*) ''
      endif
      close(34)

 201  format(a9,1x,i1,4(1x,i2),2(1x,e14.8),1x,i1,2(1x,i2),5(1x,e14.8))

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
      integer i,idum,j,k,momenta_conf
      icontr=n_ctr_found
      iwgt=1
      do i=1,icontr
         read(n_ctr_str(i),*)(wgt(j,i),j=1,3),idum,(pdg(j,i),j=1
     &        ,nexternal),QCDpower(i),(bjx(j,i),j=1,2),(scales2(j,i),j=1
     &        ,3),momenta_conf,itype(i),nFKS(i),wgts(1,i)
         do j=1,nexternal
            do k=0,3
               momenta(k,j,i)=momenta_str(k,j,momenta_conf)
            enddo
         enddo
      enddo
      end
      
      subroutine reweight_scale_ext(yfactR,yfactF)
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'run.inc'
      include 'reweight0.inc'
      integer i,pd,lp,iwgt_save,kr,kf
      double precision yfactR(maxscales),yfactF(maxscales)
     $     ,mu2_f(maxscales),mu2_r(maxscales),xlum(maxscales),pdg2pdf
     $     ,mu2_q,rwgt_muR_dep_fac,g(maxscales),alphas,pi
      parameter (pi=3.14159265358979323846d0)
      external pdg2pdf,rwgt_muR_dep_fac,alphas
      iwgt_save=iwgt
      do i=1,icontr
         iwgt=iwgt_save
         mu2_q=scales2(1,i)
         do kr=1,numscales
            mu2_r(kr)=scales2(2,i)*yfactR(kr)**2
c Update the strong coupling
            g(kr)=sqrt(4d0*pi*alphas(sqrt(mu2_r(kr))))
         enddo
         do kf=1,numscales
            mu2_f(kf)=scales2(3,i)*yfactF(kf)**2
c call the PDFs
            xlum(kf)=1d0
            LP=SIGN(1,LPP(1))
            pd=pdg(1,i)
            if (pd.eq.21) pd=0
            xlum(kf)=xlum(kf)*PDG2PDF(ABS(LPP(1)),pd*LP,bjx(1,i)
     &           ,DSQRT(mu2_f(kf)))
            LP=SIGN(1,LPP(2))
            pd=pdg(2,i)
            if (pd.eq.21) pd=0
            xlum(kf)=xlum(kf)*PDG2PDF(ABS(LPP(2)),pd*LP,bjx(2,i)
     &           ,DSQRT(mu2_f(kf)))
         enddo
         do kr=1,numscales
            do kf=1,numscales
               iwgt=iwgt+1 ! increment the iwgt for the wgts() array
               if (iwgt.gt.max_wgt) then
                  write (*,*) 'ERROR too many weights in reweight_scale'
     &                 ,iwgt,max_wgt
                  stop 1
               endif
c add the weights to the array
               wgts(iwgt,i)=xlum(kf) * (wgt(1,i)+wgt(2,i)*log(mu2_r(kr)
     &              /mu2_q)+wgt(3,i)*log(mu2_f(kf)/mu2_q))*g(kr)
     &              **QCDpower(i)
               wgts(iwgt,i)=wgts(iwgt,i)
     &              *rwgt_muR_dep_fac(sqrt(mu2_r(kr)),sqrt(mu2_r(1)))
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
      integer i,pd,lp,iwgt_save,izero,n
      parameter (izero=0)
      double precision mu2_f,mu2_r,pdg2pdf,mu2_q,rwgt_muR_dep_fac
     &     ,xlum,alphas,g,pi
      parameter (pi=3.14159265358979323846d0)
      external pdg2pdf,rwgt_muR_dep_fac,alphas
      do n=0,numPDFs
         iwgt=iwgt+1
         if (iwgt.gt.max_wgt) then
            write (*,*) 'ERROR too many weights in reweight_pdf',iwgt
     &           ,max_wgt
            stop 1
         endif
         call InitPDF(n)
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
            xlum=xlum*PDG2PDF(ABS(LPP(1)),pd*LP,bjx(1,i),DSQRT(mu2_f))
            LP=SIGN(1,LPP(2))
            pd=pdg(2,i)
            if (pd.eq.21) pd=0
            xlum=xlum*PDG2PDF(ABS(LPP(2)),pd*LP,bjx(2,i),DSQRT(mu2_f))
c add the weights to the array
            wgts(iwgt,i)=xlum * (wgt(1,i) + wgt(2,i)*log(mu2_r/mu2_q) +
     &           wgt(3,i)*log(mu2_f/mu2_q))*g**QCDpower(i)
            wgts(iwgt,i)=wgts(iwgt,i)
     &           *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r))
         enddo
      enddo
      call InitPDF(izero)
      return
      end
      

      subroutine fill_rwgt_arrays
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight0.inc'
      integer kr,kf,n,iw,i
      do kr=1,numscales
         do kf=1,numscales
            wgtxsecmu(kr,kf)=0d0
         enddo
      enddo
      do n=0,numPDFs
         wgtxsecPDF(n)=0d0
      enddo
      do i=1,icontr
         iw=2
         do kr=1,numscales
            do kf=1,numscales
               wgtxsecmu(kr,kf)=wgtxsecmu(kr,kf)+wgts(iw,i)
               iw=iw+1
            enddo
         enddo
         do n=0,numPDFs
            wgtxsecPDF(n)=wgtxsecPDF(n)+wgts(iw,i)
            iw=iw+1
         enddo
      enddo
      if (numscales.eq.0) then
         wgtxsecmu(1,1)=wgtxsecPDF(0)
      endif
      return
      end

      

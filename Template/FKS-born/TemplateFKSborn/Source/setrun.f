      subroutine setrun
c----------------------------------------------------------------------
c     Sets the run parameters reading them from the run_card.dat
c
c 1. PDF set
c 2. Collider parameters
c 3. cuts
c---------------------------------------------------------------------- 
      implicit none
c
c     parameters
c
      integer maxpara
      parameter (maxpara=1000)
c
c     local
c     
      integer npara
      character*20 param(maxpara),value(maxpara)
c
c     include
c
      include 'genps.inc'
      include 'PDF/pdf.inc'
      include 'run.inc'
      include 'alfas.inc'
      include 'MODEL/coupl.inc'

      double precision D
      common/to_dj/D
c
c     local
c
      character*20 ctemp
      integer k,i,l1,l2
      character*132 buff
      integer lp1,lp2
      real*8 eb1,eb2
      real*8 pb1,pb2
C
C     input cuts
C
      include 'cuts.inc'
C
C     BEAM POLARIZATION
C
      REAL*8 POL(2)
      common/to_polarization/ POL
      data POL/1d0,1d0/
c
c     Les Houches init block (for the <init> info)
c
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
c
      include 'nexternal.inc'
      integer    maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'leshouche.inc'
c
c
c
      logical gridrun,gridpack
      integer          iseed
      common /to_seed/ iseed
c
c----------
c     start
c----------
c
c     read the run_card.dat
c
      call load_para(npara,param,value)

c*********************************************************************
c max jet flavor                                                     *
c*********************************************************************

      call  get_integer (npara,param,value,"maxjetflavor",maxjetflavor,4)

c*********************************************************************
c     Minimum pt's                                                   *
c*********************************************************************

      call get_real   (npara,param,value," ptj ",ptj    ,0d0)
c$$$      call get_real   (npara,param,value," ptb ",ptb    ,20d0)
c$$$      call get_real   (npara,param,value," pta ",pta    ,20d0)
c$$$      call get_real   (npara,param,value," ptl ",ptl    ,20d0)
c$$$      call get_real   (npara,param,value," misset ",misset ,0d0)
c$$$      call get_real   (npara,param,value," ptonium ",ptonium,0d0)
      ptb     = 0d0
      pta     = 0d0
      ptl     = 0d0
      misset  = 0d0
      ptonium = 0d0
      

c*********************************************************************
c     Maximum pt's                                                   *
c*********************************************************************

c$$$      call get_real   (npara,param,value," ptjmax "   ,ptjmax,1d5)
c$$$      call get_real   (npara,param,value," ptbmax "   ,ptbmax,1d5)
c$$$      call get_real   (npara,param,value," ptamax "   ,ptamax,1d5)
c$$$      call get_real   (npara,param,value," ptlmax "   ,ptlmax,1d5)
c$$$      call get_real   (npara,param,value," missetmax ",missetmax,1d5)
      ptjmax=1d5
      ptbmax=1d5
      ptamax=1d5
      ptlmax=1d5
      missetmax=1d5

c*********************************************************************
c     Maximum rapidity (absolute value)                              *
c*********************************************************************

c$$$      call get_real   (npara,param,value," etaj ",    etaj,4d0)
c$$$      call get_real   (npara,param,value," etab ",    etab,4d0)
c$$$      call get_real   (npara,param,value," etaa ",    etaa,4d0)
c$$$      call get_real   (npara,param,value," etal ",    etal,4d0)
c$$$      call get_real   (npara,param,value," etaonium ",etaonium,1d2)
      etaj=1d2
      etab=1d2
      etaa=1d2
      etal=1d2
      etaonium=1d2

c*********************************************************************
c     Minimum rapidity (absolute value)                              *
c*********************************************************************

c$$$      call get_real   (npara,param,value," etajmin ",etajmin,0d0)
c$$$      call get_real   (npara,param,value," etabmin ",etabmin,0d0)
c$$$      call get_real   (npara,param,value," etaamin ",etaamin,0d0)
c$$$      call get_real   (npara,param,value," etalmin ",etalmin,0d0)
      etajmin=0d0
      etabmin=0d0
      etaamin=0d0
      etalmin=0d0

c*********************************************************************
c     Minimum E's                                                   *
c*********************************************************************

c$$$      call get_real   (npara,param,value," ej ", ej, 0d0)
c$$$      call get_real   (npara,param,value," eb ", eb, 0d0)
c$$$      call get_real   (npara,param,value," ea ", ea, 0d0)
c$$$      call get_real   (npara,param,value," el ", el, 0d0)
      ej=0d0
      eb=0d0
      ea=0d0
      el=0d0

c*********************************************************************
c     Maximum E's                                                    *
c*********************************************************************

c$$$      call get_real   (npara,param,value," ejmax ", ejmax, 1d5)
c$$$      call get_real   (npara,param,value," ebmax ", ebmax, 1d5)
c$$$      call get_real   (npara,param,value," eamax ", eamax, 1d5)
c$$$      call get_real   (npara,param,value," elmax ", elmax, 1d5)
      ejmax=1d5
      ebmax=1d5
      eamax=1d5
      elmax=1d5

c*********************************************************************
c     Minimum DeltaR distance                                        *
c*********************************************************************

c$$$      call get_real   (npara,param,value," drjj ",drjj,0.4d0)
c$$$      call get_real   (npara,param,value," drbb ",drbb,0.4d0)
c$$$      call get_real   (npara,param,value," drll ",drll,0.4d0)
c$$$      call get_real   (npara,param,value," draa ",draa,0.4d0)
c$$$      call get_real   (npara,param,value," drbj ",drbj,0.4d0)
c$$$      call get_real   (npara,param,value," draj ",draj,0.4d0)
c$$$      call get_real   (npara,param,value," drjl ",drjl,0.4d0)
c$$$      call get_real   (npara,param,value," drab ",drab,0.4d0)
c$$$      call get_real   (npara,param,value," drbl ",drbl,0.4d0)
c$$$      call get_real   (npara,param,value," dral ",dral,0.4d0)
      drjj=0d0
      drbb=0d0
      drll=0d0
      draa=0d0
      drbj=0d0
      draj=0d0
      drjl=0d0
      drab=0d0
      drbl=0d0
      dral=0d0

c*********************************************************************
c     Maximum DeltaR distance                                        *
c*********************************************************************

c$$$      call get_real   (npara,param,value," drjjmax ",drjjmax,1d2)
c$$$      call get_real   (npara,param,value," drbbmax ",drbbmax,1d2)
c$$$      call get_real   (npara,param,value," drllmax ",drllmax,1d2)
c$$$      call get_real   (npara,param,value," draamax ",draamax,1d2)
c$$$      call get_real   (npara,param,value," drbjmax ",drbjmax,1d2)
c$$$      call get_real   (npara,param,value," drajmax ",drajmax,1d2)
c$$$      call get_real   (npara,param,value," drjlmax ",drjlmax,1d2)
c$$$      call get_real   (npara,param,value," drabmax ",drabmax,1d2)
c$$$      call get_real   (npara,param,value," drblmax ",drblmax,1d2)
c$$$      call get_real   (npara,param,value," dralmax ",dralmax,1d2)
      drjjmax=1d2
      drbbmax=1d2
      drllmax=1d2
      draamax=1d2
      drbjmax=1d2
      drajmax=1d2
      drjlmax=1d2
      drabmax=1d2
      drblmax=1d2
      dralmax=1d2

c*********************************************************************
c     Minimum invariant mass for pairs                               *
c*********************************************************************

c$$$      call get_real   (npara,param,value," mmjj ",mmjj,0d0)
c$$$      call get_real   (npara,param,value," mmbb ",mmbb,0d0)
c$$$      call get_real   (npara,param,value," mmaa ",mmaa,0d0)
c$$$      call get_real   (npara,param,value," mmll ",mmll,0d0)
      mmjj=0d0
      mmbb=0d0
      mmaa=0d0
      mmll=0d0

c*********************************************************************
c     Maximum invariant mass for pairs                               *
c*********************************************************************

c$$$      call get_real   (npara,param,value," mmjjmax ",mmjjmax,1d5)
c$$$      call get_real   (npara,param,value," mmbbmax ",mmbbmax,1d5)
c$$$      call get_real   (npara,param,value," mmaamax ",mmaamax,1d5)
c$$$      call get_real   (npara,param,value," mmllmax ",mmllmax,1d5)
      mmjjmax=1d5
      mmbbmax=1d5
      mmaamax=1d5
      mmllmax=1d5

c*********************************************************************
c     Min Maxi invariant mass for all leptons                        *
c*********************************************************************

c$$$      call get_real   (npara,param,value," mmnl    ",mmnl   ,0d0)
c$$$      call get_real   (npara,param,value," mmnlmax ",mmnlmax,1d5)
      mmnl=0d0
      mmnlmax=1d5

c*********************************************************************
c     Inclusive cuts                                                 *
c*********************************************************************

c$$$      call get_real   (npara,param,value," xptj ",xptj,0d0)
c$$$      call get_real   (npara,param,value," xptb ",xptb,0d0)
c$$$      call get_real   (npara,param,value," xpta ",xpta,0d0)
c$$$      call get_real   (npara,param,value," xptl ",xptl,0d0)
c$$$      call get_real   (npara,param,value," xmtcentral ",xmtc,0d0)
      xptj=0d0
      xptb=0d0
      xpta=0d0
      xptl=0d0
      xmtc=0d0

c*********************************************************************
c     WBF cuts                                                       *
c*********************************************************************

c$$$      call get_real   (npara,param,value," xetamin ",xetamin,0d0)
c$$$      call get_real   (npara,param,value," deltaeta",deltaeta,0d0)
      xetamin=0d0
      deltaeta=0d0

c*********************************************************************
c     Jet measure cuts                                               *
c*********************************************************************

c$$$      call get_real   (npara,param,value," xqcut ",xqcut,0d0)
c$$$      call get_real   (npara,param,value," d ",D,1d0)
      xqcut=0d0
      d=1d0

c*********************************************************************
c Set min pt of one heavy particle                                   *
c*********************************************************************

c$$$ 	call get_real   (npara,param,value,"ptheavy",ptheavy,0d0)
        ptheavy=0d0

c*********************************************************************
c Check   the pt's of the jets sorted by pt                          *
c*********************************************************************

c$$$ 	call get_real   (npara,param,value,"ptj1min",ptj1min,0d0)
c$$$ 	call get_real   (npara,param,value,"ptj1max",ptj1max,1d5)
c$$$ 	call get_real   (npara,param,value,"ptj2min",ptj2min,0d0)
c$$$ 	call get_real   (npara,param,value,"ptj2max",ptj2max,1d5)
c$$$ 	call get_real   (npara,param,value,"ptj3min",ptj3min,0d0)
c$$$ 	call get_real   (npara,param,value,"ptj3max",ptj3max,1d5)
c$$$ 	call get_real   (npara,param,value,"ptj4min",ptj4min,0d0)
c$$$ 	call get_real   (npara,param,value,"ptj4max",ptj4max,1d5)
c$$$	call get_real   (npara,param,value,"cutuse" ,cutuse,0d0)
        ptj1min=0d0
        ptj1max=1d5
        ptj2min=0d0
        ptj2max=1d5
        ptj3min=0d0
        ptj3max=1d5
        ptj4min=0d0
        ptj4max=1d5
        cutuse=0d0

c*********************************************************************
c Check  Ht                                                          *
c*********************************************************************

c$$$	call get_real   (npara,param,value,"ht2min",ht2min,0d0)
c$$$	call get_real   (npara,param,value,"ht3min",ht3min,0d0)
c$$$	call get_real   (npara,param,value,"ht4min",ht4min,0d0)
c$$$	call get_real   (npara,param,value,"ht2max",ht2max,1d5)
c$$$	call get_real   (npara,param,value,"ht3max",ht3max,1d5)
c$$$	call get_real   (npara,param,value,"ht4max",ht4max,1d5)
c$$$	call get_real   (npara,param,value,"htjmin",htjmin,0d0)
c$$$	call get_real   (npara,param,value,"htjmax",htjmax,1d5)
        ht2min=0d0
        ht3min=0d0
        ht4min=0d0
        ht2max=1d5
        ht3max=1d5
        ht4max=1d5
        htjmin=0d0
        htjmax=1d5

c*********************************************************************
c     Random Number Seed                                             *
c*********************************************************************

c$$$      call get_logical   (npara,param,value," gridrun ",gridrun,.false.)
c$$$      call get_logical   (npara,param,value," gridpack ",gridpack,.false.)
c$$$      if (gridrun.and.gridpack)then
c$$$         call get_integer   (npara,param,value," gseed ",iseed,0)
c$$$      else 
c$$$         call get_integer (npara,param,value," iseed ",iseed,0)
c$$$      endif
        gridrun=.false.
        gridpack=.false.
        iseed=0

c************************************************************************     
c     Renormalization and factorization scales                          *
c************************************************************************     
c
c Note: case_trap2 in rw_routines will convert input strings
c to lowercase

      call get_logical(npara,param,value," fixed_ren_scale ",
     #                 fixed_ren_scale,.true.)
      call get_logical(npara,param,value," fixed_fac_scale ",
     #                 fixed_fac_scale,.true.)
      call get_logical(npara,param,value," fixed_qes_scale ",
     #                 fixed_QES_scale,.true.)
c
      call get_real(npara,param,value," mur_ref_fixed ",
     #              muR_ref_fixed,91.188d0)
      call get_real(npara,param,value," muf1_ref_fixed ",
     #              muF1_ref_fixed,91.188d0)
      call get_real(npara,param,value," muf2_ref_fixed ",
     #              muF2_ref_fixed,91.188d0)
      call get_real(npara,param,value," qes_ref_fixed ",
     #              QES_ref_fixed,91.188d0)
c For backward compatibility
      scale = muR_ref_fixed
      q2fact(1) = muF1_ref_fixed**2      ! fact scale**2 for pdf1
      q2fact(2) = muF2_ref_fixed**2      ! fact scale**2 for pdf2     
c
      call get_real(npara,param,value," mur_over_ref ",muR_over_ref,1d0)
      call get_real(npara,param,value," muf1_over_ref ",muF1_over_ref,1d0)
      call get_real(npara,param,value," muf2_over_ref ",muF2_over_ref,1d0)
      call get_real(npara,param,value," qes_over_ref ",QES_over_ref,1d0)
c For backward compatibility
      scalefact=muR_over_ref
      ellissextonfact=QES_over_ref
c
      call get_logical(npara,param,value," fixed_couplings ",
     #                 fixed_couplings,.true.)
c$$$      call get_integer(npara,param,value," ickkw "          ,ickkw    , 0  )
c$$$      call get_logical(npara,param,value," chcluster ",chcluster,.false.)
c$$$c     ktscheme for xqcut: 1: pT/Durham kT; 2: pythia pTE/Durham kT
c$$$      call get_integer (npara,param,value," ktscheme ",ktscheme,1)
      ickkw=0
      chcluster=.false.
      ktscheme=1
      
      if(ickkw.gt.0)then
c$$$         call get_real   (npara,param,value," alpsfact "       ,alpsfact , 1d0)
         alpsfact=1d0
      endif
      if(ickkw.eq.2)then
        call get_integer(npara,param,value," highestmult "    ,nhmult, 0)
        call get_string (npara,param,value," issgridfile ",
     &       issgridfile,'issudgrid.dat')
      endif

c************************************************************************     
c    Collider energy and type                                           *
c************************************************************************     
c     lpp  = -1 (antiproton), 0 (no pdf), 1 (proton)
c     lpp  =  2 (proton emitting a photon without breaking)
c     lpp  =  3 (electron emitting a photon)
c     ebeam= energy of each beam in GeV

      call get_integer(npara,param,value," lpp1 "   ,lp1,1  )
      call get_integer(npara,param,value," lpp2 "   ,lp2,1  )
      call get_real   (npara,param,value," ebeam1 " ,eb1,7d3)
      call get_real   (npara,param,value," ebeam2 " ,eb2,7d3)
     
      lpp(1)=lp1
      lpp(2)=lp2
      ebeam(1)=eb1
      ebeam(2)=eb2

c************************************************************************     
c    Beam polarization
c************************************************************************     
c$$$      call get_real   (npara,param,value," polbeam1 " ,pb1,0d0)
c$$$      call get_real   (npara,param,value," polbeam2 " ,pb2,0d0)
      pb1=0d0
      pb2=0d0

      if(pb1.ne.0d0.and.lp1.eq.0) pol(1)=sign(1+abs(pb1)/100d0,pb1)
      if(pb2.ne.0d0.and.lp2.eq.0) pol(2)=sign(1+abs(pb2)/100d0,pb2)

      if(pb1.ne.0.or.pb2.ne.0) write(*,*) 'Setting beam polarization ',
     $     sign((abs(pol(1))-1)*100,pol(1)),
     $     sign((abs(pol(2))-1)*100,pol(2))

c************************************************************************     
c    BW cutoff (M+/-bwcutoff*Gamma)
c************************************************************************     
      call get_real   (npara,param,value," bwcutoff " ,bwcutoff,15d0)

c************************************************************************     
c    Collider pdf                                                       *
c************************************************************************     

      call get_string (npara,param,value," pdlabel ",pdlabel,'cteq6l1')
c
c     if lhapdf is used the following number identifies the set
c
      call get_integer(npara,param,value," lhaid  ",lhaid,10042)

c !!! Default behavior changed (MH, Aug. 07) !!!
c If no pdf, read the param_card and use the value from there and
c order of alfas running = 2

      if(lp1.ne.0.or.lp2.ne.0) then
          write(*,*) 'A PDF is used, so alpha_s(MZ) is going to be modified'
          call setpara('param_card.dat')
          asmz=G**2/(16d0*atan(1d0))
          write(*,*) 'Old value of alpha_s from param_card: ',asmz
          call pdfwrap
          write(*,*) 'New value of alpha_s from PDF ',pdlabel,':',asmz
      else
          call setpara('param_card.dat',.true.)
          asmz=G**2/(16d0*atan(1d0))
          nloop=2
          pdlabel='none'
          write(*,*) 'No PDF is used, alpha_s(MZ) from param_card is used'
          write(*,*) 'Value of alpha_s from param_card: ',asmz
          write(*,*) 'The default order of alpha_s running is fixed to ',nloop
      endif
c !!! end of modification !!!

C       Fill common block for Les Houches init info
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
          idbmup(i)=idup(i,1)
        else
          idbmup(i)=lpp(i)
        endif
        ebmup(i)=ebeam(i)
      enddo
      call get_pdfup(pdlabel,pdfgup,pdfsup,lhaid)

      return
 99   write(*,*) 'error in reading'
      return
      end

C-------------------------------------------------
C   GET_PDFUP
C   Convert MadEvent pdf name to LHAPDF number
C-------------------------------------------------

      subroutine get_pdfup(pdfin,pdfgup,pdfsup,lhaid)
      implicit none

      character*(*) pdfin
      integer mpdf
      integer npdfs,i,pdfgup(2),pdfsup(2),lhaid

      parameter (npdfs=13)
      character*7 pdflabs(npdfs)
      data pdflabs/
     $   'none',
     $   'mrs02nl',
     $   'mrs02nn',
     $   'cteq4_m',
     $   'cteq4_l',
     $   'cteq4_d',
     $   'cteq5_m',
     $   'cteq5_d',
     $   'cteq5_l',
     $   'cteq5m1',
     $   'cteq6_m',
     $   'cteq6_l',
     $   'cteq6l1'/
      integer numspdf(npdfs)
      data numspdf/
     $   00000,
     $   20250,
     $   20270,
     $   19150,
     $   19170,
     $   19160,
     $   19050,
     $   19060,
     $   19070,
     $   19051,
     $   10000,
     $   10041,
     $   10042/


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
        write(*,*)'pdf ',pdfin,' not implemented in get_pdfup.'
        write(*,*)'known pdfs are'
        write(*,*) pdflabs
        write(*,*)'using ',pdflabs(12)
        mpdf=numspdf(12)
      endif

      do i=1,2
        pdfgup(i)=-1
        pdfsup(i)=mpdf
      enddo

      return
      end

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
c     include
c
c      include 'genps.inc'
      include 'maxparticles.inc'
      include 'run_config.inc'
      include 'PDF/pdf.inc'
      include 'run.inc'
      include 'alfas.inc'
      include 'MODEL/coupl.inc'

      double precision D
      common/to_dj/D
c
c     PARAM_CARD
c
      character*30 param_card_name
      common/to_param_card_name/param_card_name
c
c     local
c     
      integer npara
      character*20 param(maxpara),value(maxpara)
      character*20 ctemp
      integer k,i,l1,l2
      character*132 buff
      real*8 sf1,sf2
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
      include 'maxamps.inc'
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
      data pdfwgt/.false./
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
c Automatically set ptj and mjj = xqcut (if xqcut > 0)
c*********************************************************************

      call  get_logical (npara,param,value,"auto_ptj_mjj",auto_ptj_mjj,.true.)

c*********************************************************************
c cut on decay products (pt/e/eta/dr/mij) or not
c*********************************************************************

      call  get_logical (npara,param,value,"cut_decays",cut_decays,.true.)

c*********************************************************************
c     Minimum pt's                                                   *
c*********************************************************************

      call get_real   (npara,param,value," ptj    ",ptj,20d0)
      call get_real   (npara,param,value," ptb    ",ptb,20d0)
      call get_real   (npara,param,value," pta    ",pta,20d0)
      call get_real   (npara,param,value," ptl    ",ptl,20d0)
      call get_real   (npara,param,value," misset ",misset,0d0)
      call get_real   (npara,param,value," ptonium ",ptonium,0d0)

c*********************************************************************
c     Maximum pt's                                                   *
c*********************************************************************

      call get_real   (npara,param,value," ptjmax ",ptjmax,1d5)
      call get_real   (npara,param,value," ptbmax ",ptbmax,1d5)
      call get_real   (npara,param,value," ptamax ",ptamax,1d5)
      call get_real   (npara,param,value," ptlmax ",ptlmax,1d5)
      call get_real   (npara,param,value," missetmax ",missetmax,1d5)

c*********************************************************************
c     Maximum rapidity (absolute value)                              *
c*********************************************************************

      call get_real   (npara,param,value," etaj ",etaj,4d0)
      call get_real   (npara,param,value," etab ",etab,4d0)
      call get_real   (npara,param,value," etaa ",etaa,4d0)
      call get_real   (npara,param,value," etal ",etal,4d0)
      call get_real   (npara,param,value," etaonium ",etaonium,1d2)

c*********************************************************************
c     Minimum rapidity (absolute value)                              *
c*********************************************************************

      call get_real   (npara,param,value," etajmin ",etajmin,0d0)
      call get_real   (npara,param,value," etabmin ",etabmin,0d0)
      call get_real   (npara,param,value," etaamin ",etaamin,0d0)
      call get_real   (npara,param,value," etalmin ",etalmin,0d0)

c*********************************************************************
c     Minimum E's                                                   *
c*********************************************************************

      call get_real   (npara,param,value," ej ", ej, 0d0)
      call get_real   (npara,param,value," eb ", eb, 0d0)
      call get_real   (npara,param,value," ea ", ea, 0d0)
      call get_real   (npara,param,value," el ", el, 0d0)


c*********************************************************************
c     Maximum E's                                                    *
c*********************************************************************

      call get_real   (npara,param,value," ejmax ", ejmax, 1d5)
      call get_real   (npara,param,value," ebmax ", ebmax, 1d5)
      call get_real   (npara,param,value," eamax ", eamax, 1d5)
      call get_real   (npara,param,value," elmax ", elmax, 1d5)

c*********************************************************************
c     Minimum DeltaR distance                                        *
c*********************************************************************

      call get_real   (npara,param,value," drjj ",drjj,0.4d0)
      call get_real   (npara,param,value," drbb ",drbb,0.4d0)
      call get_real   (npara,param,value," drll ",drll,0.4d0)
      call get_real   (npara,param,value," draa ",draa,0.4d0)
      call get_real   (npara,param,value," drbj ",drbj,0.4d0)
      call get_real   (npara,param,value," draj ",draj,0.4d0)
      call get_real   (npara,param,value," drjl ",drjl,0.4d0)
      call get_real   (npara,param,value," drab ",drab,0.4d0)
      call get_real   (npara,param,value," drbl ",drbl,0.4d0)
      call get_real   (npara,param,value," dral ",dral,0.4d0)

c*********************************************************************
c     Maximum DeltaR distance                                        *
c*********************************************************************

      call get_real   (npara,param,value," drjjmax ",drjjmax,1d2)
      call get_real   (npara,param,value," drbbmax ",drbbmax,1d2)
      call get_real   (npara,param,value," drllmax ",drllmax,1d2)
      call get_real   (npara,param,value," draamax ",draamax,1d2)
      call get_real   (npara,param,value," drbjmax ",drbjmax,1d2)
      call get_real   (npara,param,value," drajmax ",drajmax,1d2)
      call get_real   (npara,param,value," drjlmax ",drjlmax,1d2)
      call get_real   (npara,param,value," drabmax ",drabmax,1d2)
      call get_real   (npara,param,value," drblmax ",drblmax,1d2)
      call get_real   (npara,param,value," dralmax ",dralmax,1d2)

c*********************************************************************
c     Minimum invariant mass for pairs                               *
c*********************************************************************

      call get_real   (npara,param,value," mmjj ",mmjj,0d0)
      call get_real   (npara,param,value," mmbb ",mmbb,0d0)
      call get_real   (npara,param,value," mmaa ",mmaa,0d0)
      call get_real   (npara,param,value," mmll ",mmll,0d0)

c*********************************************************************
c     Maximum invariant mass for pairs                               *
c*********************************************************************

      call get_real   (npara,param,value," mmjjmax ",mmjjmax,1d5)
      call get_real   (npara,param,value," mmbbmax ",mmbbmax,1d5)
      call get_real   (npara,param,value," mmaamax ",mmaamax,1d5)
      call get_real   (npara,param,value," mmllmax ",mmllmax,1d5)

c*********************************************************************
c     Min Maxi invariant mass for all leptons                        *
c*********************************************************************

      call get_real   (npara,param,value," mmnl    ",mmnl   ,0d0)
      call get_real   (npara,param,value," mmnlmax ",mmnlmax,1d5)

c*********************************************************************
c     Inclusive cuts                                                 *
c*********************************************************************

      call get_real   (npara,param,value," xptj ",xptj,0d0)
      call get_real   (npara,param,value," xptb ",xptb,0d0)
      call get_real   (npara,param,value," xpta ",xpta,0d0)
      call get_real   (npara,param,value," xptl ",xptl,0d0)
      call get_real   (npara,param,value," xmtcentral ",xmtc,0d0)

c*********************************************************************
c     WBF cuts                                                       *
c*********************************************************************

      call get_real   (npara,param,value," xetamin ",xetamin,0d0)
      call get_real   (npara,param,value," deltaeta",deltaeta,0d0)

c*********************************************************************
c     Jet measure cuts                                               *
c*********************************************************************

      call get_real   (npara,param,value," xqcut ",xqcut,0d0)
      call get_real   (npara,param,value," d ",D,1d0)

c*********************************************************************
c Set min pt of one heavy particle                                   *
c*********************************************************************

 	call get_real   (npara,param,value,"ptheavy",ptheavy,0d0)

c*********************************************************************
c	Pt of pairs of leptons (charged and neutrals)                *
c*********************************************************************

      call get_real   (npara,param,value," ptllmin ",ptllmin,0d0)
      call get_real   (npara,param,value," ptllmax ",ptllmax,1d5)

c*********************************************************************
c Check   the pt's of the jets sorted by pt                          *
c*********************************************************************

 	call get_real   (npara,param,value,"ptj1min",ptj1min,0d0)
 	call get_real   (npara,param,value,"ptj1max",ptj1max,1d5)
 	call get_real   (npara,param,value,"ptj2min",ptj2min,0d0)
 	call get_real   (npara,param,value,"ptj2max",ptj2max,1d5)
 	call get_real   (npara,param,value,"ptj3min",ptj3min,0d0)
 	call get_real   (npara,param,value,"ptj3max",ptj3max,1d5)
 	call get_real   (npara,param,value,"ptj4min",ptj4min,0d0)
 	call get_real   (npara,param,value,"ptj4max",ptj4max,1d5)
	call get_integer   (npara,param,value,"cutuse",cutuse,0)

c*********************************************************************
c Check  Ht                                                          *
c*********************************************************************

	call get_real   (npara,param,value,"ht2min",ht2min,0d0)
	call get_real   (npara,param,value,"ht3min",ht3min,0d0)
	call get_real   (npara,param,value,"ht4min",ht4min,0d0)
	call get_real   (npara,param,value,"ht2max",ht2max,1d5)
	call get_real   (npara,param,value,"ht3max",ht3max,1d5)
	call get_real   (npara,param,value,"ht4max",ht4max,1d5)

	call get_real   (npara,param,value,"htjmin",htjmin,0d0)
	call get_real   (npara,param,value,"htjmax",htjmax,1d5)

        call get_real   (npara,param,value,"ihtmin",ihtmin,0d0)
        call get_real   (npara,param,value,"ihtmax",ihtmax,1d5)

c*********************************************************************
c     Random Number Seed                                             *
c*********************************************************************

      call get_logical   (npara,param,value," gridrun ",gridrun,.false.)
      call get_logical   (npara,param,value," gridpack ",gridpack,.false.)
      if (gridrun.and.gridpack)then
         call get_integer   (npara,param,value," gseed ",iseed,0)
      else 
         call get_integer (npara,param,value," iseed ",iseed,0)
      endif

c************************************************************************     
c     Renormalization and factorization scales                          *
c************************************************************************     
c
c     If the following flags to .false. then event-by-event
c     scale choice is requested. In this case edit the 
c     user subroutines set_ren_scale and set_fac_scale in setpara.f

      call get_logical(npara,param,value," fixed_ren_scale ",fixed_ren_scale,.true.)
      call get_logical(npara,param,value," fixed_fac_scale ",fixed_fac_scale,.true.)
      call get_real   (npara,param,value," scale "          ,scale,91.188d0)
      call get_real   (npara,param,value," dsqrt_q2fact1 "  ,sf1  ,91.188d0)
      call get_real   (npara,param,value," dsqrt_q2fact2 "  ,sf2  ,91.188d0)

      q2fact(1) = sf1**2      ! fact scale**2 for pdf1
      q2fact(2) = sf2**2      ! fact scale**2 for pdf2     

      call get_real   (npara,param,value," scalefact "      ,scalefact, 1d0)
      call get_logical(npara,param,value," fixed_couplings ",fixed_couplings,.true.)
      call get_integer(npara,param,value," ickkw "          ,ickkw    , 0  )
      call get_logical(npara,param,value," chcluster ",chcluster,.false.)
c     ktscheme for xqcut: 1: pT/Durham kT; 2: pythia pTE/Durham kT
      call get_integer (npara,param,value," ktscheme ",ktscheme,1)

      if(ickkw.gt.0)then
         call get_real   (npara,param,value," alpsfact "       ,alpsfact , 1d0)
         call get_logical   (npara,param,value," pdfwgt "       ,pdfwgt , .true.)
      endif
      if(ickkw.eq.2)then
        call get_integer(npara,param,value," highestmult "    ,nhmult, 0)
        call get_string (npara,param,value," issgridfile ",issgridfile,'issudgrid.dat')
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
      call get_real   (npara,param,value," polbeam1 " ,pb1,0d0)
      call get_real   (npara,param,value," polbeam2 " ,pb2,0d0)

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
          call setpara(param_card_name)
          asmz=G**2/(16d0*atan(1d0))
          write(*,*) 'Old value of alpha_s from param_card: ',asmz
          call pdfwrap
          write(*,*) 'New value of alpha_s from PDF ',pdlabel,':',asmz
      else
          call setpara(param_card_name)
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
          idbmup(i)=idup(i,1,1)
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
           pdfgup(i)=0
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
        pdfgup(i)=0
        pdfsup(i)=mpdf
      enddo

      return
      end

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
      include 'genps.inc'
      include 'run_config.inc'
      include 'PDF/pdf.inc'
      include 'vector.inc'      ! defines VECSIZE_MEMMAX
      include 'run.inc'
      include 'alfas.inc'
      include 'MODEL/coupl.inc' ! needs VECSIZE_MEMMAX (defined in vector.inc)

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
      integer*8          iseed
      common /to_seed/ iseed
c
c----------
c     start
c----------
c
c     read the run_card.dat
c
      include 'run_card.inc'

c     if no matching ensure that no pdfreweight are done
      if (ickkw.eq.0) pdfwgt = .false.

      q2fact(1) = sf1**2      ! fact scale**2 for pdf1
      q2fact(2) = sf2**2      ! fact scale**2 for pdf2     

      if(pb1.ne.0d0)then
         if (abs(lpp(1)).eq.1.or.abs(lpp(1)).eq.2)then
            write(*,*) 'proton/anti-proton beam polarization are not allowed'
            stop 1
         endif
         pol(1)=sign(1+abs(pb1)/100d0,pb1)
      endif
      if(pb2.ne.0d0)then
         if (abs(lpp(2)).eq.1.or.abs(lpp(2)).eq.2)then
            write(*,*) 'proton/anti-proton beam polarization are not allowed'
            stop 1
         endif
         pol(2)=sign(1+abs(pb2)/100d0,pb2)
      endif

      
      if(pb1.ne.0d0.and.lpp(1).eq.0) pol(1)=sign(1+abs(pb1)/100d0,pb1)
      if(pb2.ne.0d0.and.lpp(2).eq.0) pol(2)=sign(1+abs(pb2)/100d0,pb2)

      if(pb1.ne.0.or.pb2.ne.0) write(*,*) 'Setting beam polarization ',
     $     sign((abs(pol(1))-1)*100,pol(1)),
     $     sign((abs(pol(2))-1)*100,pol(2))


      if(pdlabel.eq.'eva') then
            ! pbX=-100 (pure LH beam) => fLpol=1.0 (in eva)
            ! pbX=0    (RH + LH beam) => fLpol=0.5 (in eva)
            ! pbX=+100 (pure RH beam) => fLpol=0.0 (in eva)
            pol(1) = (-1d0/200d0)*pb1 + 0.5d0
            pol(2) = (-1d0/200d0)*pb2 + 0.5d0
      else
            if(pdsublabel(1).eq.'eva') then
                  pol(1) = (-1d0/200d0)*pb1 + 0.5d0
            endif
            if(pdsublabel(2).eq.'eva') then
                  pol(2) = (-1d0/200d0)*pb2 + 0.5d0
            endif
      endif

c !!! Default behavior changed (MH, Aug. 07) !!!
c If no pdf, read the param_card and use the value from there and
c order of alfas running = 2

      if(lpp(1).ne.0.or.lpp(2).ne.0) then
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

C     If use_syst, ensure that all variational parameters are 1
c           In principle this should be always the case since the
c           banner.py is expected to correct such wrong run_card.
      if(use_syst)then
c         if(scalefact.ne.1)then
c            write(*,*) 'Warning: use_syst=T, setting scalefact to 1'
c            scalefact=1
c         endif
         if(alpsfact.ne.1)then
            write(*,*) 'Warning: use_syst=T, setting alpsfact to 1'
            alpsfact=1
         endif
      endif

C       Fill common block for Les Houches init info
      do i=1,2
        if(lpp(i).eq.1.or.lpp(i).eq.2) then
          if (nb_proton(i).eq.1.and.nb_neutron(i).eq.0) then
              idbmup(i)=2212
          elseif (nb_proton(i).eq.0.and.nb_neutron(i).eq.1) then
              idbmup(i)=2112
          else
              idbmup(i) = 1000000000 + (nb_proton(i)+nb_neutron(i))*10
     $                               + nb_proton(i)*10000
          endif
        elseif(lpp(i).eq.-1.or.lpp(i).eq.-2) then
          if (nb_proton(i).eq.1.and.nb_neutron(i).eq.0) then
              idbmup(i)=-2212
          else
              idbmup(i) = -1*(1000000000 + (nb_proton(i)+nb_neutron(i))*10
     $                                    + nb_proton(i)*10000)
          endif
        elseif(lpp(i).eq.3) then
          idbmup(i)=11
        elseif(lpp(i).eq.-3) then
          idbmup(i)=-11
        elseif(lpp(i).eq.4) then
            idbmup(i)=13
        elseif(lpp(i).eq.-4) then
            idbmup(i)=-13
        elseif(lpp(i).eq.0) then
          idbmup(i)=idup(i,1,1)
        else
          idbmup(i)=lpp(i)
        endif
      enddo
      ebmup(1)=ebeam(1)
      ebmup(2)=ebeam(2)
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

      parameter (npdfs=21)
      character*7 pdflabs(npdfs)
      data pdflabs/
     $   'none',
     $   'eva',
     $   'iww',
     $   'edff',
     $   'chff',
     $   'dressed', 
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
     $   'cteq6l1',     
     $   'nn23lo',
     $   'nn23lo1',
     $   'nn23nlo'/
      integer numspdf(npdfs)
      data numspdf/
     $   00000, ! none
     $   00000, ! eva
     $   00000, ! iww
     $   00000, ! edff
     $   00000, ! chff
     $   00000, ! dressed
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
     $   10042,
     $   246800,
     $   247000,
     $   244800/


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


      double precision FUNCTION DDILOG(X)
*
* $Id: imp64.inc,v 1.1.1.1 1996/04/01 15:02:59 mclareni Exp $
*
* $Log: imp64.inc,v $
* Revision 1.1.1.1  1996/04/01 15:02:59  mclareni
* Mathlib gen
*
*
* imp64.inc
*
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION C(0:19)
      PARAMETER (Z1 = 1, HF = Z1/2)
      PARAMETER (PI = 3.14159 26535 89793 24D0)
      PARAMETER (PI3 = PI**2/3, PI6 = PI**2/6, PI12 = PI**2/12)
      DATA C( 0) / 0.42996 69356 08136 97D0/
      DATA C( 1) / 0.40975 98753 30771 05D0/
      DATA C( 2) /-0.01858 84366 50145 92D0/
      DATA C( 3) / 0.00145 75108 40622 68D0/
      DATA C( 4) /-0.00014 30418 44423 40D0/
      DATA C( 5) / 0.00001 58841 55418 80D0/
      DATA C( 6) /-0.00000 19078 49593 87D0/
      DATA C( 7) / 0.00000 02419 51808 54D0/
      DATA C( 8) /-0.00000 00319 33412 74D0/
      DATA C( 9) / 0.00000 00043 45450 63D0/
      DATA C(10) /-0.00000 00006 05784 80D0/
      DATA C(11) / 0.00000 00000 86120 98D0/
      DATA C(12) /-0.00000 00000 12443 32D0/
      DATA C(13) / 0.00000 00000 01822 56D0/
      DATA C(14) /-0.00000 00000 00270 07D0/
      DATA C(15) / 0.00000 00000 00040 42D0/
      DATA C(16) /-0.00000 00000 00006 10D0/
      DATA C(17) / 0.00000 00000 00000 93D0/
      DATA C(18) /-0.00000 00000 00000 14D0/
      DATA C(19) /+0.00000 00000 00000 02D0/
      IF(X .EQ. 1) THEN
       H=PI6
      ELSEIF(X .EQ. -1) THEN
       H=-PI12
      ELSE
       T=-X
       IF(T .LE. -2) THEN
        Y=-1/(1+T)
        S=1
        A=-PI3+HF*(LOG(-T)**2-LOG(1+1/T)**2)
       ELSEIF(T .LT. -1) THEN
        Y=-1-T
        S=-1
        A=LOG(-T)
        A=-PI6+A*(A+LOG(1+1/T))
       ELSE IF(T .LE. -HF) THEN
        Y=-(1+T)/T
        S=1
        A=LOG(-T)
        A=-PI6+A*(-HF*A+LOG(1+T))
       ELSE IF(T .LT. 0) THEN
        Y=-T/(1+T)
        S=-1
        A=HF*LOG(1+T)**2
       ELSE IF(T .LE. 1) THEN
        Y=T
        S=1
        A=0
       ELSE
        Y=1/T
        S=-1
        A=PI6+HF*LOG(T)**2
       ENDIF
       H=Y+Y-1
       ALFA=H+H
       B1=0
       B2=0
       DO 1 I = 19,0,-1
       B0=C(I)+ALFA*B1-B2
       B2=B1
    1  B1=B0
       H=-(S*(B0-H*B2)+A)
      ENDIF
      DDILOG=H
      RETURN
      END

C============================================================================
C                CTEQ-TEA Parton Distribution Functions: version 2015
C                       October 1, 2015
C
C   When using these PDFs, please cite the references below
C   
C
C   This package provides a standard interface for CT10, CT12
C   (unpublished), and CT14 parton distribution functions,
C   including CT14QED.
C
C   The following sets of CTEQ PDF table files can be computed 
C    with this program: 
C               PDF                             References         
C   (1) 1+50 sets of CT10 NNLO PDF's;             [1]
C   (2' ) 1+52 sets of CT10 NLO PDF's;            [2]
C   (2'') 1+52 sets of CT10W NLO PDF's;           [2]
C   (3) 4 sets of CT10W NNLO and NLO PDF's        [2]
C       with alternative alpha_s values; 
C   (4) 1+56 sets of CT14 NNLO PDF's;             [3]
C   (5) 1+56 sets of CT14 NLO PDF's;              [3]
C   (6) 2 sets of CT14 LO PDF's;                  [3]
C   (7) 11 CT14 NNLO sets and 11 CT14 NLO sets    [3]
C       with alternative alpha_s values         
C   (8) 3 CT14 NNLO and 3 CT14 NLO sets           [3]
C       with up to 3, 4, and 6 active quark flavors
C   (9) 4 CT14 NNLO sets with intrinsic charm     [X]
C   (10) 6 CT14QED sets (LO QED, NLO QCD)         [4]
C        3 each for proton and neutron

C   References
C   [1] J. Gao, M. Guzzi, J. Huston, H.-L. Lai, Z. Li, P. M. Nadolsky, 
C       J. Pumplin, D. Stump, C.-P. Yuan,  arXiv:1302.6246 [hep-ph]
C   [2] H.-L. Lai, M. Guzzi, J. Huston, Z. Li, P. M. Nadolsky,
C       J. Pumplin, and C.-P. Yuan, arXiv: 1007.2241 [hep-ph]
C   [3] S. Dulat, T.-J. Hou, J. Gao, M. Guzzi, J. Huston, 
C       P. M. Nadolsky, J. Pumplin, C. Schmidt, D. Stump, and
C       C.-P. Yuan, arXiv:1506.07443
C   [4] C. Schmidt, J. Pumplin, D. Stump, C.-P. Yuan, arXiv:1509.02905
C 
C ===========================================================================
C   The table grids are generated for 
C    *  10^-9 < x < 1 and 1.3 < Q < 10^5 (GeV).
C
C   PDF values outside of the above range are returned using extrapolation.
C
C   The Table_Files are assumed to be in the working directory.
C
C   Before using the PDF, it is necessary to do the initialization by
C       Call SetCT14(Tablefile)
C   where Tablefile is a 40-character text string with the name of the
c   the desired PDF specified in the above table.table (.pds) file
C
C   Other provided functions include:
C   The function CT14Pdf (Iparton, X, Q)
C     returns the parton distribution inside the proton for parton [Iparton]
C     at [X] Bjorken_X and scale [Q] (GeV) in PDF set [Iset].
C     Iparton  is the parton label (5, 4, 3, 2, 1, 0, -1, ......, -5)
C                              for (b, c, s, d, u, g, u_bar, ..., b_bar).
C     Iparton=10 is the label for the photon, if it is available.
C
C   The function CT14Alphas (Q) 
C     returns the value of the QCD coupling strength alpha_s(Q) at
C     an energy scale Q. The alpha_s value is obtained from the interpolation
C     of a numerical table of alpha_s included in each .pds file. 
C     In the PDF fit, the alpha_s values are obtained at each momentum
c     scale by evolution in the HOPPET program at the respective QCD order 
C     (NLO or NNLO). The table of alpha_s values at discrete Q values
C     is included in the input .pds file. The function CT14Alphas
c     estimates alpha_s at an arbitrary Q value, which agrees
c     with the direct evolution by HOPPET within a fraction of percent
c     point at typical Q. 
C 
C   The function CT14Mass(i)
c     returns the value of the quark mass for the i-th flavor.
c     The flavors are:
c     1  2  3  4  5  6
c     u  d  s  c  b  t
c
C   Values of various PDF parameters assumed in the computation of the
c    PDFs can be obtained by 
C     Call CT14GetPars( xmin,Qini,Qmax,Nloops,Nfl,IHDN,Lpho),
C   which returns
c     xmin, the minimal value of x;
c     Qmin,  the initial Q scale for the PDF evolution;  
c     Qmax,  the maximal Q scale included in the PDF table;
c     Nloop, the number of QCD loops (order of the PDF in the QCD coupling);
c     Nfl,   the maximal number of quark flavors assumed in the PDF and 
c            alpha_s evolution;
C     IHDN,  Hadron (proton = 1,  neutron = 0)
C     Lpho,  Logical variable, which returns .true. if LO QED evolution
C            and photon PDF included, otherwise .false.

C   These programs, as provided, are in double precision.  By removing the
C   "Implicit Double Precision" lines, they can also be run in single
C   precision.
C   If you have detailed questions concerning these CT14 distributions,
C   or if you find problems/bugs using this package, direct inquires to
C   nadolsky@smu.edu.
C
C===========================================================================

      Function CT14Pdf (Iparton, X, Q)
      Implicit Double Precision (A-H,O-Z)
      Logical Warn, Lpho, Lbig, Lnp
      integer isetch, ipdsformat
      Common
     > / CtqPar2 / Nx, Nt, NfMx, MxVal, Ihadron
     > / QCDtbl /  AlfaQ, Qalfa, Ipk, Iorder, Nfl !for external use
     >  /Setchange/ Isetch, ipdsset, ipdsformat
      Common / Photon / Lpho

      Data Warn /.true./
      Data Qsml /.3d0/
      save Warn

      if (ipdsset.ne.1) 
     >  STOP 'CT14Pdf: the PDF table was not initialized'

      If (X .lt. 0d0 .or. X .gt. 1D0) Then
        Print *, 'X out of range in CT14Pdf: ', X
        CT14Pdf = 0D0
        Return
      Endif

      If (Q .lt. Qsml) Then
        Print *, 'Q out of range in CT14Pdf: ', Q
        Stop
      Endif

      Lbig = (abs(Iparton).gt.NfMx)
      Lnp = (Iparton.ne.10)
      If (Lbig.and.((Lpho.and.Lnp).or.(.not.Lpho))) Then
        If (Warn) Then
C        print a warning for calling extra flavor
          Warn = .false.
          Print *, 'Warning: Iparton out of range in CT14Pdf! '
          Print *, 'Iparton, MxFlvN0: ', Iparton, NfMx
        Endif
        CT14Pdf = 0D0
      else

        CT14Pdf = PartonX12 (Iparton, X, Q)
        if (CT14Pdf.lt.0D0) CT14Pdf = 0D0
      endif                     !if (abs(Iparton...

      Return

C                             ********************
      End

      Subroutine SetCT14(Tablefile)
      Implicit Double Precision (A-H,O-Z)
      Character Tablefile*40
      Common /Setchange/ Isetch, ipdsset, ipdsformat
      data ipdsset, ipdsformat/0,0/
      save

      IU= NextUnb()
c      Open(IU, File=Tablefile, Status='OLD', Err=100)
       call OpenData(TableFile)
      Call Readpds0 (IU)
      Close (IU)
      Isetch=1; ipdsset=1
      Return

 100  Print *, ' Data file ', Tablefile, ' cannot be opened '
     >  //'in SetCT14!!'
      Stop
C                             ********************
      End

      subroutine CT14GetPars(xmin,Qini,Qmax,Nloops,Nfl,IHDN,Lphoton)
c Get various parameters associated with the PDF grid
c Output: xmin  is the minimal value of x 
c         Qmin  is the initial Q scale  
c         Qmax  is the maximal Q scale
c         Nloop is the number of QCD loops
c         Nfl   is the maximal number of quark flavors
c         IHDN  is the hadron type (proton: IHDN=1, neutron: IHDN=0)
c         Lphoton  is .true. if LO QED and photon included,
c               otherwise .false
      implicit none
      double precision Qini0, Qmax0, Xmin0, xmin, Qini, Qmax
      integer Nloops, Ipk, Iorder, Nfl,Nfl0, IHDN
      integer Nx, Nt, Nfmx, Mxval, Ihadron
      double precision AlfaQ, Qalfa
      Logical Lpho, Lphoton

      common / XQrange / Qini0, Qmax0, Xmin0
      common / QCDtbl /  AlfaQ, Qalfa, Ipk, Iorder, Nfl0
      common / CtqPar2 / Nx, Nt, NfMx, MxVal, Ihadron
      common / Photon/ Lpho
     
      Qini=Qini0; Qmax=Qmax0; Xmin=Xmin0
      Nloops=Iorder-1; Nfl=Nfl0; Lphoton=Lpho
      IHDN=Ihadron

      return 
      end


      Function CT14Alphas (QQ)

      Implicit Double Precision (A-H,O-Z)

Carl  The following match those in mEvlPac.F for v05t:
      PARAMETER (MXX = 204, MXQ = 50, MXF = 6, MaxVal=4)
c      PARAMETER (MXPQX = (MXF+1+MaxVal) * MXQ * MXX)
      double precision Alsout
      
      Common
     > / CtqPar1 / qBase,XV(0:MXX), TV(0:MXQ),AlsCTEQ(0:mxq)
     > / CtqPar2 / Nx, Nt, NfMx, MxVal, Ihadron
     > / CtqPar3 / UPD(0:mxx-1,0:mxq-1,-(mxf+1):(MaxVal))
     > / XQrange / Qini, Qmax, Xmin
     > /Setchange/ Isetch, ipdsset, ipdsformat

      Data Q, JQ /-1D0, 0/
      save

      if (ipdsset.ne.1) 
     >  STOP 'CT14Alphas: the PDF table was not initialized'

      
      if (ipdsformat.lt.11) then
        print *
        print *, 
     >    'STOP in CT14alphas: the PDF table file has an older format'
        print *,
     >    'and does not include the table of QCD coupling values.'
        print *, 
     >    'You can still compute the PDFs, but do not call'
        print *,
     >    'the CT14alphas function for the interpolation of alpha_s.'
        stop
      endif

      Q = QQ
      tt = log(log(Q/qBase))

c         --------------   Find lower end of interval containing Q, i.e.,
c                          get jq such that qv(jq) .le. q .le. qv(jq+1)...
      JLq = -1
      JU = NT+1
 13   If (JU-JLq .GT. 1) Then
        JM = (JU+JLq) / 2
        If (tt .GE. TV(JM)) Then
            JLq = JM
          Else
            JU = JM
          Endif
          Goto 13
       Endif

      If     (JLq .LE. 0) Then
         Jq = 0
      Elseif (JLq .LE. Nt-2) Then
C                                  keep q in the middle, as shown above
         Jq = JLq - 1
      Else
C                         JLq .GE. Nt-1 case:  Keep at least 4 points >= Jq.
        Jq = Nt - 3

      Endif
C                                 This is the interpolation variable in Q
      Call Polint4F (TV(jq), AlsCTEQ(jq), tt, Alsout)
      
      CT14Alphas = Alsout
      
      Return
C                                       ********************
      End


      function CT14Mass(i)
c     Returns the value of the quark mass for the i-th flavor 
c     The flavors are:
c     1  2  3  4  5  6
c     u  d  s  c  b  t
      implicit none
      double precision CT14Mass, Amass
      integer  Isetch, ipdsset, i, ipdsformat
      common/Setchange/ Isetch, ipdsset, ipdsformat
     >  / Masstbl / Amass(6)


      if (ipdsset.ne.1) 
     >  STOP 'CT14Mass: the PDF table was not initialized'

      CT14Mass = Amass(i)

      return 
      end


      Subroutine Readpds0 (Nu)
      Implicit Double Precision (A-H,O-Z)
      Character Line*80
      integer ipdsformat
Carl  The following match those in mEvlPac.F for v05t:
      PARAMETER (MXX = 204, MXQ = 50, MXF = 6, MaxVal=4)
c      PARAMETER (MXPQX = (MXF+1+MaxVal) * MXQ * MXX)
      double precision qv(0:mxq)
      logical Lpho

      Common
     > / CtqPar1 / qBase,XV(0:MXX),TV(0:MXQ),AlsCTEQ(0:mxq)
     > / CtqPar2 / Nx, Nt, NfMx, MxVal, Ihadron
     > / CtqPar3 / UPD(0:mxx-1,0:mxq-1,-(mxf+1):(MaxVal))
     > / XQrange / Qini, Qmax, Xmin
     > / Masstbl / Amass(6)
     > / QCDtbl /  AlfaQ, Qalfa, Ipk, Iorder, Nfl !for external use
     > /Setchange/ Isetch, ipdsset, ipdsformat
      common / Photon/ Lpho

Carl  Default is no photon
      Ipho=0
      Lpho=.false.

      Read  (Nu, '(A)') Line
      Read  (Nu, '(A)') Line

      if (Line(1:11) .eq. '  ipk, Ordr') then !post-CT10 .pds format;
c Set alphas(MZ) at scale Zm, quark masses, and evolution type
        ipdsformat = 10           !Post-CT10 .pds format
        Read (Nu, *) ipk, Dr, Qalfa, AlfaQ, (amass(i),i=1,6) 
        Iorder = Nint(Dr)        
        read (Nu, '(A)') Line
        if (Line(1:7) .eq. '  IMASS' ) then
          ipdsformat = 11         !CT12 .pds format
          read (Nu, '(A)') Line
          read (Line,*,End=22,Err=22)
     >           aimass, fswitch, N0, IHDN, N0, Nfmx, MxVal, Ipho
          goto 26
22        continue
          Ipho = 0
          read (Line,*)
     >           aimass, fswitch, N0, IHDN, N0, Nfmx, MxVal
26        continue
          Nfl=Nfmx
          Ihadron=IHDN
          IF (Ipho.eq.1) then
             Lpho=.true.
          endif
        else                      !Pre-CT12 format
          Read  (Nu, *) N0, N0, N0, NfMx, MxVal
        endif                     !Line(1:7)
        
      else                        !old .pds format;
        ipdsformat = 6            !CTEQ6.6 .pds format; alpha_s  is not specified
        Read (Nu, *) Dr, fl, Alambda, (amass(i),i=1,6)  !set Lambda_QCD
        Iorder = Nint(Dr); Nfl = Nint(fl)

        Read  (Nu, '(A)') Line
        Read  (Nu, *) dummy,IHDN,dummy, NfMx, MxVal, N0
        Ihadron=IHDN
      endif                       !Line(1:11...
      
      Read  (Nu, '(A)') Line
      Read  (Nu, *) NX,  NT, N0, NG, N0
      
      if (ng.gt.0) Read  (Nu, '(A)') (Line, i=1,ng+1)

      Read  (Nu, '(A)') Line
      if (ipdsformat.ge.11) then  !CT12 format with alpha_s values
        Read  (Nu, *) QINI, QMAX, (qv(i),TV(I), AlsCTEQ(I), I =0, NT)
      else                        !pre-CT12 format
        Read  (Nu, *) QINI, QMAX, (qv(i),TV(I), I =0, NT)
      endif                       !ipdsformat.ge.11

c check that qBase is consistent with the definition of Tv(0:nQ) for 2 values of Qv
      qbase1 = Qv(1)/Exp(Exp(Tv(1)))
      qbase2 = Qv(nT)/Exp(Exp(Tv(NT)))
      if (abs(qbase1-qbase2).gt.1e-5) then
        print *, 'Readpds0: something wrong with qbase'
        print *,'qbase1, qbase2=',qbase1,qbase2
        stop
      else
        qbase=(qbase1+qbase2)/2.0d0
      endif                     !abs(qbase1...

      Read  (Nu, '(A)') Line
      Read  (Nu, *) XMIN, aa, (XV(I), I =1, NX)
      XV(0)=0D0
      
C                  Since quark = anti-quark for nfl>2 at this stage,
C                  we Read  out only the non-redundent data points
C                  No of flavors = NfMx sea + 1 gluon + Nfval valence+ipho
Carl               including the photon if (Lpho.eq.true)
Carl  Nfmin = the lowest parton index in the array (It is negative.)
      Nfmin = -(NfMx+ipho)
Carl
c      Nblk = (NX+1) * (NT+1)
c      Npts =  Nblk  * (NfMx+1+MxVal)
      Read  (Nu, '(A)') Line
      READ  (Nu, *, IOSTAT=IRR)
     >   (((UPD(ix,it,ip), ix=0,NX),it=0,NT),ip=NFmin,MxVal)
c      Read  (Nu, *, IOSTAT=IRET) (UPD(I), I=1,Npts)

      Return
C                        ****************************
      End

      Function PartonX12 (IPRTN, XX, QQ)

c  Given the parton distribution function in the array U in
c  COMMON / PEVLDT / , this routine interpolates to find
c  the parton distribution at an arbitray point in x and q.
c
      Implicit Double Precision (A-H,O-Z)

Carl  The following match those in mEvlPac.F for v05t:
      PARAMETER (MXX = 204, MXQ = 50, MXF = 6, MaxVal=4)
c      PARAMETER (MXPQX = (MXF+1+MaxVal) * MXQ * MXX)

      Common
     > / CtqPar1 / qBase, XV(0:MXX), TV(0:MXQ),AlsCTEQ(0:mxq)
     > / CtqPar2 / Nx, Nt, NfMx, MxVal, Ihadron
     > / CtqPar3 / UPD(0:mxx-1,0:mxq-1,-(mxf+1):(MaxVal))
     > / XQrange / Qini, Qmax, Xmin
     > /Setchange/ Isetch, ipdsset, ipdsformat

      Dimension fvec(4), fij(4)
      Dimension xvpow(0:mxx)
      Data OneP / 1.00001 /
      Data xpow / 0.3d0 /       !**** choice of interpolation variable
      Data nqvec / 4 /
      Data ientry / 0 /
      Data X, Q, JX, JQ /-1D0, -1D0, 0, 0/
      Save xvpow
      Save X, Q, JX, JQ, JLX, JLQ
      Save ss, const1, const2, const3, const4, const5, const6
      Save sy2, sy3, s23, tt, t12, t13, t23, t24, t34, ty2, ty3
      Save tmp1, tmp2, tdet

c store the powers used for interpolation on first call...
      if(Isetch .eq. 1) then
         Isetch = 0

         xvpow(0) = 0D0
         do i = 1, nx
            xvpow(i) = xv(i)**xpow
         enddo
      elseIf((XX.eq.X).and.(QQ.eq.Q)) then
      	goto 99
      endif

      X = XX
      Q = QQ
      tt = log(log(Q/qBase))

c      -------------    find lower end of interval containing x, i.e.,
c                       get jx such that xv(jx) .le. x .le. xv(jx+1)...
      JLx = -1
      JU = Nx+1
 11   If (JU-JLx .GT. 1) Then
         JM = (JU+JLx) / 2
         If (X .Ge. XV(JM)) Then
            JLx = JM
         Else
            JU = JM
         Endif
         Goto 11
      Endif
C                     Ix    0   1   2      Jx  JLx         Nx-2     Nx
C                           |---|---|---|...|---|-x-|---|...|---|---|
C                     x     0  Xmin               x                 1
C
      If     (JLx .LE. -1) Then
        Print '(A,1pE12.4)','Severe error: x <= 0 in PartonX12! x = ',x
        Stop
      ElseIf (JLx .Eq. 0) Then
         Jx = 0
      Elseif (JLx .LE. Nx-2) Then

C                For interior points, keep x in the middle, as shown above
         Jx = JLx - 1
      Elseif (JLx.Eq.Nx-1 .or. x.LT.OneP) Then

C                  We tolerate a slight over-shoot of one (OneP=1.00001),
C              perhaps due to roundoff or whatever, but not more than that.
C                                      Keep at least 4 points >= Jx
         Jx = JLx - 2
      Else
        Print '(A,1pE12.4)','Severe error: x > 1 in PartonX12! x = ',x
        Stop
      Endif
C          ---------- Note: JLx uniquely identifies the x-bin; Jx does not.

C                       This is the variable to be interpolated in
      ss = x**xpow

      If (JLx.Ge.2 .and. JLx.Le.Nx-2) Then

c     initiation work for "interior bins": store the lattice points in s...
      svec1 = xvpow(jx)
      svec2 = xvpow(jx+1)
      svec3 = xvpow(jx+2)
      svec4 = xvpow(jx+3)

      s12 = svec1 - svec2
      s13 = svec1 - svec3
      s23 = svec2 - svec3
      s24 = svec2 - svec4
      s34 = svec3 - svec4

      sy2 = ss - svec2
      sy3 = ss - svec3

c constants needed for interpolating in s at fixed t lattice points...
      const1 = s13/s23
      const2 = s12/s23
      const3 = s34/s23
      const4 = s24/s23
      s1213 = s12 + s13
      s2434 = s24 + s34
      sdet = s12*s34 - s1213*s2434
      tmp = sy2*sy3/sdet
      const5 = (s34*sy2-s2434*sy3)*tmp/s12
      const6 = (s1213*sy2-s12*sy3)*tmp/s34

      EndIf

c         --------------Now find lower end of interval containing Q, i.e.,
c                          get jq such that qv(jq) .le. q .le. qv(jq+1)...
      JLq = -1
      JU = NT+1
 12   If (JU-JLq .GT. 1) Then
         JM = (JU+JLq) / 2
         If (tt .GE. TV(JM)) Then
            JLq = JM
         Else
            JU = JM
         Endif
         Goto 12
       Endif

      If     (JLq .LE. 0) Then
         Jq = 0
      Elseif (JLq .LE. Nt-2) Then
C                                  keep q in the middle, as shown above
         Jq = JLq - 1
      Else
C                         JLq .GE. Nt-1 case:  Keep at least 4 points >= Jq.
        Jq = Nt - 3

      Endif
C                                   This is the interpolation variable in Q

      If (JLq.GE.1 .and. JLq.LE.Nt-2) Then
c                                        store the lattice points in t...
      tvec1 = Tv(jq)
      tvec2 = Tv(jq+1)
      tvec3 = Tv(jq+2)
      tvec4 = Tv(jq+3)

      t12 = tvec1 - tvec2
      t13 = tvec1 - tvec3
      t23 = tvec2 - tvec3
      t24 = tvec2 - tvec4
      t34 = tvec3 - tvec4

      ty2 = tt - tvec2
      ty3 = tt - tvec3

      tmp1 = t12 + t13
      tmp2 = t24 + t34

      tdet = t12*t34 - tmp1*tmp2

      EndIf


c get the pdf function values at the lattice points...

 99   If (Iprtn.eq.10) then
         Ip = -nfmx-1
      else If (Iprtn .Gt. MxVal) Then
         Ip = - Iprtn
      Else
         Ip = Iprtn
      EndIf

      Do it = 1, nqvec
        J1  = jq-1+it

       If (Jx .Eq. 0) Then
C                      For the first 4 x points, interpolate x^2*f(x,Q)
C                      This applies to the two lowest bins JLx = 0, 1
C            We can not put the JLx.eq.1 bin into the "interrior" section
C                           (as we do for q), since Upd(J1) is undefined.
         fij(1) = 0
         fij(2) = Upd(1,j1,ip) * XV(1)**2
         fij(3) = Upd(2,j1,ip) * XV(2)**2
         fij(4) = Upd(3,j1,ip) * XV(3)**2
C
C                 Use Polint which allows x to be anywhere w.r.t. the grid

         Call Polint4F (XVpow(0), Fij(1), ss, Fx)

         If (x .GT. 0D0)  Fvec(it) =  Fx / x**2
C                                              Pdf is undefined for x.eq.0
       ElseIf  (JLx .Eq. Nx-1) Then
C                                                This is the highest x bin:

        Call Polint4F (XVpow(Nx-3), Upd(jx,j1,ip), ss, Fx)

        Fvec(it) = Fx

       Else
C                       for all interior points, use Jon's in-line function
C                              This applied to (JLx.Ge.2 .and. JLx.Le.Nx-2)
         sf2 = Upd(jx+1,J1,ip)
         sf3 = Upd(jx+2,J1,ip)

         g1 =  sf2*const1 - sf3*const2
         g4 = -sf2*const3 + sf3*const4

         Fvec(it) = (const5*(Upd(jx,j1,ip)-g1)
     &               + const6*(Upd(jx+3,j1,ip)-g4)
     &               + sf2*sy3 - sf3*sy2) / s23

       Endif

      enddo
C                                   We now have the four values Fvec(1:4)
c     interpolate in t...

      If (JLq .LE. 0) Then
C                         1st Q-bin, as well as extrapolation to lower Q
        Call Polint4F (TV(0), Fvec(1), tt, ff)

      ElseIf (JLq .GE. Nt-1) Then
C                         Last Q-bin, as well as extrapolation to higher Q
        Call Polint4F (TV(Nt-3), Fvec(1), tt, ff)
      Else
C                         Interrior bins : (JLq.GE.1 .and. JLq.LE.Nt-2)
C       which include JLq.Eq.1 and JLq.Eq.Nt-2, since Upd is defined for
C                         the full range QV(0:Nt)  (in contrast to XV)
        tf2 = fvec(2)
        tf3 = fvec(3)

        g1 = ( tf2*t13 - tf3*t12) / t23
        g4 = (-tf2*t34 + tf3*t24) / t23

        h00 = ((t34*ty2-tmp2*ty3)*(fvec(1)-g1)/t12
     &    +  (tmp1*ty2-t12*ty3)*(fvec(4)-g4)/t34)

        ff = (h00*ty2*ty3/tdet + tf2*ty3 - tf3*ty2) / t23
      EndIf

      PartonX12 = ff

      Return
C                                       ********************
      End


      SUBROUTINE POLINT4F (XA,YA,X,Y)

      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
C  The POLINT4 routine is based on the POLINT routine from "Numerical Recipes",
C  but assuming N=4, and ignoring the error estimation.
C  suggested by Z. Sullivan.
      DIMENSION XA(*),YA(*)

      H1=XA(1)-X
      H2=XA(2)-X
      H3=XA(3)-X
      H4=XA(4)-X

      W=YA(2)-YA(1)
      DEN=W/(H1-H2)
      D1=H2*DEN
      C1=H1*DEN

      W=YA(3)-YA(2)
      DEN=W/(H2-H3)
      D2=H3*DEN
      C2=H2*DEN

      W=YA(4)-YA(3)
      DEN=W/(H3-H4)
      D3=H4*DEN
      C3=H3*DEN

      W=C2-D1
      DEN=W/(H1-H3)
      CD1=H3*DEN
      CC1=H1*DEN

      W=C3-D2
      DEN=W/(H2-H4)
      CD2=H4*DEN
      CC2=H2*DEN

      W=CC2-CD1
      DEN=W/(H1-H4)
      DD1=H4*DEN
      DC1=H1*DEN

      If((H3+H4).lt.0D0) Then
         Y=YA(4)+D3+CD2+DD1
      Elseif((H2+H3).lt.0D0) Then
         Y=YA(3)+D2+CD1+DC1
      Elseif((H1+H2).lt.0D0) Then
         Y=YA(2)+C2+CD1+DC1
      ELSE
         Y=YA(1)+C1+CC1+DC1
      ENDIF

      RETURN
C               *************************
      END

      Function NextUnb()
C                                 Returns an unallocated FORTRAN i/o unit.
      Logical EX
C
      Do 10 N = 10, 300
         INQUIRE (UNIT=N, OPENED=EX)
         If (.NOT. EX) then
            NextUnb = N
            Return
         Endif
 10   Continue
      Stop ' There is no available I/O unit. '
C               *************************
      End
C


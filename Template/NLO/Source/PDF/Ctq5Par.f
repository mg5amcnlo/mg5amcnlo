C   CTEQ5M1 and CTEQ5L Parton Distribution Functions in Parametrized Form
C                             
C               September 15, 1999
C
C   Ref: "GLOBAL QCD ANALYSIS OF PARTON STRUCTURE OF THE NUCLEON:
C         CTEQ5 PPARTON DISTRIBUTIONS"
C   hep-ph/9903282
C
C   The CTEQ5M1 set given here is an updated version of the original CTEQ5M
C     set posted, in the table version, on the Web page of CTEQ.
C     The differences between CTEQ5M and CTEQ5M1 are insignificant for almost
C     all applications. 
C   The improvement is in the QCD evolution which is now more accurate, and
C   which agrees completely with the benchmark work of the HERA 96/97 Workshop.

C   The differences between the parametrized and the corresponding table ver-
C sions (on which it is based) are of similar order as between the two version.
C    
C!! Because accurate parametrizations over a wide range of (x,Q) is hard to
C   obtain, only the most widely used sets CTEQ5M and CTEQ5L are available 
C   in parametrized form for now. 

C   These parametrizations were obtained by Jon Pumplin.
C
C                    ******************************
C  Iset   PDF        Description                 Alpha_s(Mz)  Lam4  Lam5
C ---------------------------------------------------------------------------
C   1    CTEQ5M1  Standard NLO MSbar scheme         0.118     326   226
C   3    CTEQ5L   Leading Order                     0.127     192   146
C ---------------------------------------------------------------------------
C   Note the Qcd-lambda values given for CTEQ5L is for the leading order
C     form of Alpha_s!!  Alpha_s(Mz) gives the absolute calibration.

C  The two Iset value are adopted to agree with the standard table versions.

C   The following user-callable routines are provided:
C 
C     FUNCTION Ctq5Pd (Iset, Iprtn, X, Q, Irt) 
C         returns the PROBABILITY density for a GIVEN flavor;
C
C     FUNCTION Ctq5df (Iset, Iprtn, X, Q, Irt)
C         returns the MOMENTUM density of a GIVEN valence or sea distribution.
C
C     SUBROUTINE Ctq5Pds(Iset, Pdf, X, Q, Irt)
C         returns an array of MOMENTUM densities for ALL flavors;
C****************************************************************************
C     Added by Tim Stelzer    
c               subroutine pftopdg(x,Q,pdf)
c               subroutine pdfset(parm,val)
c
c
C
C   The arguments of these routines are as follows: 
C
C   Iset is the set number:  1 for CTEQ5M1 or 3 for CTEQ5L  
C
C   Iprtn  is the parton label (6, 5, 4, 3, 2, 1, 0, -1, ......, -6)
C                          for (t, b, c, s, d, u, g, u_bar, ..., t_bar)
C  *** WARNING: We use the parton label 2 as D-quark and 1 as U-quark, 
C               which might be different from your labels.
C
C   X, Q are the usual x, Q; 
C
C   Irt is an error code: 0 if there was no error; 1 or more if (x,q) was 
C   outside the range of validity of the parametrization.
C       
C  Range of validity:
C  
C     The range of (x, Q) covered by this parametrization of the QCD evolved
C     parton distributions is 1E-6 < x < 1 ; 1.1 GeV < Q < 10 TeV.  Of course,
C     the PDF's are constrained by data only in a subset of that region; and 
C     the assumed DGLAP evolution is unlikely to be valid for all of it either.
C
C     The range of (x, Q) used in the CTEQ5 round of global analysis is 
C     approximately 0.01 < x < 0.75 ; and 4 GeV^2 < Q^2 < 400 GeV^2 for 
C     fixed target experiments; 0.0001 < x < 0.3 from HERA data; and   
C     Q^2 up to 40,000 GeV^2 from Tevatron inclusive Jet data.
C
C   DOUBLE PRECISION is used throughout in these routines, but conversion to 
C   SINGLE PRECISION is possible by removing the Implicit Double Precision statements. 
C
C **************************************************************************


C ********************************************************
      FUNCTION CTQ5PD(ISET, IPARTON, X, Q, IRT)
C ********************************************************
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)

c if called at a point (x,q) that is outside the region that was 
c actually parametrized, return a value of 0, and set the error code IRT=1.  
c The user can remove the following IF statement to receive instead an 
c extrapolated value, which may be wildly unphysical.
      if((x .lt. 1.e-6). or. (x .gt. 1.) 
     &	 .or. (q .lt. .99) .or. (q .gt. 10000.)) then
         ctq5pd = 0.d0
         irt = 1
         return
      endif

      irt = 0
      if(iset .eq. 3) then
         ctq5pd = ctq5L(iparton,x,q)
      elseif(iset .eq. 1) then
         ctq5pd = ctq5Mi(iparton,x,q)
      else
         print *,'iset=',iset,' has not been parametrized.' 
	   print '(/A)', 'Use the interpolation-table version instead.'
         stop
      endif

      return
      end

C ********************************************************
      FUNCTION CTQ5DF(ISET, IFL, X, Q, IRT)
C ********************************************************
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)

      CTQ5DF = X * CTQ5PD(ISET, IPARTON, X, Q, IRT)
        
      RETURN
      END

C ********************************************************
      SUBROUTINE CTQ5PDS(ISET, PDF, X, Q, IRT)
C ********************************************************
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DIMENSION PDF (-6:6)

      IRT = 0

      DO IFL= -6,2
         PDF(IFL) = CTQ5PD(ISET,IFL,X,Q,IRT1)
         IRT = IRT + IRT1

         IF (IFL .LE. -3) THEN
            PDF(-IFL) = PDF(IFL)
         ENDIF

      ENDDO

      RETURN
      END

c --------------------------------------------------------------------------
	double precision function ctq5MI(ifl,x,q)
c Parametrization of cteq5MI parton distribution functions (J. Pumplin 9/99).
c ifl: 1=u,2=d,3=s,4=c,5=b;0=gluon;-1=ubar,-2=dbar,-3=sbar,-4=cbar,-5=bbar.
c --------------------------------------------------------------------------
	implicit double precision (a-h,o-z)
	integer ifl

	ii = ifl
	if(ii .gt. 2) then
	   ii = -ii
	endif

	if(ii .eq. -1) then
	   sum = faux5MI(-1,x,q)
	   ratio = faux5MI(-2,x,q)
	   ctq5MI = sum/(1.d0 + ratio)

	elseif(ii .eq. -2) then
	   sum = faux5MI(-1,x,q)
	   ratio = faux5MI(-2,x,q)
	   ctq5MI = sum*ratio/(1.d0 + ratio)

	elseif(ii .ge. -5) then
	   ctq5MI = faux5MI(ii,x,q)

	else
	   ctq5MI = 0.d0 

	endif

	return
	end

c ---------------------------------------------------------------------
      double precision function faux5MI(ifl,x,q)
c auxiliary function for parametrization of CTEQ5MI (J. Pumplin 9/99).
c ---------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer ifl

      parameter (nex=8, nlf=2)
      dimension am(0:nex,0:nlf,-5:2)
      dimension alfvec(-5:2), qmavec(-5:2)
      dimension mexvec(-5:2), mlfvec(-5:2)
      dimension ut1vec(-5:2), ut2vec(-5:2)
      dimension af(0:nex)

      data mexvec( 2) / 8 /
      data mlfvec( 2) / 2 /
      data ut1vec( 2) /  0.5141718E+01 /
      data ut2vec( 2) / -0.1346944E+01 /
      data alfvec( 2) /  0.5260555E+00 /
      data qmavec( 2) /  0.0000000E+00 /
      data (am( 0,k, 2),k=0, 2)
     & /  0.4289071E+01, -0.2536870E+01, -0.1259948E+01 /
      data (am( 1,k, 2),k=0, 2)
     & /  0.9839410E+00,  0.4168426E-01, -0.5018952E-01 /
      data (am( 2,k, 2),k=0, 2)
     & / -0.1651961E+02,  0.9246261E+01,  0.5996400E+01 /
      data (am( 3,k, 2),k=0, 2)
     & / -0.2077936E+02,  0.9786469E+01,  0.7656465E+01 /
      data (am( 4,k, 2),k=0, 2)
     & /  0.3054926E+02,  0.1889536E+01,  0.1380541E+01 /
      data (am( 5,k, 2),k=0, 2)
     & /  0.3084695E+02, -0.1212303E+02, -0.1053551E+02 /
      data (am( 6,k, 2),k=0, 2)
     & / -0.1426778E+02,  0.6239537E+01,  0.5254819E+01 /
      data (am( 7,k, 2),k=0, 2)
     & / -0.1909811E+02,  0.3695678E+01,  0.5495729E+01 /
      data (am( 8,k, 2),k=0, 2)
     & /  0.1889751E-01,  0.5027193E-02,  0.6624896E-03 /

      data mexvec( 1) / 8 /
      data mlfvec( 1) / 2 /
      data ut1vec( 1) /  0.4138426E+01 /
      data ut2vec( 1) / -0.3221374E+01 /
      data alfvec( 1) /  0.4960962E+00 /
      data qmavec( 1) /  0.0000000E+00 /
      data (am( 0,k, 1),k=0, 2)
     & /  0.1332497E+01, -0.3703718E+00,  0.1288638E+00 /
      data (am( 1,k, 1),k=0, 2)
     & /  0.7544687E+00,  0.3255075E-01, -0.4706680E-01 /
      data (am( 2,k, 1),k=0, 2)
     & / -0.7638814E+00,  0.5008313E+00, -0.9237374E-01 /
      data (am( 3,k, 1),k=0, 2)
     & / -0.3689889E+00, -0.1055098E+01, -0.4645065E+00 /
      data (am( 4,k, 1),k=0, 2)
     & /  0.3991610E+02,  0.1979881E+01,  0.1775814E+01 /
      data (am( 5,k, 1),k=0, 2)
     & /  0.6201080E+01,  0.2046288E+01,  0.3804571E+00 /
      data (am( 6,k, 1),k=0, 2)
     & / -0.8027900E+00, -0.7011688E+00, -0.8049612E+00 /
      data (am( 7,k, 1),k=0, 2)
     & / -0.8631305E+01, -0.3981200E+01,  0.6970153E+00 /
      data (am( 8,k, 1),k=0, 2)
     & /  0.2371230E-01,  0.5372683E-02,  0.1118701E-02 /

      data mexvec( 0) / 8 /
      data mlfvec( 0) / 2 /
      data ut1vec( 0) / -0.1026789E+01 /
      data ut2vec( 0) / -0.9051707E+01 /
      data alfvec( 0) /  0.9462977E+00 /
      data qmavec( 0) /  0.0000000E+00 /
      data (am( 0,k, 0),k=0, 2)
     & /  0.1191990E+03, -0.8548739E+00, -0.1963040E+01 /
      data (am( 1,k, 0),k=0, 2)
     & / -0.9449972E+02,  0.1074771E+01,  0.2056055E+01 /
      data (am( 2,k, 0),k=0, 2)
     & /  0.3701064E+01, -0.1167947E-02,  0.1933573E+00 /
      data (am( 3,k, 0),k=0, 2)
     & /  0.1171345E+03, -0.1064540E+01, -0.1875312E+01 /
      data (am( 4,k, 0),k=0, 2)
     & / -0.1014453E+03, -0.5707427E+00,  0.4511242E-01 /
      data (am( 5,k, 0),k=0, 2)
     & /  0.6365168E+01,  0.1275354E+01, -0.4964081E+00 /
      data (am( 6,k, 0),k=0, 2)
     & / -0.3370693E+01, -0.1122020E+01,  0.5947751E-01 /
      data (am( 7,k, 0),k=0, 2)
     & / -0.5327270E+01, -0.9293556E+00,  0.6629940E+00 /
      data (am( 8,k, 0),k=0, 2)
     & /  0.2437513E-01,  0.1600939E-02,  0.6855336E-03 /

      data mexvec(-1) / 8 /
      data mlfvec(-1) / 2 /
      data ut1vec(-1) /  0.5243571E+01 /
      data ut2vec(-1) / -0.2870513E+01 /
      data alfvec(-1) /  0.6701448E+00 /
      data qmavec(-1) /  0.0000000E+00 /
      data (am( 0,k,-1),k=0, 2)
     & /  0.2428863E+02,  0.1907035E+01, -0.4606457E+00 /
      data (am( 1,k,-1),k=0, 2)
     & /  0.2006810E+01, -0.1265915E+00,  0.7153556E-02 /
      data (am( 2,k,-1),k=0, 2)
     & / -0.1884546E+02, -0.2339471E+01,  0.5740679E+01 /
      data (am( 3,k,-1),k=0, 2)
     & / -0.2527892E+02, -0.2044124E+01,  0.1280470E+02 /
      data (am( 4,k,-1),k=0, 2)
     & / -0.1013824E+03, -0.1594199E+01,  0.2216401E+00 /
      data (am( 5,k,-1),k=0, 2)
     & /  0.8070930E+02,  0.1792072E+01, -0.2164364E+02 /
      data (am( 6,k,-1),k=0, 2)
     & / -0.4641050E+02,  0.1977338E+00,  0.1273014E+02 /
      data (am( 7,k,-1),k=0, 2)
     & / -0.3910568E+02,  0.1719632E+01,  0.1086525E+02 /
      data (am( 8,k,-1),k=0, 2)
     & / -0.1185496E+01, -0.1905847E+00, -0.8744118E-03 /

      data mexvec(-2) / 7 /
      data mlfvec(-2) / 2 /
      data ut1vec(-2) /  0.4782210E+01 /
      data ut2vec(-2) / -0.1976856E+02 /
      data alfvec(-2) /  0.7558374E+00 /
      data qmavec(-2) /  0.0000000E+00 /
      data (am( 0,k,-2),k=0, 2)
     & / -0.6216935E+00,  0.2369963E+00, -0.7909949E-02 /
      data (am( 1,k,-2),k=0, 2)
     & /  0.1245440E+01, -0.1031510E+00,  0.4916523E-02 /
      data (am( 2,k,-2),k=0, 2)
     & / -0.7060824E+01, -0.3875283E-01,  0.1784981E+00 /
      data (am( 3,k,-2),k=0, 2)
     & / -0.7430595E+01,  0.1964572E+00, -0.1284999E+00 /
      data (am( 4,k,-2),k=0, 2)
     & / -0.6897810E+01,  0.2620543E+01,  0.8012553E-02 /
      data (am( 5,k,-2),k=0, 2)
     & /  0.1507713E+02,  0.2340307E-01,  0.2482535E+01 /
      data (am( 6,k,-2),k=0, 2)
     & / -0.1815341E+01, -0.1538698E+01, -0.2014208E+01 /
      data (am( 7,k,-2),k=0, 2)
     & / -0.2571932E+02,  0.2903941E+00, -0.2848206E+01 /

      data mexvec(-3) / 7 /
      data mlfvec(-3) / 2 /
      data ut1vec(-3) /  0.4518239E+01 /
      data ut2vec(-3) / -0.2690590E+01 /
      data alfvec(-3) /  0.6124079E+00 /
      data qmavec(-3) /  0.0000000E+00 /
      data (am( 0,k,-3),k=0, 2)
     & / -0.2734458E+01, -0.7245673E+00, -0.6351374E+00 /
      data (am( 1,k,-3),k=0, 2)
     & /  0.2927174E+01,  0.4822709E+00, -0.1088787E-01 /
      data (am( 2,k,-3),k=0, 2)
     & / -0.1771017E+02, -0.1416635E+01,  0.8467622E+01 /
      data (am( 3,k,-3),k=0, 2)
     & / -0.4972782E+02, -0.3348547E+01,  0.1767061E+02 /
      data (am( 4,k,-3),k=0, 2)
     & / -0.7102770E+01, -0.3205337E+01,  0.4101704E+00 /
      data (am( 5,k,-3),k=0, 2)
     & /  0.7169698E+02, -0.2205985E+01, -0.2463931E+02 /
      data (am( 6,k,-3),k=0, 2)
     & / -0.4090347E+02,  0.2103486E+01,  0.1416507E+02 /
      data (am( 7,k,-3),k=0, 2)
     & / -0.2952639E+02,  0.5376136E+01,  0.7825585E+01 /

      data mexvec(-4) / 7 /
      data mlfvec(-4) / 2 /
      data ut1vec(-4) /  0.2783230E+01 /
      data ut2vec(-4) / -0.1746328E+01 /
      data alfvec(-4) /  0.1115653E+01 /
      data qmavec(-4) /  0.1300000E+01 /
      data (am( 0,k,-4),k=0, 2)
     & / -0.1743872E+01, -0.1128921E+01, -0.2841969E+00 /
      data (am( 1,k,-4),k=0, 2)
     & /  0.3345755E+01,  0.3187765E+00,  0.1378124E+00 /
      data (am( 2,k,-4),k=0, 2)
     & / -0.2037615E+02,  0.4121687E+01,  0.2236520E+00 /
      data (am( 3,k,-4),k=0, 2)
     & / -0.4703104E+02,  0.5353087E+01, -0.1455347E+01 /
      data (am( 4,k,-4),k=0, 2)
     & / -0.1060230E+02, -0.1551122E+01, -0.1078863E+01 /
      data (am( 5,k,-4),k=0, 2)
     & /  0.5088892E+02, -0.8197304E+01,  0.8083451E+01 /
      data (am( 6,k,-4),k=0, 2)
     & / -0.2819070E+02,  0.4554086E+01, -0.5890995E+01 /
      data (am( 7,k,-4),k=0, 2)
     & / -0.1098238E+02,  0.2590096E+01, -0.8062879E+01 /

      data mexvec(-5) / 6 /
      data mlfvec(-5) / 2 /
      data ut1vec(-5) /  0.1619654E+02 /
      data ut2vec(-5) / -0.3367346E+01 /
      data alfvec(-5) /  0.5109891E-02 /
      data qmavec(-5) /  0.4500000E+01 /
      data (am( 0,k,-5),k=0, 2)
     & / -0.6800138E+01,  0.2493627E+01, -0.1075724E+01 /
      data (am( 1,k,-5),k=0, 2)
     & /  0.3036555E+01,  0.3324733E+00,  0.2008298E+00 /
      data (am( 2,k,-5),k=0, 2)
     & / -0.5203879E+01, -0.8493476E+01, -0.4523208E+01 /
      data (am( 3,k,-5),k=0, 2)
     & / -0.1524239E+01, -0.3411912E+01, -0.1771867E+02 /
      data (am( 4,k,-5),k=0, 2)
     & / -0.1099444E+02,  0.1320930E+01, -0.2353831E+01 /
      data (am( 5,k,-5),k=0, 2)
     & /  0.1699299E+02, -0.3565802E+02,  0.3566872E+02 /
      data (am( 6,k,-5),k=0, 2)
     & / -0.1465793E+02,  0.2703365E+02, -0.2176372E+02 /

      if(q .le. qmavec(ifl)) then
         faux5MI = 0.d0
         return
      endif

      if(x .ge. 1.d0) then
         faux5MI = 0.d0
         return
      endif

      tmp = log(q/alfvec(ifl))
      if(tmp .le. 0.d0) then
         faux5MI = 0.d0
         return
      endif

      sb = log(tmp)
      sb1 = sb - 1.2d0
      sb2 = sb1*sb1

      do i = 0, nex
         af(i) = 0.d0
         sbx = 1.d0
         do k = 0, mlfvec(ifl)
            af(i) = af(i) + sbx*am(i,k,ifl)
            sbx = sb1*sbx
         enddo
      enddo

      y = -log(x)
      u = log(x/0.00001d0)

      part1 = af(1)*y**(1.d0+0.01d0*af(4))*(1.d0+ af(8)*u)
      part2 = af(0)*(1.d0 - x) + af(3)*x 
      part3 = x*(1.d0-x)*(af(5)+af(6)*(1.d0-x)+af(7)*x*(1.d0-x))
      part4 = ut1vec(ifl)*log(1.d0-x) + 
     &	      AF(2)*log(1.d0+exp(ut2vec(ifl))-x)

      faux5MI = exp(log(x) + part1 + part2 + part3 + part4)

c include threshold factor...
      faux5MI = faux5MI * (1.d0 - qmavec(ifl)/q)

      return
      end
c --------------------------------------------------------------------------
	double precision function ctq5L(ifl,x,q)
c Parametrization of cteq5L parton distribution functions (J. Pumplin 9/99).
c ifl: 1=u,2=d,3=s,4=c,5=b;0=gluon;-1=ubar,-2=dbar,-3=sbar,-4=cbar,-5=bbar.
c --------------------------------------------------------------------------
	implicit double precision (a-h,o-z)
	integer ifl

	ii = ifl
	if(ii .gt. 2) then
	   ii = -ii
	endif

	if(ii .eq. -1) then
	   sum = faux5L(-1,x,q)
	   ratio = faux5L(-2,x,q)
	   ctq5L = sum/(1.d0 + ratio)

	elseif(ii .eq. -2) then
	   sum = faux5L(-1,x,q)
	   ratio = faux5L(-2,x,q)
	   ctq5L = sum*ratio/(1.d0 + ratio)

	elseif(ii .ge. -5) then
	   ctq5L = faux5L(ii,x,q)

	else
	   ctq5L = 0.d0 

	endif

	return
	end

c ---------------------------------------------------------------------
      double precision function faux5L(ifl,x,q)
c auxiliary function for parametrization of CTEQ5L (J. Pumplin 9/99).
c ---------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer ifl

      parameter (nex=8, nlf=2)
      dimension am(0:nex,0:nlf,-5:2)
      dimension alfvec(-5:2), qmavec(-5:2)
      dimension mexvec(-5:2), mlfvec(-5:2)
      dimension ut1vec(-5:2), ut2vec(-5:2)
      dimension af(0:nex)

      data mexvec( 2) / 8 /
      data mlfvec( 2) / 2 /
      data ut1vec( 2) /  0.4971265E+01 /
      data ut2vec( 2) / -0.1105128E+01 /
      data alfvec( 2) /  0.2987216E+00 /
      data qmavec( 2) /  0.0000000E+00 /
      data (am( 0,k, 2),k=0, 2)
     & /  0.5292616E+01, -0.2751910E+01, -0.2488990E+01 /
      data (am( 1,k, 2),k=0, 2)
     & /  0.9714424E+00,  0.1011827E-01, -0.1023660E-01 /
      data (am( 2,k, 2),k=0, 2)
     & / -0.1651006E+02,  0.7959721E+01,  0.8810563E+01 /
      data (am( 3,k, 2),k=0, 2)
     & / -0.1643394E+02,  0.5892854E+01,  0.9348874E+01 /
      data (am( 4,k, 2),k=0, 2)
     & /  0.3067422E+02,  0.4235796E+01, -0.5112136E+00 /
      data (am( 5,k, 2),k=0, 2)
     & /  0.2352526E+02, -0.5305168E+01, -0.1169174E+02 /
      data (am( 6,k, 2),k=0, 2)
     & / -0.1095451E+02,  0.3006577E+01,  0.5638136E+01 /
      data (am( 7,k, 2),k=0, 2)
     & / -0.1172251E+02, -0.2183624E+01,  0.4955794E+01 /
      data (am( 8,k, 2),k=0, 2)
     & /  0.1662533E-01,  0.7622870E-02, -0.4895887E-03 /

      data mexvec( 1) / 8 /
      data mlfvec( 1) / 2 /
      data ut1vec( 1) /  0.2612618E+01 /
      data ut2vec( 1) / -0.1258304E+06 /
      data alfvec( 1) /  0.3407552E+00 /
      data qmavec( 1) /  0.0000000E+00 /
      data (am( 0,k, 1),k=0, 2)
     & /  0.9905300E+00, -0.4502235E+00,  0.1624441E+00 /
      data (am( 1,k, 1),k=0, 2)
     & /  0.8867534E+00,  0.1630829E-01, -0.4049085E-01 /
      data (am( 2,k, 1),k=0, 2)
     & /  0.8547974E+00,  0.3336301E+00,  0.1371388E+00 /
      data (am( 3,k, 1),k=0, 2)
     & /  0.2941113E+00, -0.1527905E+01,  0.2331879E+00 /
      data (am( 4,k, 1),k=0, 2)
     & /  0.3384235E+02,  0.3715315E+01,  0.8276930E+00 /
      data (am( 5,k, 1),k=0, 2)
     & /  0.6230115E+01,  0.3134639E+01, -0.1729099E+01 /
      data (am( 6,k, 1),k=0, 2)
     & / -0.1186928E+01, -0.3282460E+00,  0.1052020E+00 /
      data (am( 7,k, 1),k=0, 2)
     & / -0.8545702E+01, -0.6247947E+01,  0.3692561E+01 /
      data (am( 8,k, 1),k=0, 2)
     & /  0.1724598E-01,  0.7120465E-02,  0.4003646E-04 /

      data mexvec( 0) / 8 /
      data mlfvec( 0) / 2 /
      data ut1vec( 0) / -0.4656819E+00 /
      data ut2vec( 0) / -0.2742390E+03 /
      data alfvec( 0) /  0.4491863E+00 /
      data qmavec( 0) /  0.0000000E+00 /
      data (am( 0,k, 0),k=0, 2)
     & /  0.1193572E+03, -0.3886845E+01, -0.1133965E+01 /
      data (am( 1,k, 0),k=0, 2)
     & / -0.9421449E+02,  0.3995885E+01,  0.1607363E+01 /
      data (am( 2,k, 0),k=0, 2)
     & /  0.4206383E+01,  0.2485954E+00,  0.2497468E+00 /
      data (am( 3,k, 0),k=0, 2)
     & /  0.1210557E+03, -0.3015765E+01, -0.1423651E+01 /
      data (am( 4,k, 0),k=0, 2)
     & / -0.1013897E+03, -0.7113478E+00,  0.2621865E+00 /
      data (am( 5,k, 0),k=0, 2)
     & / -0.1312404E+01, -0.9297691E+00, -0.1562531E+00 /
      data (am( 6,k, 0),k=0, 2)
     & /  0.1627137E+01,  0.4954111E+00, -0.6387009E+00 /
      data (am( 7,k, 0),k=0, 2)
     & /  0.1537698E+00, -0.2487878E+00,  0.8305947E+00 /
      data (am( 8,k, 0),k=0, 2)
     & /  0.2496448E-01,  0.2457823E-02,  0.8234276E-03 /

      data mexvec(-1) / 8 /
      data mlfvec(-1) / 2 /
      data ut1vec(-1) /  0.3862583E+01 /
      data ut2vec(-1) / -0.1265969E+01 /
      data alfvec(-1) /  0.2457668E+00 /
      data qmavec(-1) /  0.0000000E+00 /
      data (am( 0,k,-1),k=0, 2)
     & /  0.2647441E+02,  0.1059277E+02, -0.9176654E+00 /
      data (am( 1,k,-1),k=0, 2)
     & /  0.1990636E+01,  0.8558918E-01,  0.4248667E-01 /
      data (am( 2,k,-1),k=0, 2)
     & / -0.1476095E+02, -0.3276255E+02,  0.1558110E+01 /
      data (am( 3,k,-1),k=0, 2)
     & / -0.2966889E+01, -0.3649037E+02,  0.1195914E+01 /
      data (am( 4,k,-1),k=0, 2)
     & / -0.1000519E+03, -0.2464635E+01,  0.1964849E+00 /
      data (am( 5,k,-1),k=0, 2)
     & /  0.3718331E+02,  0.4700389E+02, -0.2772142E+01 /
      data (am( 6,k,-1),k=0, 2)
     & / -0.1872722E+02, -0.2291189E+02,  0.1089052E+01 /
      data (am( 7,k,-1),k=0, 2)
     & / -0.1628146E+02, -0.1823993E+02,  0.2537369E+01 /
      data (am( 8,k,-1),k=0, 2)
     & / -0.1156300E+01, -0.1280495E+00,  0.5153245E-01 /

      data mexvec(-2) / 7 /
      data mlfvec(-2) / 2 /
      data ut1vec(-2) /  0.1895615E+00 /
      data ut2vec(-2) / -0.3069097E+01 /
      data alfvec(-2) /  0.5293999E+00 /
      data qmavec(-2) /  0.0000000E+00 /
      data (am( 0,k,-2),k=0, 2)
     & / -0.6556775E+00,  0.2490190E+00,  0.3966485E-01 /
      data (am( 1,k,-2),k=0, 2)
     & /  0.1305102E+01, -0.1188925E+00, -0.4600870E-02 /
      data (am( 2,k,-2),k=0, 2)
     & / -0.2371436E+01,  0.3566814E+00, -0.2834683E+00 /
      data (am( 3,k,-2),k=0, 2)
     & / -0.6152826E+01,  0.8339877E+00, -0.7233230E+00 /
      data (am( 4,k,-2),k=0, 2)
     & / -0.8346558E+01,  0.2892168E+01,  0.2137099E+00 /
      data (am( 5,k,-2),k=0, 2)
     & /  0.1279530E+02,  0.1021114E+00,  0.5787439E+00 /
      data (am( 6,k,-2),k=0, 2)
     & /  0.5858816E+00, -0.1940375E+01, -0.4029269E+00 /
      data (am( 7,k,-2),k=0, 2)
     & / -0.2795725E+02, -0.5263392E+00,  0.1290229E+01 /

      data mexvec(-3) / 7 /
      data mlfvec(-3) / 2 /
      data ut1vec(-3) /  0.3753257E+01 /
      data ut2vec(-3) / -0.1113085E+01 /
      data alfvec(-3) /  0.3713141E+00 /
      data qmavec(-3) /  0.0000000E+00 /
      data (am( 0,k,-3),k=0, 2)
     & /  0.1580931E+01, -0.2273826E+01, -0.1822245E+01 /
      data (am( 1,k,-3),k=0, 2)
     & /  0.2702644E+01,  0.6763243E+00,  0.7231586E-02 /
      data (am( 2,k,-3),k=0, 2)
     & / -0.1857924E+02,  0.3907500E+01,  0.5850109E+01 /
      data (am( 3,k,-3),k=0, 2)
     & / -0.3044793E+02,  0.2639332E+01,  0.5566644E+01 /
      data (am( 4,k,-3),k=0, 2)
     & / -0.4258011E+01, -0.5429244E+01,  0.4418946E+00 /
      data (am( 5,k,-3),k=0, 2)
     & /  0.3465259E+02, -0.5532604E+01, -0.4904153E+01 /
      data (am( 6,k,-3),k=0, 2)
     & / -0.1658858E+02,  0.2923275E+01,  0.2266286E+01 /
      data (am( 7,k,-3),k=0, 2)
     & / -0.1149263E+02,  0.2877475E+01, -0.7999105E+00 /

      data mexvec(-4) / 7 /
      data mlfvec(-4) / 2 /
      data ut1vec(-4) /  0.4400772E+01 /
      data ut2vec(-4) / -0.1356116E+01 /
      data alfvec(-4) /  0.3712017E-01 /
      data qmavec(-4) /  0.1300000E+01 /
      data (am( 0,k,-4),k=0, 2)
     & / -0.8293661E+00, -0.3982375E+01, -0.6494283E-01 /
      data (am( 1,k,-4),k=0, 2)
     & /  0.2754618E+01,  0.8338636E+00, -0.6885160E-01 /
      data (am( 2,k,-4),k=0, 2)
     & / -0.1657987E+02,  0.1439143E+02, -0.6887240E+00 /
      data (am( 3,k,-4),k=0, 2)
     & / -0.2800703E+02,  0.1535966E+02, -0.7377693E+00 /
      data (am( 4,k,-4),k=0, 2)
     & / -0.6460216E+01, -0.4783019E+01,  0.4913297E+00 /
      data (am( 5,k,-4),k=0, 2)
     & /  0.3141830E+02, -0.3178031E+02,  0.7136013E+01 /
      data (am( 6,k,-4),k=0, 2)
     & / -0.1802509E+02,  0.1862163E+02, -0.4632843E+01 /
      data (am( 7,k,-4),k=0, 2)
     & / -0.1240412E+02,  0.2565386E+02, -0.1066570E+02 /

      data mexvec(-5) / 6 /
      data mlfvec(-5) / 2 /
      data ut1vec(-5) /  0.5562568E+01 /
      data ut2vec(-5) / -0.1801317E+01 /
      data alfvec(-5) /  0.4952010E-02 /
      data qmavec(-5) /  0.4500000E+01 /
      data (am( 0,k,-5),k=0, 2)
     & / -0.6031237E+01,  0.1992727E+01, -0.1076331E+01 /
      data (am( 1,k,-5),k=0, 2)
     & /  0.2933912E+01,  0.5839674E+00,  0.7509435E-01 /
      data (am( 2,k,-5),k=0, 2)
     & / -0.8284919E+01,  0.1488593E+01, -0.8251678E+00 /
      data (am( 3,k,-5),k=0, 2)
     & / -0.1925986E+02,  0.2805753E+01, -0.3015446E+01 /
      data (am( 4,k,-5),k=0, 2)
     & / -0.9480483E+01, -0.9767837E+00, -0.1165544E+01 /
      data (am( 5,k,-5),k=0, 2)
     & /  0.2193195E+02, -0.1788518E+02,  0.9460908E+01 /
      data (am( 6,k,-5),k=0, 2)
     & / -0.1327377E+02,  0.1201754E+02, -0.6277844E+01 /

      if(q .le. qmavec(ifl)) then
         faux5L = 0.d0
         return
      endif

      if(x .ge. 1.d0) then
         faux5L = 0.d0
         return
      endif

      tmp = log(q/alfvec(ifl))
      if(tmp .le. 0.d0) then
         faux5L = 0.d0
         return
      endif

      sb = log(tmp)
      sb1 = sb - 1.2d0
      sb2 = sb1*sb1

      do i = 0, nex
         af(i) = 0.d0
         sbx = 1.d0
         do k = 0, mlfvec(ifl)
            af(i) = af(i) + sbx*am(i,k,ifl)
            sbx = sb1*sbx
         enddo
      enddo

      y = -log(x)
      u = log(x/0.00001d0)

      part1 = af(1)*y**(1.d0+0.01d0*af(4))*(1.d0+ af(8)*u)
      part2 = af(0)*(1.d0 - x) + af(3)*x 
      part3 = x*(1.d0-x)*(af(5)+af(6)*(1.d0-x)+af(7)*x*(1.d0-x))
      part4 = ut1vec(ifl)*log(1.d0-x) + 
     &	      AF(2)*log(1.d0+exp(ut2vec(ifl))-x)

      faux5L = exp(log(x) + part1 + part2 + part3 + part4)

c include threshold factor...
      faux5L = faux5L * (1.d0 - qmavec(ifl)/q)

      return
      end

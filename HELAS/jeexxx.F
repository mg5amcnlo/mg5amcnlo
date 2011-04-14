      subroutine jeexxx(eb,ef,shlf,chlf,phi,nhb,nhf,nsf , jee)
c
c This subroutine computes an off-shell photon wavefunction emitted from
c the electron or positron beam, with a special care for the small angle
c region.  The momenta are measured in the laboratory frame, where the
c e- (e+) beam is along the positive (negative) z axis.
c
c input:
c       real    eb             : energy (gev)    of beam  e-/e+
c       real    ef             : energy (gev)    of final e-/e+
c       real    shlf           : sin(theta/2)    of final e-/e+
c       real    chlf           : cos(theta/2)    of final e-/e+
c       real    phi            : azimuthal angle of final e-/e+
c       integer nhb  = -1 or 1 : helicity        of beam  e-/e+
c       integer nhf  = -1 or 1 : helicity        of final e-/e+
c       integer nsf  = -1 or 1 : +1 for electron, -1 for positron
c
c output:
c       complex jee(6)         : off-shell photon          j^mu(<e|a|e>)
c     
      implicit none
      double complex jee(6),coeff
      double precision cs(2),eb,ef,shlf,chlf,phi,alpha,gal,hi,sf,sfh
      double precision x,me2,q2,rfp,rfm,snp,csp,rxc,c,s
      integer nhb,nhf,nsf

      double precision rZero, rHalf, rOne, rTwo, rFour, rOte
      double precision rPi, rIalph
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rOne = 1.0d0 )
      parameter( rTwo = 2.0d0, rFour = 4.0d0, rOte = 128.9d0 )
      parameter( rPi = 3.14159265358979323846d0 )
      parameter( rIalph = 137.0359895d0 )

      double precision me
      parameter( me = 0.51099906d-3 )

#ifdef HELAS_CHECK
      double precision zero
      double precision epsi
      parameter( epsi = 1.0d-5 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( eb.le.rZero )
     &     write(stdo,*) ' helas-error : eb in jeexxx is not positive'
      if ( ef.le.rZero )
     &     write(stdo,*) ' helas-error : ef in jeexxx is not positive'
      if ( ef.gt.eb ) then
         write(stdo,*) ' helas-error : ef in jeexxx is greater than eb'
         write(stdo,*) '             : eb = ',eb,' : ef = ',ef
      endif
      if ( shlf.lt.rZero .or. shlf.gt.rOne ) then
         write(stdo,*) ' helas-error : shlf in jeexxx is improper'
         write(stdo,*) '               shlf = ',shlf
      endif
      if ( chlf.lt.rZero .or. chlf.gt.rOne ) then
         write(stdo,*) ' helas-error : chlf in jeexxx is improper'
         write(stdo,*) '               chlf = ',chlf
      endif
      zero = abs(shlf**2+chlf**2-rOne)
      if ( zero.gt.epsi ) then
         write(stdo,*)
     &        ' helas-error : shlf and chlf in jeexxx are inconsistent'
         write(stdo,*) '               shlf,chlf = ',shlf,chlf
      endif
      if ( phi.lt.rZero .or. phi.gt.rTwo*rPi ) then
         write(stdo,*)
     &   ' helas-warn  : phi in jeexxx does not lie on 0.0 thru 2.0*pi'
         write(stdo,*) 
     &   '             : phi = ',phi
      endif
      if ( abs(nhb).ne.1 ) then
         write(stdo,*) ' helas-error : nhb in jeexxx is not -1,1'
         write(stdo,*) '             : nhb = ',nhb
      endif
      if ( abs(nhf).ne.1 ) then
         write(stdo,*) ' helas-error : nhf in jeexxx is not -1,1'
         write(stdo,*) '             : nhf = ',nhf
      endif
      if ( abs(nsf).ne.1 ) then
         write(stdo,*) ' helas-error : nsf in jeexxx is not -1,1'
         write(stdo,*) '             : nsf = ',nsf
      endif
      if ( eb.lt.rOne ) then
         write(stdo,*)
     &   ' helas-warn  : use of jeexxx is not appropriate: eb too low'
         write(stdo,*)
     &   '             : eb = ',eb
      endif
#endif

      alpha = rOne/rOte
      gal = sqrt(alpha*rFour*rPi)

      hi = nhb
      sf = nsf
      sfh = nhb*nsf
      cs((3+nsf)/2) = shlf
      cs((3-nsf)/2) = chlf
c cs(1)=chlf and cs(2)=shlf for electron
c cs(1)=shlf and cs(2)=chlf for positron
      x = ef/eb
      me2 = me**2
      q2 = - rFour*cs(2)**2*(ef*eb-me2)
     &     + sf*(rOne-x)**2/x*(shlf+chlf)*(shlf-chlf)*me2
      rfp = (1+nsf)
      rfm = (1-nsf)
      snp = sin(phi)
      csp = cos(phi)

      if ( nhb.eq.nhf ) then
         rxc = rTwo*x/(rOne-x)*cs(1)**2
         coeff = gal*rTwo*eb*sqrt(x)*cs(2)/q2
     &          *(dcmplx( rfp )-rfm*dcmplx( csp, -snp*hi ))*rHalf
         jee(1) = dcmplx( rZero )
         jee(2) = coeff*dcmplx( (rOne+rxc)*csp, -sfh*snp )
         jee(3) = coeff*dcmplx( (rOne+rxc)*snp,  sfh*csp )
         jee(4) = coeff*(-sf*rxc/cs(1)*cs(2))
      else
         coeff = gal*me/q2/sqrt(x)
     &          *(dcmplx( rfp )+rfm*dcmplx( csp, snp*hi ))*rHalf*hi
         jee(1) = -coeff*(rOne+x)*cs(2)*dcmplx( csp , sfh*snp )
         jee(2) =  coeff*(rOne-x)*cs(1)
         jee(3) =  jee(2)*dcmplx( rZero, sfh )
         jee(4) =  jee(1)*sf*(rOne-x)/(rOne+x)
      endif

      c = (chlf+shlf)*(chlf-shlf)
      s = rTwo*chlf*shlf

      jee(5) = -eb*dcmplx( rOne-x, sf-x*c )
      jee(6) =  eb*x*s*dcmplx( csp, snp )
c
      return
      end

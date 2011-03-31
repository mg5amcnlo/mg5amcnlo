      subroutine eaoxxx(eb,ea,shlf,chlf,phi,nhe,nha , eao)
c
c This subroutine computes an off-shell positron wavefunction after
c emitting a photon from the positron beam, with a special care for the
c small angle region.  The momenta are measured in the laboratory frame,
c where the e+ beam is along the negative z axis.
c
c input:
c       real    eb             : energy (GeV)    of beam  e+
c       real    ea             : energy (GeV)    of final photon
c       real    shlf           : sin(theta/2)    of final photon
c       real    chlf           : cos(theta/2)    of final photon
c       real    phi            : azimuthal angle of final photon
c       integer nhe  = -1 or 1 : helicity        of beam  e+
c       integer nha  = -1 or 1 : helicity        of final photon
c
c output:
c       complex eao(6)         : off-shell positron             <e,A,e'|
c
      implicit none
      double complex eao(6),phs
      double precision eb,ea,shlf,chlf,phi,alpha,gal,rnhe,x,c,s,d
      double precision coeff,xnnp,xnnm,snp,csp
      integer nhe,nha,nn

      double precision rHalf, rOne, rTwo, rFour, rOte
      double precision rPi, rIalph
      parameter( rHalf = 0.5d0, rOne = 1.0d0, rTwo = 2.0d0 )
      parameter( rFour = 4.0d0, rOte = 128.9d0 )
      parameter( rPi = 3.14159265358979323846d0 )
      parameter( rIalph = 137.0359895d0 )

      double precision me
      parameter( me = 0.510998902d-3 )

#ifdef HELAS_CHECK
      double precision zero
      double precision rZero
      parameter( rZero = 0.0d0 )
      double precision epsi
      parameter( epsi = 1.0d-5 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( eb.le.rZero )
     &     write(stdo,*) ' helas-error : eb in eaoxxx is not positive'
      if ( ea.le.rZero )
     &     write(stdo,*) ' helas-error : ea in eaoxxx is not positive'
      if (ea.gt.eb ) then
         write(stdo,*) ' helas-error : ea in eaoxxx is greater than eb'
         write(stdo,*) '               eb = ',eb,' : ea = ',ea
      endif
      if ( shlf.lt.rZero .or. shlf.gt.rOne ) then
         write(stdo,*) ' helas-error : shlf in eaoxxx is improper'
         write(stdo,*) '               shlf = ',shlf
      endif
      if ( chlf.lt.rZero .or. chlf.gt.rOne ) then
         write(stdo,*) ' helas-error : chlf in eaoxxx is improper'
         write(stdo,*) '               chlf = ',chlf
      endif
      zero = abs(shlf**2+chlf**2-rOne)
      if ( zero.gt.epsi ) then
         write(stdo,*)
     &        ' helas-error : shlf and chlf in eaoxxx are inconsistent'
         write(stdo,*)
     &        '             : shlf,chlf = ',shlf,chlf
      endif
      if ( phi.lt.rZero .or. phi.gt.rTwo*rPi ) then
         write(stdo,*)
     &   ' helas-warn  : phi in eaoxxx does not lie on 0.0 thru 2.0*pi'
         write(stdo,*)
     &   '             : phi = ',phi
      endif
      if (abs(nhe).ne.1 ) then
         write(stdo,*) ' helas-error : nhe in eaoxxx is not -1,1'
         write(stdo,*) '               nhe = ',nhe
      endif
      if ( abs(nha).ne.1 ) then
         write(stdo,*) ' helas-error : nha in eaoxxx is not -1,1'
         write(stdo,*) '               nha = ',nha
      endif
      if ( eb.lt.rOne ) then
         write(stdo,*)
     &   ' helas-warn  : use of eaoxxx is not appropriate: eb too low'
         write(stdo,*)
     &   '             : eb = ',eb
      endif
#endif

      alpha = rOne/rOte
      gal = sqrt(alpha*rFour*rPi)

      nn = nha*nhe
      rnhe = nhe
      x = ea/eb
      c = (chlf+shlf)*(chlf-shlf)
      s = rTwo*chlf*shlf
      d = -rOne/(ea*eb*(rFour*chlf**2-(me/eb)**2*c))
      coeff = nn*gal*sqrt(eb)*d
      xnnp = x*(1+nn)
      xnnm = x*(1-nn)
      snp = sin(phi)
      csp = cos(phi)
      phs = dcmplx( csp, -rnhe*snp )

      eao((5-3*nhe)/2) =               coeff*me*s*(rOne+xnnp*rHalf)
      eao((5-nhe)/2)   = rnhe*xnnp    *coeff*me*shlf**2*phs
      eao((5+nhe)/2)   =               coeff*eb*s*(-rTwo+xnnm)
      eao((5+3*nhe)/2) = real(nha-nhe)*coeff*eb*x*chlf**2*phs*rTwo

      eao(5) = eb*dcmplx( x-rOne, x*c+rOne )
      eao(6) = eb*x*s*dcmplx( csp, snp )
c
      return
      end

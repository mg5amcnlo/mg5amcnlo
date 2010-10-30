      subroutine eaixxx(eb,ea,shlf,chlf,phi,nhe,nha , eai)
c
c This subroutine computes an off-shell electron wavefunction after
c emitting a photon from the electron beam, with a special care for the
c small angle region.  The momenta are measured in the laboratory frame,
c where the e- beam is along the positive z axis.
c
c input:
c       real    eb             : energy (GeV)    of beam  e-
c       real    ea             : energy (GeV)    of final photon
c       real    shlf           : sin(theta/2)    of final photon
c       real    chlf           : cos(theta/2)    of final photon
c       real    phi            : azimuthal angle of final photon
c       integer nhe  = -1 or 1 : helicity        of beam  e-
c       integer nha  = -1 or 1 : helicity        of final photon
c
c output:
c       complex eai(6)         : off-shell electron             |e',A,e>
c
      implicit none
      double complex eai(6),phs
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
      if ( eb.le.rZero ) then
         write(stdo,*) ' helas-error : eb in eaixxx is not positive'
      endif
      if ( ea.le.rZero ) then
         write(stdo,*) ' helas-error : ea in eaixxx is not positive'
      endif
      if ( ea.gt.eb ) then
         write(stdo,*) ' helas-error : ea in eaixxx is greater than eb'
         write(stdo,*) '             : eb = ',eb,' : ea = ',ea
      endif
      if ( shlf.lt.rZero .or. shlf.gt.rOne ) then
         write(stdo,*) ' helas-error : shlf in eaixxx is improper'
         write(stdo,*) '             : shlf = ',shlf
      endif
      if ( chlf.lt.rZero .or. chlf.gt.rOne ) then
         write(stdo,*) ' helas-error : chlf in eaixxx is improper'
         write(stdo,*) '             : chlf = ',chlf
      endif
      zero = abs(shlf**2+chlf**2-rOne)
      if ( zero.gt.epsi ) then
         write(stdo,*)
     &        ' helas-error : shlf and chlf in eaixxx are inconsistent'
         write(stdo,*)
     &        '             : shlf,chlf = ',shlf,chlf
      endif
      if ( phi.lt.rZero .or. phi.gt.rTwo*rPi ) then
         write(stdo,*)
     &   ' helas-warn  : phi in eaixxx does not lie on 0.0 thru 2.0*pi'
         write(stdo,*)
     &   '             : phi = ',phi
      endif
      if ( abs(nhe).ne.1 ) then
         write(stdo,*) ' helas-error : nhe in eaixxx is not -1,1'
         write(stdo,*) '               nhe = ',nhe
      endif
      if ( abs(nha).ne.1 ) then
         write(stdo,*) ' helas-error : nha in eaixxx is not -1,1'
         write(stdo,*) '               nha = ',nha
      endif
      if ( eb.lt.rOne ) then
         write(stdo,*)
     &   ' helas-warn  : use of eaixxx is not appropriate: eb too low'
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
      d = -rOne/(ea*eb*(rFour*shlf**2+(me/eb)**2*c))
      coeff = -nn*gal*sqrt(eb)*d
      xnnp = x*(1+nn)
      xnnm = x*(1-nn)
      snp = sin(phi)
      csp = cos(phi)
      phs = dcmplx( csp, rnhe*snp )

      eai((5-3*nhe)/2) = -rnhe*coeff*me*s*(rOne+xnnp*rHalf)
      eai((5-nhe)/2)   =  xnnp*coeff*me*chlf**2*phs
      eai((5+nhe)/2)   =  rnhe*coeff*eb*s*(-rTwo+xnnm)
      eai((5+3*nhe)/2) =  xnnm*coeff*eb*shlf**2*phs*rTwo

      eai(5) =  eb*dcmplx( rOne-x, rOne-x*c )
      eai(6) = -eb*x*s*dcmplx( csp, snp )
c
      return
      end

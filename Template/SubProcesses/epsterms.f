      Subroutine epsii(id1,id2,sik,epssq,eps)
C  Calculates the 1/eps^2 and 1/eps term 
C  for initial state emitter and initial state spectator
      implicit none

      include 'coupl.inc'
      include 'dipole.inc'
C  Arguments
      integer id1,id2
      real*8 sik,epssq,eps

c  Local
      REAL*8 cf,ca,pi,L,gsq
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926d0)
      INTEGER i

      gsq=GG(1)**2
      L=dlog(mu**2/sik)
      epssq=0d0
      eps=0d0

C  quark-quark splitting (i:quark, ij~: quark)
      if(id1.eq.1 .and. id2.eq.1) then
        epssq=gsq/(8d0*pi**2)
        eps=gsq*(3d0 + 2d0*L)/(16.*pi**2) 

C  quark-gluon splitting (i:quark, ij~: gluon)
      elseif(id1.eq.1.and.id2.eq.0) then
       return

C  gluon-quark splitting (i:gluon, ij~: quark)
      elseif(id1.eq.0.and.id2.eq.1) then
       return

C  gluon-gluon splitting (i:gluon, ij~: gluon)
      elseif(id1.eq.0.and.id2.eq.0) then
        epssq=gsq/(8d0*pi**2)
        eps=(gsq*((11 + 6*L) - 2*Nf/ca))/(48.*pi**2)

      endif
      end


      Subroutine epsif(id1,id2,sik,mk,epssq,eps)
C  Calculates the 1/eps^2 and 1/eps term 
C  for initial state emitter and final state spectator
      implicit none

      include 'coupl.inc'
      include 'dipole.inc'
C  Arguments
      integer id1,id2
      real*8 sik,mk,epssq,eps

c  Local
      REAL*8 cf,ca,pi,L,gsq,musq_k
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926d0)
      INTEGER i

      gsq=GG(1)**2
      L=dlog(mu**2/sik)
      musq_k=mk**2/sik
      epssq=0d0
      eps=0d0

C  quark-quark splitting (i:quark, ij~: quark)
      if(id1.eq.1.and.id2.eq.1) then
        epssq=gsq/(8d0*pi**2)
        eps=(gsq*(3+2*L+2*DLog(1+musq_k)))/(16.*pi**2)

C  quark-gluon splitting (i:quark, ij~: gluon)
      elseif(id1.eq.1.and.id2.eq.0) then
       return

C  gluon-quark splitting (i:gluon, ij~: quark)
      elseif(id1.eq.0.and.id2.eq.1) then
       return

C  gluon-gluon splitting (i:gluon, ij~: gluon)
      elseif(id1.eq.0.and.id2.eq.0) then
        epssq=gsq/(8d0*pi**2)
        eps=(gsq*((11+6*L)-2*Nf/ca+6*DLog(1+musq_k)))/(48.*pi**2)

      endif
      end


      Subroutine epsfi(id1,id2,sik,mi,epssq,eps)
C  Calculates the 1/eps^2 and 1/eps term 
C  for final state emitter and initia; state spectator
      implicit none

      include 'coupl.inc'
      include 'dipole.inc'
C  Arguments
      integer id1,id2
      real*8 sik,mi,epssq,eps

c  Local
      REAL*8 cf,ca,pi,L,gsq,musq_i
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926d0)
      INTEGER i

      gsq=GG(1)**2
      L=dlog(mu**2/sik)
      musq_i=mi**2/sik
      epssq=0d0
      eps=0d0

C  massive quark (ij~:quark)
      if(id1.eq.1.and.mi.gt.0d0) then
         eps=gsq*(1+DLog(musq_i/(1+musq_i)))/(8.*pi**2)

C  massless quark( ij~:quark)
      elseif(id1.eq.1.and.mi.eq.0d0) then
         epssq= gsq/(8d0*pi**2)
         eps=gsq*(3 + 2*L)/(16.*pi**2)

C  gluon splitting (ij~: gluon)
C  1. g-> QQ (i:massive quark )
      elseif(id1.eq.0.and. mi.gt.0d0 ) then
         return
c  2. g-> qq (i:massless quark)
      elseif(id1.eq.0.and.id2.eq.1.and.mi.eq.0d0) then
         eps=-gsq/(24.*pi**2)/ca

C  3. g-> gg (i: gluon )
      elseif(id1.eq.0) then
         epssq=gsq/(4d0*pi**2)
         eps=gsq*(11 + 6*L)/(24.*pi**2)

      endif
      end


      Subroutine epsff(mi,mk,sik,id,id1,epssq,eps)
C  Calculates the 1/eps^2 and 1/eps term 
C  for final state emitter and initia; state spectator
      implicit none

      include 'coupl.inc'
      include 'dipole.inc'
c Arguments
      REAL*8 mi,mk,sik,epssq,eps
      INTEGER id,id1
c Global
      REAL*8 ddilog,lambda_tr
      external ddilog,lambda_tr
c  Local
      REAL*8 cf,ca,pi,L,gsq,vik,rho,musq_i,musq_k,Qik,Qik2
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926d0)
      INTEGER i

c For the massive case, sik is the Qik^2 from CDST and vice versa.

      musq_i=mi**2/sik
      musq_k=mk**2/sik
      Qik2=sik-mi**2-mk**2
      Qik=Sqrt(Qik2)
      gsq=GG(1)**2
      L=DLog(mu**2/sik)
      vik=Sqrt(lambda_tr(1d0,musq_i,musq_k))/(1d0-musq_i-musq_k)
      rho=Sqrt((1d0-vik)/(1d0+vik))

      epssq=0d0
      eps=0d0

C  ij~ massive and massive spectator
      if(mi.gt.0d0.and.mk.gt.0d0) then
        eps=(gsq*(1d0 + DLog(rho)/vik))/(8.*pi**2)

C  ij~ massive and massless spectator
      elseif(mi.gt.0d0.and.mk.eq.0d0) then
        epssq=gsq/(16d0*pi**2)
        eps=(gsq*(2 + L + DLog(musq_i/(1 - musq_i)**2)))/(16d0*Pi**2)

C  ij~ massless quark and massive spectator
      elseif(id.eq.1.and.mi.eq.0d0.and.mk.gt.0d0) then
        epssq=gsq/(16d0*pi**2)
        eps=(gsq*(3 + L + DLog(musq_k/(1 - musq_k)**2)))/(16d0*pi**2)

C  ij~ massless quark and massless spectator
      elseif(id.eq.1.and.mi.eq.0d0.and.mk.eq.0d0) then
        epssq=gsq/(8d0*pi**2)
        eps=(gsq*(3 + 2*L))/(16.*pi**2)

C  ij~ gluon and massive spectator
      elseif(id.eq.0.and.mk.gt.0d0) then
C  1. g->QQ splitting
         if(id1.eq.1.and.mi.gt.0d0) then
            return

C  2. g->qq aplitting
         elseif(id1.eq.1.and.mi.eq.0d0) then
            eps=-gsq/(24.*pi**2)/ca

C  3. g->gg splitting
         elseif(id1.eq.0) then
            epssq=gsq/(8d0*pi**2)
            eps=(gsq*(11d0/3d0 + L + DLog(musq_k/(1 - musq_k)**2)))
     &           /(8.*Pi**2)
         endif
      
C  ij~ gluon and massless spectator
      elseif(id.eq.0.and.mk.eq.0d0) then
C  1. g->QQ cplitting
         if(id1.eq.1.and.mi.gt.0d0) then
            return

C  2. g->qq aplitting
         elseif(id.eq.0.and.id1.eq.1.and.mi.eq.0d0) then
            eps=-gsq/(24.*pi**2)/ca

C  3. g->gg splitting
         elseif(id1.eq.0) then
            epssq=gsq/(4d0*pi**2)
            eps=(gsq*(11+6*L))/(24.*pi**2)
         endif
      endif
      end





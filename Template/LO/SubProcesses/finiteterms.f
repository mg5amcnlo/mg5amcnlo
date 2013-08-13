ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c FINITE PART OF THE INTEGRATE DIPOLES
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Returns fi(8):
c fi(1), regular piece at z!=1
c fi(2), delta piece at z=1
c fi(3), plus distribution at z!=1
c fi(4), plus distribution at z=1
c fi(5), mass correction to plus distribution at z!=1
c fi(6), mass correction to plus distribution at z=1
c fi(7), delta piece at z=1-4m^2/s
c fi(8), plus distribution at z!=1-4m^2/s
c fi(9), plus distribution at z=1-4m^2/s
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


      SUBROUTINE finiteff(mi,mk,sik,id,id1,fi)
c  calculates the finite terms when both emitter
c  and spectator are in the final state
      implicit none

      include "coupl.inc"
      include 'dipole.inc'
c Arguments
      REAL*8 mi,mk,sik,fi(9)
      INTEGER id,id1,i
c Global
      REAL*8 ddilog,lambda_tr
      external ddilog,lambda_tr
c Local
      REAl*8 cf,ca,rhoi,rhok,rho,musq_i,musq_k,gsq,pi
      REAL*8 xp,yp,vik,muk,rho1,rho2,L,yl,rs,Qik2,a,b,c,d,xm,x,mui
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926535897932385d0)
      complex*16 test

      do i=1,9
         fi(i)=0d0
      enddo

c For the massive cases, sik is the Qik^2 from CDST and vice versa.

      musq_i=mi**2/sik
      musq_k=mk**2/sik
      muk=Sqrt(musq_k)
      mui=Sqrt(musq_i)
      Qik2=sik-mi**2-mk**2
      gsq=GG(1)**2
      L=DLog(mu**2/sik)

C  ij~ massive and massive spectator
      if(mi.gt.0d0 .and. mk .gt. 0d0.and.id.eq.1) then
         rs=0d0
         vik=Sqrt(lambda_tr(1d0,musq_i,musq_k))/(1d0-musq_i-musq_k)
         rhoi=Sqrt((1d0-vik+2d0*musq_i/(1d0-musq_i-musq_k))/
     &        (1d0+vik+2d0*musq_i/(1d0-musq_i-musq_k)))
         rhok=Sqrt((1d0-vik+2d0*musq_k/(1d0-musq_i-musq_k))/
     &        (1d0+vik+2d0*musq_k/(1d0-musq_i-musq_k)))
         rho=Sqrt((1d0-vik)/(1d0+vik))
         fi(2)=(gsq*(6 + 2*L + 2*rs + 
     &        2*mk*((4*mk - 2*Sqrt(sik))/Qik2 + 1/(mk - Sqrt(sik))) - 
     &        (2*Pi**2)/(3.*vik) - 4*Log(-mi**2 + (mk - Sqrt(sik))**2) + 
     &        2*Log(mi - (mi*mk)/Sqrt(sik)) - 
     &        (4*mi**2*Log(mi/(-mk + Sqrt(sik))))/Qik2 + 3*Log(sik) - 
     &        (-4*Log(rho**2)*Log(1 + rho**2) + Log(rhoi**2)**2 + 
     &        Log(rhok**2)**2 -
     &        4*Log(rho)*(L + 2*Log(sik/Qik2)))/(2.*vik) - 
     &        (2*(-2*ddilog(rho**2) + ddilog(1 - rhoi**2) + 
     &        ddilog(1 - rhok**2)))/vik))/(16.*Pi**2)

C adding alpha dependent terms
         if(alpha_ff.ne.1d0) then
            a=(2*muk)/(1 - musq_i - musq_k)
            b=(2*(1 - muk))/(1 - musq_i - musq_k)
            c=(2*(1 - muk)*muk)/(1 - musq_i - musq_k)
            d=(1 - musq_i - musq_k)/2.
            xp=(-musq_i + (1 - muk)**2 +
     &           Sqrt(lambda_tr(1d0,musq_i,musq_k)))/(1-musq_i-musq_k)
            xm=(-musq_i + (1 - muk)**2 -
     &           Sqrt(lambda_tr(1d0,musq_i,musq_k)))/(1-musq_i-musq_k)
            yp=1 - (2*(1 - muk)*muk)/(1 - musq_i - musq_k)
            x=yp - alpha_ff*yp + Sqrt(((4*musq_i*musq_k)/
     &           ((musq_i - (1 - muk)**2)*(1 - musq_i - musq_k)) +
     &           1/yp - alpha_ff*yp)*(yp - alpha_ff*yp))
            fi(2)=fi(2)+gsq*(1/(1- muk)-(2*(2-2*musq_i-muk))/
     &           (1-musq_i-musq_k)+(3*(1+alpha_ff*yp))/2. + 
     &           (musq_i*(1 - alpha_ff*yp))/(2.*(musq_i +
     &           alpha_ff*(1 - musq_i - musq_k)*yp)) - 
     &           2*DLog((alpha_ff*(1 - musq_i - musq_k)*yp)/
     &           (-musq_i + (1 - muk)**2)) + 
     &           ((1 + musq_i - musq_k)*
     &           DLog((musq_i + alpha_ff*(1 - musq_i - musq_k)*yp)
     &           /(1 - muk)**2))/(2.*(1 - musq_i - musq_k)) + 
     &           (2*(-(DLog(b)*DLog((a*(-b + xm))/((a + b)*xm))) +
     &           DLog(b - x)*DLog(((a + x)*(-b + xm))/
     &           ((a + b)*(-x + xm))) + 
     &           DLog((c + xm)/(a + xm))*DLog((-x + xm)/xm) +
     &           DLog(a*(b - xp))*DLog(xp) + 
     &           (DLog((a + x)/a)*DLog(a*(a + x)*(a + xp)**2))/2. -
     &           DLog(c)*DLog(((a - c)*xp)/(a*(c + xp))) + 
     &           DLog(d)*DLog(((a + x)*xm*xp)/(a*(-x + xm)*(-x + xp))) -
     &           DLog((a + x)*(b - xp))*DLog(-x + xp) + 
     &           DLog(c + x)*DLog(((a-c)*(-x + xp))/((a + x)*(c + xp)))-
     &           ddilog(b/(a + b)) + ddilog(c/(-a + c)) + 
     &           ddilog((b - x)/(a + b)) - ddilog((c + x)/(-a + c)) +
     &           ddilog(b/(b - xm)) - ddilog((b - x)/(b - xm)) -
     &           ddilog(xm/(a + xm)) + ddilog(xm/(c + xm)) +
     &           ddilog((-x + xm)/(a+xm)) - ddilog((-x + xm)/(c + xm))+
     &           ddilog(a/(a + xp)) - ddilog((a + x)/(a + xp)) -
     &           ddilog(xp/(-b + xp)) - ddilog(c/(c + xp)) +
     &           ddilog((c + x)/(c + xp)) +
     &           ddilog((-x + xp)/(-b + xp))))/vik)/(8.*Pi**2)

         endif

C ij~ massive and massless spectator
      elseif(mi.gt.0d0.and.mk.eq.0d0.and.id.eq.1) then
         rs=0d0
         fi(2)= (gsq*(72d0 + 6d0*L*(4d0 + L) - 11d0*Pi**2 + 24d0*rs + 
     &        (24d0*musq_i*DLog(musq_i))/(-1d0 + musq_i) + 
     &        6d0*(4d0*DLog(1 - musq_i)**2 + 
     &        (2d0 + 2d0*L - Log(musq_i))*DLog(musq_i) - 
     &        4d0*DLog(1d0 - musq_i)*(2d0 + L + DLog(musq_i))) - 
     &        24d0*ddilog(1d0 - musq_i)))/(192d0*Pi**2)

C adding alpha dependent terms
         if(alpha_ff.ne.1d0) then
            fi(2)=fi(2)+(gsq*(-2*DLog(alpha_ff) + 
     &           (2*DLog(alpha_ff+(1-alpha_ff)*musq_i))/(1-musq_i) + 
     &           (-2 + 3*alpha_ff-alpha_ff/(alpha_ff+(1-alpha_ff)*musq_i) - 
     &           ((3 - musq_i)*DLog(alpha_ff+(1-alpha_ff)*musq_i))/
     &           (1 - musq_i))/2. +2*(-(DLog(alpha_ff)*DLog(musq_i))- 
     &           ddilog((-1+ musq_i)/musq_i) + 
     &           ddilog((alpha_ff*(-1 + musq_i))/musq_i))))/(8.*pi**2)
         endif

C ij~ massless quark and massive spectator
      elseif(mi.eq.0d0 .and. mk.gt.0d0 .and. id.eq.1) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/2d0
         endif
         fi(2)= (gsq*((36*L*(1 + muk) + 
     &        6*L**2*(1 + muk) - 11*Pi**2 + 
     &        24*(5 + rs) + muk*(-11*Pi**2 + 24*(2 + rs)) - 
     &        36*(1 + muk)*Log((-1 + muk)**2) - 6*(1 + muk)*
     &        (-4*Log(1 - musq_k)**2 + Log(musq_k)*(-2*L + Log(musq_k))+ 
     &        4*Log(1 - musq_k)*(L + Log(musq_k))))/(1 + muk) 
     &        - 24*ddilog(1 - musq_k)))/
     &        (192.*Pi**2)

C adding alpha dependent terms
         if(alpha_ff.ne.1d0) then
            yp=(1d0-musq_k)/(1d0+musq_k)
            xp=yp*(1d0-alpha_ff)+
     &           Sqrt((1d0-alpha_ff)*(1d0-alpha_ff*yp**2))
            fi(2)=fi(2)+(gsq*((-3*((1-alpha_ff)*yp+DLog(alpha_ff)))/2. + 
     &           2*(-DLog((1-xp+yp)/(1+yp))**2+DLog((1+2*xp*yp-yp**2)/
     &           ((1 + xp - yp)*(1 - xp + yp)))**2/2. + 
     &           2*(DLog((1+xp-yp)/(1-yp))*DLog((1+yp)/2.) + 
     &           DLog((1+yp)/(2.*yp))*DLog((1+2*xp*yp-yp**2)/(1-yp**2))- 
     &           ddilog((1-yp)/2.) +ddilog((1 + xp - yp)/2.) + 
     &           ddilog((1 - yp)/(1 + yp)) - 
     &           ddilog((1+2*xp*yp-yp**2)/(1+yp)**2)))
     &           ))/(8.*pi**2)
         endif


C ij~ massless quark and massless spectator
      elseif(mi.eq.0d0 .and. mk.eq.0d0 .and. id.eq.1) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/2d0
         endif
         fi(2)=(gsq*(6*L*(3+L)-7*pi**2+12*(5+rs)))/(96.*pi**2)
         
C     adding alpha dependent terms
         if (alpha_ff.ne.1d0) then
            fi(2)=fi(2)+((gsq*((3*(-1 + alpha_ff - DLog(alpha_ff)))/2.
     &           - DLog(alpha_ff)**2))/(8.*pi**2))
         endif


C ij~ gluon and massive spectator
      elseif(mk.gt.0d0.and. id.eq.0) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/6d0
         endif
C  1. g->QQ splitting
       if(id1.eq.1.and.mi.gt.0d0) then
          fi(2)=(gsq*((2*(1 - (4*musq_i)/(-1 + muk)**2)**1.5*muk)/
     &         (1 + muk) + (8*Sqrt(1 - (4*musq_i)/(-1 + muk)**2)*
     &         (1 + mui - muk)*(-1 + mui + muk))/(3.*(-1 + muk)**2)
     &         - 2*DLog(mui) + 2*DLog((1 +
     &         Sqrt(1 - (4*musq_i)/(-1 + muk)**2))/2.)+2*DLog(1 - muk)- 
     &         (2*musq_k*((8*musq_i*Sqrt(1 - (4*musq_i)/(-1 + muk)**2))/
     &         (-1 + musq_k)-DLog((1-Sqrt(1-(4*musq_i)/(-1 + muk)**2))/
     &         (1 + Sqrt(1 - (4*musq_i)/(-1 + muk)**2))) + 
     &         ((-1 + 4*musq_i + musq_k)/(-1 + musq_k))**1.5*
     &         DLog((-Sqrt(1 - (4*musq_i)/(-1 + muk)**2) +
     &         Sqrt((-1 + 4*musq_i + musq_k)/(-1 + musq_k)))/
     &         (Sqrt(1 - (4*musq_i)/(-1 + muk)**2) +
     &         Sqrt((-1 + 4*musq_i + musq_k)/(-1 + musq_k))))))
     &         /(1 - musq_k)))/(24.*Pi**2)/ca


C adding alpha dependent terms
          if(alpha_ff.ne.1d0) then
           a=1.-musq_k
           c=-1.+2*musq_i+musq_k
           yp=1.-2*muk*(1.-muk)/(1.-2*musq_i-musq_k)
           b=sqrt(c**2*yp**2-4.*musq_i**2)
           d=sqrt(alpha_ff*c**2*yp**2-4.*musq_i**2)
           fi(2)=fi(2)-gsq*((c*Sqrt(-c + 2*musq_i)*(4*(b - d)*musq_i**2 
     &          + c**2*yp*(-2*alpha_ff*b-d*(-2+yp) + alpha_ff**2*b*yp)+ 
     &          4*c*musq_i*(d - d*yp + b*(-1 + alpha_ff*yp))) - 
     &          2*b*d*Sqrt(-c + 2*musq_i)*
     &          (c**2 - 2*(1 + c)*musq_i + 4*musq_i**2)
     &          *DATAN((2*musq_i)/Sqrt(-4*musq_i**2 + c**2*yp**2)) + 
     &          2*b*d*Sqrt(-c + 2*musq_i)*
     &          (c**2 - 2*(1 + c)*musq_i + 4*musq_i**2)
     &          *DATAN((2*musq_i)/
     &          Sqrt(-4*musq_i**2 + alpha_ff**2*c**2*yp**2)) + 
     &          b*d*(2*a*(c + c**2 + 2*musq_i - 4*musq_i**2)*
     &          DLog(-((-c - 2*musq_i)**2.5
     &          *(1 + c - 2*musq_i)*(-1 + yp))) - 
     &          2*a*(c + c**2 + 2*musq_i - 4*musq_i**2)*
     &          DLog((-c - 2*musq_i)**2.5
     &          *(1 + c - 2*musq_i)*(1 - alpha_ff*yp)) + 
     &          2*c*Sqrt(-c + 2*musq_i)*DLog(-2*(b + c*yp)) 
     &          + 3*c**2*Sqrt(-c + 2*musq_i)*DLog(-2*(b + c*yp)) - 
     &          4*c*musq_i*Sqrt(-c + 2*musq_i)*DLog(-2*(b + c*yp)) 
     &          - 2*c*Sqrt(-c + 2*musq_i)*Log(-2*(d + alpha_ff*c*yp)) - 
     &          3*c**2*Sqrt(-c + 2*musq_i)*DLog(-2*(d + alpha_ff*c*yp)) 
     &          + 4*c*musq_i*Sqrt(-c + 2*musq_i)*
     &          DLog(-2*(d + alpha_ff*c*yp)) - 
     &          2*a*c*DLog(-4*musq_i**2 - b*Sqrt(c**2 - 4*musq_i**2) +
     &          c**2*yp) - 2*a*c**2*DLog(-4*musq_i**2 -
     &          b*Sqrt(c**2 - 4*musq_i**2) + c**2*yp) - 
     &          4*a*musq_i*DLog(-4*musq_i**2 - b*Sqrt(c**2 - 4*musq_i**2)
     &          + c**2*yp) + 8*a*musq_i**2*DLog(-4*musq_i**2 -
     &          b*Sqrt(c**2 - 4*musq_i**2) + c**2*yp) + 
     &          2*a*c*DLog(-4*musq_i**2 - d*Sqrt(c**2 - 4*musq_i**2) +
     &          alpha_ff*c**2*yp) + 2*a*c**2*Log(-4*musq_i**2 -
     &          d*Sqrt(c**2 - 4*musq_i**2) + alpha_ff*c**2*yp) + 
     &          4*a*musq_i*DLog(-4*musq_i**2 - d*Sqrt(c**2 - 4*musq_i**2)
     &          + alpha_ff*c**2*yp) - 8*a*musq_i**2*DLog(-4*musq_i**2 -
     &          d*Sqrt(c**2 - 4*musq_i**2) + alpha_ff*c**2*yp)))/
     &          (3.*c*(-c+2*musq_i)**1.5*Sqrt(-4*musq_i**2 + c**2*yp**2)
     &          *Sqrt(-4*musq_i**2+alpha_ff**2*c**2*yp**2)))/
     &          (8.*Pi**2)/ca


          endif

C  2. g->qq splitting
       elseif(id1.eq.1.and. mi.eq.0d0) then
          fi(2)=(gsq*((-1 + muk)*(-6*L*(1 + muk) - 4*(4 + muk)
c$$$     &         + 9*(1 + muk)*rs) + 12*(-1 + muk**2)*DLog(1 - muk) + 
c     RF No scheme dependence here. All in the g -> gg.
     &         ) + 12*(-1 + musq_k)*DLog(1 - muk) + 
     &         12*musq_k*DLog((2*muk)/(1 + muk))))/
     &         (144.*(-1 + musq_k)*pi**2)/ca
C     adding alpha dependent terms
          if(alpha_ff.ne.1d0) then
             fi(2)=fi(2)-(gsq*(-((-1 + alpha_ff)*(-1 + muk)**2) + 
     &            DLog(alpha_ff) + musq_k*(DLog(4d0) + DLog(1/alpha_ff)
     &            + 2*DLog(muk) - 2*DLog(1 + muk) - 
     &            2*DLog(1+alpha_ff-(2*alpha_ff)/(1+muk))))/
     &            (3.*(-1+musq_k)))/(8.*pi**2)/ca
          endif

C  3. g->gg splitting
       elseif(id1.eq.0) then
          fi(2)=(gsq*((200 + 66*L + 9*L**2-(132*muk)/(1 + muk)-15*Pi**2
     &         - 132*DLog(1 - muk) - (24*musq_k*DLog((2*muk)/(1 +
     &         muk)))/(-1 + musq_k) +66*DLog(1 - musq_k) + 9*(4*(L -
     &         DLog(muk))*DLog(muk) - 2*(L + 2*DLog(muk))*DLog(1 -
     &         musq_k) + DLog(1 - musq_k)**2) - 36*ddilog(1 - musq_k)))
     &         /18d0)/(8.*pi**2)

C     adding alpha dependent terms
          if (alpha_ff.ne.1d0) then
             yl=1d0 + alpha_ff*(-1 + muk)**2 - musq_k - 
     &            Sqrt(abs((-1+muk)**2*(alpha_ff**2*(-1+muk)**2+
     &            (1+muk)**2-2*alpha_ff*(1+musq_k))))
             fi(2)=fi(2)-(gsq*((11*(-2+2*muk+yl)**2)/
     &            ((-1+musq_k)*(-2+yl))
     &            -44*Log(2-2*muk) - 22*Log(muk) + 24*Log(2/(1 + muk))*
     &            (Log(2/(1 + muk)) + 2*Log(1 + muk)) +
     &            (2*((-11 + 15*musq_k)*Log(2*muk) + 
     &            4*musq_k*(-Log(-8*(-1+muk)*musq_k)+
     &            Log((-2 + yl)**2+4*musq_k*(-1 + yl))) + 
     &            (11 - 15*musq_k)*Log(2 - yl)))/(-1 + musq_k) + 
     &            22*Log(2 - 2*musq_k - yl) + 
     &            22*Log(yl)-12*(4*Log(1-yl/2.)*Log(-(yl/(-1+musq_k)))-
     &            Log(-(yl/(-1+musq_k)))**2 + 
     &            Log((-2*(-2 + 2*musq_k + yl))/
     &            ((-1 + musq_k)*(-2 + yl)))**2 + 
     &            2*Log(-(yl/(-1+musq_k)))*(
     &            Log((-2*(-2+2*musq_k+yl))/((-1+musq_k)*(-2+yl))) - 
     &            2*Log(1 + yl/(-2 + 2*musq_k)))) + 48*ddilog(1 - muk) -
     &            48*ddilog(1/(1 + muk)) - 
     &            48*ddilog(yl/2.) + 48*ddilog(yl/(2 - 2*musq_k))))/
     &            (48.*pi**2)
          endif
       endif

C ij~ gluon and massless spectator
      elseif(mk.eq.0d0.and.id.eq.0) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/6d0
         endif
C     1. g->QQ splitting
         if(id1.eq.1.and.mi.gt.0d0) then
            fi(2)=(gsq*(-16*Sqrt(1-4*musq_i)+16*musq_i*Sqrt(1-4*musq_i)+
c$$$     &           9*rs-12*DLog(Sqrt(musq_i))+ 
c     RF No scheme dependence here. All in the g -> gg.
     &           12*DLog(Sqrt(musq_i))+ 
     &           12*DLog((1 + Sqrt(1 - 4*musq_i))/2.)))/(144.*pi**2)/ca
C adding alpha dependent terms
            if (alpha_ff.ne.1d0) then
               fi(2)=fi(2)-(gsq*(Sqrt(1-4*musq_i)+
     &              Sqrt(-4*musq_i**2+alpha_ff**2*(1-2*musq_i)**2) + 
     &              (2*Sqrt(-4*musq_i**2+alpha_ff**2*(1-2*musq_i)**2))/
     &              (-alpha_ff+2*(-1+alpha_ff)*musq_i) + 
     &              (-1 + 2*musq_i)*(-2*DATAN((2*musq_i)/
     &              Sqrt(1 - 4*musq_i)) + 
     &              2*DATan((2*musq_i)/
     &              Sqrt(-4*musq_i**2 + alpha_ff**2*(1 - 2*musq_i)**2))+ 
     &              DLog(-2*(-1 + 2*musq_i + Sqrt(1 - 4*musq_i))) - 
     &              DLog(-2*(alpha_ff*(-1+2*musq_i)+
     &              Sqrt(-4*musq_i**2+alpha_ff**2*(1-2*musq_i)**2))))))/
     &              (24.*pi**2)/ca
            endif
C  2. g->qq splitting
         elseif(id1.eq.1.and. mi.eq.0d0) then
c$$$            fi(2)=-(gsq*(8 + 3*L - 9*rs*ca))/(144.*pi**2)/ca
c     RF No scheme dependence here. All in the g -> gg.
            fi(2)=-(gsq*(8 + 3*L))/(144.*pi**2)/ca

C adding alpha dependent terms
            if (alpha_ff.ne.1d0) then
               fi(2)=fi(2)-(gsq*(-1 + alpha_ff - DLog(alpha_ff)))/
     &              (24.*pi**2)/ca
            endif

C  3. g->gg splitting
         elseif(id1.eq.0) then
            fi(2)=(gsq*((200+6*L*(11+3*L)-21*pi**2)+36*rs))/(144.*pi**2)

C adding alpha dependent terms
            if (alpha_ff.ne.1d0) then
               fi(2)=fi(2)-(gsq*(11-11*alpha_ff+11*DLog(alpha_ff)+
     &              6*DLog(alpha_ff)**2))/(24.*pi**2)
            endif
         endif
      endif

      end


      SUBROUTINE finitefi(mi,sik,sikzone,x,id1,id2,fi)
c  calculates the finite terms when emitter is in
c  final state and spectator is in initial state
      implicit none

      include "coupl.inc"
      include 'dipole.inc'
c Arguments
      REAL*8 mi,qsq,sik,x,fi(9),sikzone
      INTEGER id1,id2
c Global
      REAL*8 ddilog
      external ddilog
c Local
      INTEGER i
      REAl*8 cf,ca,rhoi,rhok,rho,musq_i,gsq,pi
      REAL*8 L,L_one,rs,musq_i_one
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926535897932385d0)

      musq_i=mi**2/sik
      musq_i_one=mi**2/sikzone
      gsq=GG(1)**2
      L_one=DLog(mu**2/sikzone)

      do i=1,9
         fi(i)=0d0
      enddo

C  massive quark (ij~: quark)
      if(mi .gt. 0d0 .and. id1.eq.1) then
         rs=0d0
         fi(2)=(gsq*((1d0+DLOG(musq_i_one/(musq_i_one+1d0)))*L_one
     &        -2d0*ddilog(-musq_i_one)-pi**2/3d0+2d0+
     &        DLOG(musq_i_one)**2/2d0+DLOG(1+musq_i_one)**2/2d0-
     &        2d0*DLOG(musq_i_one)*DLOG(1+musq_i_one)+DLOG(musq_i_one)))
     &        /(8.*pi**2)

c adding the alpha dependent term
         if(alpha_fi.ne.1d0) then
            fi(2)=fi(2)+gsq/(8.*pi**2)*2.*DLog(alpha_fi)*
     &           (DLog((1d0+musq_i_one)/(musq_i_one))-1d0)
         endif
         if(x.gt.(1d0-alpha_fi)) then
            fi(1)=gsq/(8.*pi**2)*((1.-x)/(2.*(1-x+musq_i)**2)+2./
     &           (1-x)*DLog(((2.-x+musq_i)*musq_i_one)
     &           /((1.+musq_i_one)*(1-x+musq_i))))
            fi(3)=gsq/(8.*pi**2)*2./(1-x)*
     &           (DLOG((1+musq_i_one)/(musq_i_one))-1d0)
            fi(4)=fi(3)
         endif

C  massless quark (ij~: quark)
      elseif(mi.eq.0d0 .and. id1.eq.1) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/2d0
         endif
         fi(2)=(gsq*(6*(7+L_one*(3+L_one))-7*pi**2+12*rs))/(96.*pi**2)
c     adding the alpha dependent term
         if (alpha_fi.ne.1d0) then
            fi(2)=fi(2)+gsq/(8.*pi**2)*(-3./2.*Dlog(alpha_fi)-
     &           Dlog(alpha_fi)**2)
         endif
         if(x.gt.(1d0-alpha_fi)) then
            fi(1)=gsq/(4d0*pi**2*(1-x))*DLog(2d0-x)
            fi(3)=-(gsq*(3/(1-x)+(4*DLog(1-x))/(1-x)))/(16.*pi**2)
            fi(4)=fi(3)
         endif

C  gluon (ij~: gluon)
      elseif(id1.eq.0) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/6d0
         endif
C  1. g-> QQ (i:massive quark )
         if(id2.eq.1.and.mi.gt.0d0) then
            fi(7)=-(gsq*(5*Sqrt(1-4*musq_i_one)+
     &           4*musq_i_one*Sqrt(1-4*musq_i_one)+DLog(64d0) + 
     &           3*DLog(musq_i_one) -
     &           6*DLog(1+Sqrt(1-4*musq_i_one))))/(72.*pi**2)/ca
c adding the alpha dependent term
            if (alpha_fi.ne.1d0) then
               fi(7)=fi(7)+(gsq*((5 - 16*musq_i_one**2 -
     &              5*Sqrt(((1 - 4*musq_i_one)*
     &              (alpha_fi - 4*musq_i_one))/alpha_fi) - 
     &              4*musq_i_one*(4 + Sqrt(((1 - 4*musq_i_one)*
     &              (alpha_fi - 4*musq_i_one))/
     &              alpha_fi**3)))/Sqrt(1 - 4*musq_i_one)-
     &              6*DLog(1+Sqrt(1-4*musq_i_one)) + 
     &              6*DLog(Sqrt(alpha_fi)+
     &              Sqrt(alpha_fi-4*musq_i_one))))/(72.*pi**2)/ca
            endif
            if(x.lt.1d0-4d0*musq_i)then
               if(x.gt.(1d0-alpha_fi)) then
                  fi(8)=(gsq*Sqrt(1+(4*musq_i)/(-1 +x))*
     &                 (1 + 2*musq_i - x))/(24.*pi**2*(-1 + x)**2)/ca
                  fi(9)=fi(8)
               endif
            endif

c  2. g-> qq (i:massless quark)
         elseif(id2.eq.1.and.mi.eq.0d0) then
c     RF No scheme dependence here. All in the g -> gg.
            fi(2)=-(gsq*(10 + 6*L_one))/(144.*pi**2)/ca
c     adding the alpha dependent term
            if(alpha_fi.ne.1d0) then
               fi(2)=fi(2)+gsq/(8d0*pi**2)*DLog(alpha_fi)/3./ca
            endif
            if(x.gt.(1d0-alpha_fi)) then
               fi(3)=gsq/(24.*pi**2*(1 - x))/ca
               fi(4)=fi(3)
            endif

C  3. g-> gg (i: gluon )
         elseif(id2.eq.0) then
            fi(2)=(gsq*(134 + 6*L_one*(11 + 3*L_one) -
     &           21*pi**2 + 36*rs))/(144.*pi**2)
c adding the alpha dependent term
            if (alpha_fi.ne.1d0) then
               fi(2)=fi(2)+gsq/(8.*pi**2)*
     &              (-2d0*DLOG(alpha_fi)**2-11d0/3d0*DLOG(alpha_fi))
            endif
            if(x.gt.(1d0-alpha_fi)) then
               fi(1)=-(gsq*DLog(2 - x))/(2.*pi**2*(-1 + x))
               fi(3)=(gsq*(-11/(1-x)+(12*DLog(1-x))/(-1+x)))/(24.*pi**2)
               fi(4)=fi(3)
            endif
         endif
      endif

      end


      SUBROUTINE finiteif(mk,sik,sikzone,x,id1,id2,fi)
c  calculates the finite terms when the emitter is
c  in the initial state and the spectator is in the
c  final state.
      implicit none
      include "coupl.inc"
      include "dipole.inc"

c Arguments
      REAL*8 mk,sik,sikzone,x,fi(9)
      INTEGER id1,id2
c Global
      REAL*8 ddilog
      external ddilog
c Local
      INTEGER i
      REAl*8 cf,ca,pi,gsq
      REAL*8 zp,musq_k,L,L_one,rs,musq_k_one
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926535897932385d0)

      gsq=GG(1)**2
      musq_k=mk**2/sik
      musq_k_one=mk**2/sikzone
      zp=(1d0-x)/(1d0-x+musq_k)
      L=DLog(mu**2/sik)
      L_one=DLog(mu**2/sikzone)

      do i=1,9
         fi(i)=0d0
      enddo
      
c quark-quark splitting
      if(id1 .eq.1.and.id2.eq.1) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/2d0
         endif
         fi(2)=(gsq*(2*L_one**2-pi**2+4*rs+6*DLog(mu**2/muf**2) + 
     &        2*DLog(1 + musq_k_one)*(2*L_one + DLog(1 + musq_k_one)) + 
     &        8*ddilog(1/(1 + musq_k_one))))/(32.*pi**2)
         fi(1)=-(gsq*(-((-1 + x**2)*(L - DLog(mu**2/muf**2)))+ 
     &        (-1+x**2)*DLog(1 - x) - 2*DLog(2 - x) + 
     &        (-1+x)*(-1 + x +(1+x)*DLog((-1+x)/(-1-musq_k+x)))))/
     &        (8.*pi**2*(-1 + x))
         fi(3)=(gsq*(L-DLog(mu**2/muf**2)-2*DLog(1 - x)))/
     &        (4.*pi**2*(-1 + x))
         fi(4)=(gsq*(L_one-DLog(mu**2/muf**2)-2*DLog(1 - x)))/
     &        (4.*pi**2*(-1 + x))
         if (mk.ne.0d0) then
            fi(5)=gsq/(4d0*pi**2)*Dlog((2d0-x)/(2d0-x+musq_k))/(1d0-x)
            fi(6)=gsq/(4d0*pi**2)*Dlog(1d0/(1d0+musq_k_one))/(1d0-x)
         endif
c adding the alpha dependent term
         if(zp.gt.alpha_if) then
            fi(1)=fi(1)-(gsq*(-((1+x)*DLog(zp/alpha_if)) + 
     &           (2*DLog(((1+alpha_if-x)*zp)/(alpha_if*(1-x+zp))))/
     &           (1 - x)))/(8.*pi**2)
         endif

c quark-gluon splitting (i:quark, ij~: gluon)
      elseif(id1.eq.1.and.id2.eq.0 ) then
         fi(1)=(cf*gsq*(x**2 - L*(2 + (-2 + x)*x) + 
     &        (2 + (-2 + x)*x)*DLog(mu**2/muf**2) + 
     &        (2 + (-2 + x)*x)*DLog(1 - x) + 
     &        (2 + (-2 + x)*x)*DLog((-1 + x)/(-1 - musq_k + x))))/
     &        (8.*pi**2*x)/ca
c adding the alpha dependent term
         if(zp.gt.alpha_if) then
            if(musq_k.gt.0d0) then
               fi(1)=fi(1)-
     &              (cf*gsq*((2*musq_k*DLog((1 - zp)/(1 - alpha_if)))/x 
     &              + ((1 + (1 - x)**2)*DLog(zp/alpha_if))/x))/
     &              (8.*pi**2)/ca
            elseif(musq_k.eq.0d0) then
               fi(1)=fi(1)-(cf*gsq*(2-2*x+x**2)*
     &              DLog(zp/alpha_if))/(8.*pi**2*x)/ca
            endif
         endif

c gluon-quark splitting (i:gluon, ij~: quark)
      elseif(id1.eq.0.and.id2.eq.1) then
         fi(1)=(gsq*(-L + 2*(1 + L)*x - 2*(1 + L)*x**2 + 
     &        (1 + 2*(-1 + x)*x)*DLog(mu**2/muf**2) + 
     &        (1 + 2*(-1 + x)*x)*DLog(1 - x) + 
     &        (1 + 2*(-1 + x)*x)*DLog((-1 + x)/(-1 - musq_k + x))))
     &        /(16.*pi**2)/cf
c adding the alpha dependent term
         if(zp.gt.alpha_if) then
            fi(1)=fi(1)-(gsq*((1-x)**2+x**2)*
     &           DLog(zp/alpha_if))/(16.*pi**2)/cf
         endif

c gluon-gluon splitting (i:gluon, ij~:gluon)
      elseif(id1.eq.0.and.id2.eq.0) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/6d0
         endif
         if(musq_k.gt.0d0) then
            fi(2)=(gsq*(6*L_one**2 - 3*pi**2 + 12*rs 
     &           + (22 - 4*Nf/ca)*DLog(mu**2/muf**2) + 
     &           6*DLog(1 + musq_k_one)*(2*L_one + Log(1 + musq_k_one)) 
     &           + 24*ddilog(1/(1 + musq_k_one))))/(96.*pi**2)
            fi(1)=(gsq*(L - 3*L*x + 3*L*x**2 - 2*L*x**3 + L*x**4 - 
     &           (-1 + x)*(-1 + x*(2 + (-1 + x)*x))*DLog(mu**2/muf**2) - 
     &           (-1 + x)*(-1 + x*(2 + (-1 + x)*x))*DLog(1 - x) 
     &           + x*DLog(2 - x) - musq_k*DLog(musq_k/(1 + musq_k - x))
     &           + musq_k*x*DLog(musq_k/(1 + musq_k - x)) - 
     &           (-1 + x)*(-1 + x*(2 + (-1 + x)*x))*Log((-1 + x)/
     &           (-1 - musq_k + x))))/(4.*pi**2*(-1 + x)*x)
            fi(4)=(gsq*(L_one - DLog(mu**2/muf**2) - 2*DLog(1 - x)))/
     &           (4.*pi**2*(-1 + x))
            fi(3)=(gsq*(L - DLog(mu**2/muf**2) - 2*DLog(1 - x)))/
     &           (4.*pi**2*(-1 + x))
            fi(5)=gsq/(4d0*pi**2)*DLog((2d0-x)/(2d0-x+musq_k))/(1d0-x)
            fi(6)=gsq/(4d0*pi**2)*Dlog((1d0)/(1d0+musq_k_one))/(1d0-x)
c     adding the alpha dependent term
            if(zp.gt.alpha_if) then
               fi(1)=fi(1)+(gsq*((-2*musq_k*DLog((1-zp)/(1-alpha_if)))/x 
     &              -2*(-1+(1 - x)/x + (1 - x)*x)*DLog(zp/alpha_if) + 
     &              (2*DLog((alpha_if*(1-x+zp))/((1+alpha_if-x)*zp)))/
     &              (1 - x)))/(8.*pi**2)
            endif
         elseif(musq_k.eq.0d0) then
            fi(2)=(gsq*((6*L_one**2 + pi**2) + 12*rs 
     &           + (22 - 4*Nf/ca)*DLog(mu**2/muf**2)))/(96.*pi**2)
            fi(1)=(gsq*((-1 + x*(2 + (-1 + x)*x))*
     &           (L-DLog(mu**2/muf**2)-DLog(1-x))-
     &           x*Dlog(2-x)/(1-x)))/(4.*pi**2*x)
            fi(4)=(gsq*(L_one-DLog(mu**2/muf**2)-2*DLog(1-x)))/
     &           (4.*pi**2*(-1 + x))
            fi(3)=(gsq*(L-DLog(mu**2/muf**2)-2*DLog(1-x)))/
     &           (4.*pi**2*(-1 + x))
c     adding the alpha dependent term
            if(zp.gt.alpha_if) then
               fi(1)=fi(1)+(gsq*((1 - 3*x + 3*x**2 
     &              - 2*x**3 + x**4)*DLog(zp/alpha_if) - 
     &              x*DLog((alpha_if*(1 - x + zp))/
     &              ((1 + alpha_if - x)*zp))))/(4.*pi**2*(-1 + x)*x)
            endif
         endif
      endif

      end


      SUBROUTINE finiteii(sik,sikzone,x,id1,id2,fi)
c  calculates the finite terms when both emitter
c  and spectator are in the initial state.
      implicit none
      include "coupl.inc"
      include "dipole.inc"

c Arguments
      REAL*8 sik,sikzone,x,fi(9)
      INTEGER id1,id2
c Global
      REAL*8 ddilog
      external ddilog
c Local
      INTEGER i
      REAl*8 cf,ca,pi,gsq
      REAL*8 L,L_one,rs
      PARAMETER (cf=4d0/3d0,ca=3d0)
      PARAMETER (pi=3.1415926535897932385d0)

      gsq=GG(1)**2
      L=dlog(mu**2/sik)
      L_one=dlog(mu**2/sikzone)

      do i=1,9
         fi(i)=0d0
      enddo

c quark-quark splitting (i:quark, ij~: quark)
      if(id1 .eq.1.and.id2.eq.1) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/2d0
         endif
         fi(2)=(gsq*(2*L_one**2-pi**2+4*rs+6*DLog(mu**2/muf**2)))/
     &        (32.*pi**2)
         fi(1)=-(gsq*(-1 + x + (-1 - x)*(L -dLog(mu**2/muf**2)) 
     &        + 2*(1 + x)*dLog(1 - x)))/(8.*pi**2)
         fi(4)=-(gsq*((2*(L_one - dLog(mu**2/muf**2)))/(1 - x)
     &        + (4*dLog(1 - x))/(-1 + x)))/(8.*pi**2)
         fi(3)=-(gsq*((2*(L - dLog(mu**2/muf**2)))/(1 - x)
     &        + (4*dLog(1 - x))/(-1 + x)))/(8.*pi**2)
c     adding the alpha dependent term
         if((1d0-x).gt.alpha_ii) then
            fi(1)=fi(1)+(gsq*(1+x**2)*DLog(alpha_ii/(1-x)))/
     &           (8.*pi**2*(1-x))
         endif


c quark-gluon splitting (i:quark, ij~:gluon)
      elseif(id1.eq.1.and.id2.eq.0) then
         fi(1)=(cf*gsq*(x**2 - L*(2 + (-2 + x)*x)
     &        +(2 + (-2 + x)*x)*dLog(mu**2/muf**2) + 
     &        2*(2 + (-2 + x)*x)*DLog(1 - x)))/(8.*pi**2*x)/ca
c     adding the alpha dependent term
         if((1d0-x).gt.alpha_ii) then
            fi(1)=fi(1)+(cf*gsq*(1+(1-x)**2)*DLog(alpha_ii/(1-x)))/
     &           (8.*pi**2*x)/ca
         endif


c gluon-quark splitting (i:gluon, ij~:quark)
      elseif(id1.eq.0.and.id2.eq.1) then
         fi(1)=(gsq*(-L + 2*(1 + L)*x - 2*(1 + L)*x**2 
     &        + (1 + 2*(-1 + x)*x)*DLog(mu**2/muf**2) + 
     &        (2 + 4*(-1 + x)*x)*DLog(1 - x)))/(16.*pi**2)/cf
c adding the alpha dependent term
         if((1d0-x).gt.alpha_ii) then
            fi(1)=fi(1)+(gsq*((1-x)**2+x**2)*DLog(alpha_ii/(1-x)))/
     &           (16.*pi**2)/cf
         endif

c gluon-gluon splitting (i:gluon, ij~:gluon)
      elseif(id1.eq.0.and.id2.eq.0) then
         if(scheme .eq. 'HV') then
            rs=0d0
         elseif(scheme .eq. 'DR') then
            rs=-1d0/6d0
         endif
         fi(2)=(gsq*(6*L_one**2 - 3*pi**2 + 12*rs + (22 -
     &        4*Nf/ca)*DLog(mu**2/muf**2)))/(96.*pi**2)
         fi(1)=(gsq*(-1 + x*(2 + (-1 + x)*x))*
     &        (L-DLog(mu**2/muf**2)-2*DLog(1-x)))/(4.*pi**2*x)
         fi(4)=-(gsq*((2*(L_one-DLog(mu**2/muf**2)))/(1 - x)
     &        + (4*DLog(1 - x))/(-1 + x)))/(8.*pi**2)
         fi(3)=-(gsq*((2*(L-DLog(mu**2/muf**2)))/(1 - x)
     &        + (4*DLog(1 - x))/(-1 + x)))/(8.*pi**2)
c adding the alpha dependent term
         if((1d0-x).gt.alpha_ii) then
            fi(1)=fi(1)+(gsq*(-1+(1-x)*x)**2*DLog(alpha_ii/(1-x)))/
     &           (4.*pi**2*(1-x)*x)
         endif
      endif
      end


*
* dilog64.F,v 1.1.1.1 1996/04/01 15:02:05 mclareni
*
* Revision 1.1.1.1  1996/04/01 15:02:05  mclareni
* Mathlib gen
*
*
      FUNCTION DDILOG(X)
*
* imp64.inc,v 1.1.1.1 1996/04/01 15:02:59 mclareni Exp
*
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







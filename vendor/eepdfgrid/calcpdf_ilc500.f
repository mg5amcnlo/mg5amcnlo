      real*8 function Iapqterm1(r)
      implicit none
      real*8 pa,pb,rb,t,z,g,r
      real*8 x,a,p,q,beta,ra
      common /xapqbeta/ x,a,p,q,beta,ra
      real*8 isr_tilde
      external isr_tilde
      pa=-2d0/3d0
      pb=beta-1d0
      t=ra*r**(1d0/(pa+1))
      z=t*(1d0-x)+x
      g=a*dexp(p*(1d0-x/z))*dexp(q*(1d0-x/z)**1.5d0)*isr_tilde(z,beta)
      g=g*z**(-1d0/3d0)
      Iapqterm1=g*(1d0-t)**pb*ra**(pa+1d0)/(pa+1d0)
      end

      real*8 function Iapqterm2(r)
      implicit none
      real*8 pa,pb,rb,t,z,g,r
      real*8 x,a,p,q,beta,ra
      common /xapqbeta/ x,a,p,q,beta,ra
      real*8 isr_tilde
      external isr_tilde
      pa=-2d0/3d0
      pb=beta-1d0
      rb=1d0-ra
      t=1d0-rb*r**(1d0/(pb+1))
      z=t*(1d0-x)+x
      g=a*dexp(p*(1d0-x/z))*dexp(q*(1d0-x/z)**1.5d0)*isr_tilde(z,beta)
      g=g*z**(-1d0/3d0)
      Iapqterm2=g*t**pa*rb**(pb+1d0)/(pb+1d0)
      end


      real*8 function Iapq(xpar,apar,ppar,qpar,betapar)
      implicit none
      real*8 Iapqterm1,Iapqterm2
      external Iapqterm1,Iapqterm2
      real*8 dgauss
      external dgauss
      real*8 xpar,apar,ppar,qpar,betapar
      real*8 res1,res2
      real*8 x,a,p,q,beta,ra
      common /xapqbeta/ x,a,p,q,beta,ra
      real*8 intlow,intupp,inteps
      x = xpar
      a = apar
      p = ppar
      q = qpar
      beta = betapar
      ra=0.5d0
      intlow = 0d0
      intupp = 1d0
      inteps = 1d-5
      res1 = dgauss(Iapqterm1,intlow,intupp,inteps)
      res2 = dgauss(Iapqterm2,intlow,intupp,inteps)
      Iapq = res1+res2
      end


c     This function calculate the reduced structure function, with energy
c     fraction given by "x", at scale "Qsquare"
      real*8 function eepdf_tilde_calc(x,Qsquare,n,partonid,beamid)
      implicit none
      real*8 x,Qsquare
      real*8 me
      data me /0.511d-3/
      real*8 PI
      real*8 alphaem
c     In Gmu scheme
      data alphaem/0.007562397d0/
      real*8 beta
      integer n,partonid,beamid
      real*8 a11,a01,a10,a00
      real*8 p1,q1,p2,q2,p3,q3,p4,q4
      data a11/0.5012d0/,a01/0.1613d0/,a10/0.1613d0/,a00/0.06379d0/
      data p1/-8.514d0/,q1/-5.808d0/,p2/-8.505d0/,q2/-5.823d0/
      data p3/-7.535d0/,q3/-6.790d0/,p4/-7.481d0/,q4/-6.849d0/
      real*8 Iapq,isr_tilde
      real*8 res
      data res/0d0/

      PI=4.D0*DATAN(1.D0)
      beta = alphaem/PI * (dlog(Qsquare/me/me)-1d0)

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
          if (partonid .ne. 11) then
             res = 0d0
          else
             if (n .eq. 1) then
                res = dsqrt(a11)*isr_tilde(x,beta)
             else if (n .eq. 2) then
                res = Iapq(x,a01/dsqrt(a11),p1,q1,beta)
             else if (n .eq. 3) then
                res = dsqrt(a11)*isr_tilde(x,beta)
             else if (n .eq. 4) then
                res = Iapq(x,dsqrt(a00),p3,q3,beta)
             else
                res = 0d0
             endif
          endif
      else if (beamid .eq. -11) then
          if (partonid .ne. -11) then
              res = 0d0
          else
             if (n .eq. 1) then
                res = dsqrt(a11)*isr_tilde(x,beta)
             else if (n .eq. 2) then
                res = dsqrt(a11)*isr_tilde(x,beta)
             else if (n .eq. 3) then
                res = Iapq(x,a10/dsqrt(a11),p2,q2,beta)
             else if (n .eq. 4) then
                res = Iapq(x,dsqrt(a00),p4,q4,beta)
             else
                res = 0d0
             endif
          endif
      endif
      eepdf_tilde_calc = res
      end

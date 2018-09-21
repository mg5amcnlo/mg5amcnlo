c     This function return the energy fraction
      real*8 function eepdf_fraction(y,Q2ref,n,partonid,beamid)
      implicit none
      real*8 x,Q2ref,y
      real*8 me
      data me /0.511d-3/
      real*8 PI
      real*8 alphaem
c     In Gmu scheme
      data alphaem/0.007562397d0/
      real*8 beta_r
      integer n,partonid,beamid
      real*8 tx
      real*8 p
      real*8 apeak,aarm,abody
      real*8 anorm1,anorm2
      real*8 a2,a3,a4,a5
      data apeak/0.3388d0/, aarm/0.2371d0/, abody/0.1868d0/
      data a2/12.09d0/, a3/-0.678d0/, a4/11.56d0/, a5/-0.664d0/
      data anorm1/0.817243d0/,anorm2/0.873045d0/

      PI=4.D0*DATAN(1.D0)
      beta_r = alphaem/PI * (dlog(Q2ref/me/me)-1d0)

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
          if (partonid .ne. 11) then
             tx=y
          else
             if (n .eq. 1) then
                tx = y**(1d0/beta_r)
             else if (n .eq. 2) then
                p=a3+beta_r+1d0
                tx=y**(1d0/p)
             else if (n .eq. 3) then
                tx = y**(1d0/beta_r)
             else if (n .eq. 4) then
                p=a5+beta_r+1d0
                tx=y**(1d0/p)
             else
                tx = y
             endif
          endif
      else if (beamid .eq. -11) then
          if (partonid .ne. -11) then
             tx = y
          else
             if (n .eq. 1) then
                tx = y**(1d0/beta_r)
             else if (n .eq. 2) then
                tx = y**(1d0/beta_r)
             else if (n .eq. 3) then
                p=a3+beta_r+1d0
                tx=y**(1d0/p)
             else if (n .eq. 4) then
                p=a5+beta_r+1d0
                tx=y**(1d0/p)
             else
                tx=y
             endif
          endif
      endif
      eepdf_fraction = 1d0-tx
      end

c     This is to calculate the factor for grid implementation
      real*8 function eepdf_factor(y,Q2ref,Q2,n,partonid,beamid)
      implicit none
      real*8 x,Q2ref,y,Q2
      real*8 me
      data me /0.511d-3/
      real*8 PI
      real*8 alphaem
c     In Gmu scheme
      data alphaem/0.007562397d0/
      real*8 beta_r,xcut,beta
      integer n,partonid,beamid
      real*8 apeak,aarm,abody
      real*8 anorm1,anorm2
      real*8 a2,a3,a4,a5
      data apeak/0.3388d0/, aarm/0.2371d0/, abody/0.1868d0/
      data a2/12.09d0/, a3/-0.678d0/, a4/11.56d0/, a5/-0.664d0/
      data anorm1/0.817243d0/,anorm2/0.873045d0/
      real*8 Aabres,isrx
      external Aabres,isrx
      real*8 tx,bpb,tmp
      real*8 res

      PI=4.D0*DATAN(1.D0)
      beta_r = alphaem/PI * (dlog(Q2ref/me/me)-1d0)
      beta = alphaem/PI * (dlog(Q2/me/me)-1d0)

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
          if (partonid .ne. 11) then
             tx=y
             res=1d0
          else
             if (n .eq. 1) then
                res = y**(beta/beta_r-1)
             else if (n .eq. 2) then
                res = y**((beta+a3+1d0)/(beta_r+a3+1d0)-1)
                res = res/(1d0-y)
             else if (n .eq. 3) then
                res = y**(beta/beta_r-1)
             else if (n .eq. 4) then
                res = y**((beta+a5+1d0)/(beta_r+a5+1d0)-1)
                res = res/(1d0-y)
             else
                tx = y
                res=1d0
             endif
          endif
      else if (beamid .eq. -11) then
          if (partonid .ne. -11) then
             tx = y
             res=1d0
          else
             if (n .eq. 1) then
                res = y**(beta/beta_r-1)
             else if (n .eq. 2) then
                res = y**(beta/beta_r-1)
             else if (n .eq. 3) then
                res = y**((beta+a3+1d0)/(beta_r+a3+1d0)-1)
                res = res/(1d0-y)
             else if (n .eq. 4) then
                res = y**((beta+a5+1d0)/(beta_r+a5+1d0)-1)
                res = res/(1d0-y)
             else
                tx=y
                res=1d0
             endif
          endif
      endif
      eepdf_factor = res
      end

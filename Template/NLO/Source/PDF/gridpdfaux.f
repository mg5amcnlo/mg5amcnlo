c     This function return the power of (1-x)
      real*8 function eepdf_tilde_power(Q2,n,partonid,beamid)
      implicit none
      real*8 me
      data me /0.511d-3/
      real*8 PI
      real*8 alphaem
c     In Gmu scheme
      data alphaem/0.007562397d0/
      real*8 beta,Q2
      integer n,partonid,beamid
      real*8 k
      real*8 apeak,aarm,abody
      real*8 anorm1,anorm2
      real*8 a2,a3,a4,a5
      data apeak/0.3388d0/, aarm/0.2371d0/, abody/0.1868d0/
      data a2/12.09d0/, a3/-0.678d0/, a4/11.56d0/, a5/-0.664d0/
      data anorm1/0.817243d0/,anorm2/0.873045d0/

      PI=4.D0*DATAN(1.D0)
      beta = alphaem/PI * (dlog(Q2/me/me)-1d0)

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
        if (partonid .ne. 11) then
          k=0d0
        else
          if (n .eq. 1) then
            k=1d0-beta
          else if (n .eq. 2) then
            k=-beta-a3
          else if (n .eq. 3) then
            k=1d0-beta
          else if (n .eq. 4) then
            k=-beta-a5
          else
            k=0d0
          endif
        endif
      else if (beamid .eq. -11) then
        if (partonid .ne. -11) then
          k=0d0
        else
          if (n .eq. 1) then
            k=1d0-beta
          else if (n .eq. 2) then
            k=1d0-beta
          else if (n .eq. 3) then
            k=-beta-a3
          else if (n .eq. 4) then
            k=-beta-a5
          else
            k=0d0
          endif
        endif
      endif
      eepdf_tilde_power = k
      end

c     This is to calculate the factor for grid implementation
      real*8 function eepdf_tilde_factor(x,Q2,n,partonid,beamid)
      implicit none
      real*8 x,Q2
      real*8 me
      data me /0.511d-3/
      real*8 PI
      real*8 alphaem
c     In Gmu scheme
      data alphaem/0.007562397d0/
      real*8 beta
      integer n,partonid,beamid
      real*8 apeak,aarm,abody
      real*8 anorm1,anorm2
      real*8 a2,a3,a4,a5
      data apeak/0.3388d0/, aarm/0.2371d0/, abody/0.1868d0/
      data a2/12.09d0/, a3/-0.678d0/, a4/11.56d0/, a5/-0.664d0/
      data anorm1/0.817243d0/,anorm2/0.873045d0/
      real*8 tx,bpb,tmp
      real*8 res

      PI=4.D0*DATAN(1.D0)
      beta = alphaem/PI * (dlog(Q2/me/me)-1d0)

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
        if (partonid .ne. 11) then
          res=1d0
        else
          if (n .eq. 1) then
            res = x**(-beta/5d0)
          else if (n .eq. 2) then
            res = x**(-beta/5d0)
          else if (n .eq. 3) then
            res = x**(-beta/5d0)
          else if (n .eq. 4) then
            res = x**(-beta/5d0)
          else
            res = 1d0
          endif
        endif
      else if (beamid .eq. -11) then
        if (partonid .ne. -11) then
          res = 1d0
        else
          if (n .eq. 1) then
            res = x**(-beta/5d0)
          else if (n .eq. 2) then
            res = x**(-beta/5d0)
          else if (n .eq. 3) then
            res = x**(-beta/5d0)
          else if (n .eq. 4) then
            res = x**(-beta/5d0)
          else
            res = 1d0
          endif
        endif
      endif
      eepdf_tilde_factor = res
      end

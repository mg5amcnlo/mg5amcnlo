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
      real*8 isr_tilde
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
                res = isr_tilde(x,beta)
             else
                res = 0d0
             endif
          endif
      else if (beamid .eq. -11) then
          if (partonid .ne. -11) then
              res = 0d0
          else
             if (n .eq. 1) then
                res = isr_tilde(x,beta)
             else
                res = 0d0
             endif
          endif
      endif
      eepdf_tilde_calc = res
      end

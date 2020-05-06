      integer function eepdf_n_components(partonid,beamid)
      implicit none
      integer partonid,beamid
      integer ncom
c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
        if (partonid .ne. 11) then
          ncom=0
        else
          ncom=1
        endif
      else if (beamid .eq. -11) then
        if (partonid .ne. -11) then
          ncom=0
        else
          ncom=1
        endif
      endif
      eepdf_n_components=ncom
      end

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
      real*8 k,b

      PI=4.D0*DATAN(1.D0)
      beta = alphaem/PI * (dlog(Q2/me/me)-1d0)
      b=-2.D0/3.D0

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
        if (partonid .ne. 11) then
          k=0d0
        else
          if (n .eq. 1) then
            k=1d0-beta
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
          else
            k=0d0
          endif
        endif
      endif
      eepdf_tilde_power = k
      end

c     This function return the type of this component
      integer function eepdf_tilde_type(n,partonid,beamid)
      implicit none
      integer n,partonid,beamid
      integer res

c     electron beam
      if (beamid .eq. 11) then
c     other partons are zero
        if (partonid .ne. 11) then
          res=0
        else
          if (n .eq. 1) then
            res=1
          else
            res=0
          endif
        endif
      else if (beamid .eq. -11) then
        if (partonid .ne. -11) then
          res=0
        else
          if (n .eq. 1) then
            res=1
          else
            res=0
          endif
        endif
      endif
      eepdf_tilde_type = res
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
            res = 1d0
          else
            res = 1d0
          endif
        endif
      else if (beamid .eq. -11) then
        if (partonid .ne. -11) then
          res = 1d0
        else
          if (n .eq. 1) then
            res = 1d0
          else
            res = 1d0
          endif
        endif
      endif
      eepdf_tilde_factor = res
      end

      logical function qlzero(psq)
      implicit none
      include 'qlonshellcutoff.f'
      double precision psq
      if (abs(psq) .lt. qlonshellcutoff) then
      qlzero=.true.
      else
      qlzero=.false.
      endif
      return
      end

      logical function qlnonzero(psq)
      implicit none
      include 'qlonshellcutoff.f'
      double precision psq
      if (abs(psq) .gt. qlonshellcutoff) then
      qlnonzero=.true.
      else
      qlnonzero=.false.
      endif
      return
      end

C -------------------------------------------------------------
C
      integer nmxln
      parameter (nmxln=3000)
      integer idt,istchg
      real stmerr,stwerr
      double precision stmass,stwidth
      character*21 stname
      common/stdtbl/ idt(nmxln),istchg(nmxln),
     1             stmerr(2,nmxln),stwerr(2,nmxln),
     2             stmass(nmxln),stwidth(nmxln),stname(nmxln)
      save /stdtbl/
C
C -------------------------------------------------------------

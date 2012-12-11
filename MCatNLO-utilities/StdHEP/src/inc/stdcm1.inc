c -------------------------------------------------------------
c
c   stdecom  - center-of-mass energy
c   stdxsec  - cross-section
c   stdseed1 - random number seed
c   stdseed2 - random number seed
c   nevtreq  - number of events to be generated
c   nevtgen  - number of events actually generated
c   nevtwrt  - number of events written to output file
c   nevtlh   - number of Les Houches events written to output file
c
c  generatorname - name of Monte Carlo generator
c  pdfname       - name of PDF method used
c
      real stdecom,stdxsec
      double precision stdseed1,stdseed2
      integer nevtreq,nevtgen,nevtwrt,nevtlh
      character*20 generatorname, pdfname
      common /stdcm1/ stdecom,stdxsec,stdseed1,stdseed2,
     1                nevtreq,nevtgen,nevtwrt,nevtlh
      common /stdcm2/ generatorname, pdfname
      save /stdcm1/
      save /stdcm2/
c
c -------------------------------------------------------------

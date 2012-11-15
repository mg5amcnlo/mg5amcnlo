c *************************************************************
c  Variables used for the integrated dipoles
c *************************************************************
      logical intdipole
      common /intd/ intdipole


c *************************************************************
c Mass and flavour parameters
c *************************************************************

      real*8 mu              !Renormalization scale
      real*8 muf             !Factorization scale
      common /c_scales/mu,muf

c Number of flavours in the integrated dipoles for initial
c state splittings of gluons. Should be set equal to 
c the number of quark flavors in the initial state.
c Note that for final state splittings, this factor is not used,
c and included gluon to quark splittings are included via
c the explicit real-emission subprocesses. For the
c initial state this cannot be done because of the PDFs.
c (Running of alpha_s is independent of this factor)
      integer Nf
      parameter (Nf=4)


c **************************************************************
c Alpha parameters for phase space restriction
c **************************************************************

      real*8 alpha_ff,alpha_fi,alpha_if,alpha_ii
      parameter (alpha_ff=1d0)
      parameter (alpha_fi=1d0)
      parameter (alpha_if=1d0)
      parameter (alpha_ii=1d0)


c ***************************************************************
c Renormalisation scheme
c Two renormalisation schemes are implemented:
c 1. The 't Hooft-Veltman scheme (scheme=HV)
c 2. dimensional reduction      (scheme=DR)
c ***************************************************************
      character*2 scheme
      parameter (scheme='HV')


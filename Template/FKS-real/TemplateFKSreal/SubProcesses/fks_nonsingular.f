      subroutine smatrix_damped(pp,wgt)
c Wrapper to multiply the matrix elements in sreal (singular terms) or
c smatrix (non-singular terms) with the S-function and the f damping
c function.
      implicit none
      include "nexternal.inc"
      include "fks.inc"

      double precision pp(0:3,nexternal),wgt,xi_i_fks,y_ij_fks

      double precision fks_Sij,f_damp,dot,getyijfks
      external fks_Sij,f_damp,dot,getyijfks

      double precision x,sum
      integer i,configuration

      double precision sqrtshat,shat
      common/parton_cms/sqrtshat,shat

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision zero,tiny
      parameter (zero=0d0)
      parameter (tiny=1d-8)

      if(pp(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
        wgt=0.d0
        return
      endif

      call smatrix(pp,wgt)

C TO TEST THAT THE NORMALIZATION OF THE REAL MATRIX ELEMENT AND N+1 BODY
C PHASE SPACE IS CONSISTENT WITH STANDALONE MADGRAPH, COMMENT THE FOLLOWING,
C AND REQUIRE N+1 JETS IN CUTS.F
         sum=0d0
c Sum over all FKS contributions.
         do i=1,fks_configs
            i_fks=fks_i(i)      ! Needed to update in common block
            j_fks=fks_j(i)
            xi_i_fks=pp(0,i_fks)/(sqrtshat/2d0)
            x=abs(2d0*dot(pp(0,i_fks),pp(0,j_fks))/shat)
            y_ij_fks=getyijfks(pp(0,i_fks),pp(0,j_fks),i_fks,j_fks)
            sum=sum +
     &           fks_Sij(pp,i_fks,j_fks,xi_i_fks,y_ij_fks)*
     &           (1d0-f_damp(x))
         enddo
         wgt=wgt*sum
C TEST: COMMENT UP TO HERE

      return
      end


      function getyijfks(p1,p2,i,j)
      implicit none
      real*8 getyijfks,p1(0:3),p2(0:3)
      integer i,j
      real*8 xm1,xm2,beta1,beta2,tmp,dot
      include "nexternal.inc"
      include "coupl.inc"
      double precision zero
      parameter (zero=0d0)
      double precision pmass(nexternal)
      include "pmass.inc"
c
      xm1=pmass(i)
      xm2=pmass(j)
      if(p1(0).ne.0.d0.and.p2(0).ne.0.d0)then
        if(xm1.eq.0.d0)then
          beta1=p1(0)
        else
          beta1=sqrt(p1(0)**2-xm1**2)
        endif
        if(xm2.eq.0.d0)then
          beta2=p2(0)
        else
          beta2=sqrt(p2(0)**2-xm2**2)
        endif
        tmp=( p1(0)*p2(0)-dot(p1,p2) )/
     #      ( beta1*beta2 )
      else
        tmp=1.d0
      endif
      getyijfks=tmp
      return
      end





      subroutine setfksfactor(iconfig)
      implicit none

      double precision CA,CF,Nf,PI
c$$$      parameter (CA=3d0,CF=4d0/3d0,Nf=5d0)
C SET NF=0 WHEN NOT CONSIDERING G->QQ SPLITTINGS. FOR TESTS ONLY
      parameter (CA=3d0,CF=4d0/3d0,Nf=0d0)
      PARAMETER (PI=3.1415926d0)

      double precision c(0:1),gamma(0:1),gammap(0:1)
      common/fks_colors/c,gamma,gammap

      logical softtest,colltest
      common/sctests/softtest,colltest

      integer config_fks,i,j,iconfig,fac1,fac2

      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks

      include 'coupl.inc'
      include 'nexternal.inc'
      include 'fks_powers.inc'
      include 'fks.inc'

      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision delta_used
      common /cdelta_used/delta_used
      logical xexternal
      common /toxexternal/ xexternal
      logical rotategranny
      common/crotategranny/rotategranny
      integer diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor

      character*1 integrate


c The value of rotategranny may be superseded later if phase space
c parametrization allows it
      rotategranny=.false.

      softtest=.false.
      colltest=.false.
      open (unit=19,file="config.fks",status="old")
      read (19,*) config_fks
      close (19)
      if (fks_j(config_fks).gt.nincoming)then
         delta_used=deltaO
      else
         delta_used=deltaI
      endif
      
      xicut_used=xicut

      c(0)=CA
      c(1)=CF
      gamma(0)=( 11d0*CA-2d0*Nf )/6d0
      gamma(1)=CF*3d0/2d0
      gammap(0)=(67d0/9d0-2d0*PI**2/3d0)*CA-23d0/18d0*Nf
      gammap(1)=(13/2d0-2d0*PI**2/3d0)*CF

      return
      end

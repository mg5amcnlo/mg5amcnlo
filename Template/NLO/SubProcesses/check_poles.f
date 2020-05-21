      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      use mint_module
      use FKSParams
      implicit none
C
C     CONSTANTS
C
      double precision pi, zero
      parameter (pi=3.1415926535897932385d0)
      parameter (zero = 0d0)
      integer npointsChecked
      integer i, j, k
      integer return_code
      double precision tolerance, tolerance_default
      double precision, allocatable :: accuracies(:)
      double precision ren_scale, energy
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      double precision p(0:3, nexternal), prambo(0:3,100)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision pswgt
      double precision fks_double, fks_single
      double precision, allocatable :: virt_wgts(:,:)
      double precision double, single, finite
      double precision born, virt_wgt
      double precision totmass
      logical calculatedborn
      common/ccalculatedborn/calculatedborn
      logical fksprefact
      parameter (fksprefact=.true.)
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg,symfactvirt
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                  fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
cc
      include 'run.inc'
      include 'coupl.inc'
      include 'q_es.inc'
      integer nsqso,MLResArrayDim
      double precision pmass(nexternal), pmass_rambo(100)
      integer nfail
      logical first_time
      data first_time/.TRUE./
      double precision tiny
      parameter (tiny = 1d-12)
      integer getordpowfromindex_ml5
      logical, allocatable, save :: keep_order(:)
      include 'orders.inc'
      logical is_aorg(nexternal)
      common /c_is_aorg/is_aorg
      logical force_polecheck, polecheck_passed
      common /to_polecheck/force_polecheck, polecheck_passed
      integer ret_code_ml
      common /to_ret_code/ret_code_ml
      
C-----
C  BEGIN CODE
C-----  
      force_polecheck = .true.
      if (first_time) then
          call get_nsqso_loop(nsqso)          
          call get_answer_dimension(MLResArrayDim)
          allocate(virt_wgts(0:3,0:MLResArrayDim))
          allocate(accuracies(0:nsqso))
          allocate(keep_order(nsqso))
          first_time = .false.
      endif

      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts and particle masses
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
      include 'pmass.inc'
     
      call FKSParamReader('FKS_params.dat',.TRUE.,.FALSE.)
      tolerance_default = IRPoleCheckThreshold
      iconfig=1
      ichan=1
      iconfigs(1)=iconfig
c     Set the energy to be characteristic of the run
      totmass = 0.0d0
      do i=1,nexternal
        totmass = totmass + pmass(i)
      enddo
      energy = max((ebeam(1)+ebeam(2))/20.0d0,2.0d0*totmass)
c     Set the renormalization scale to be of the order of sqrt(s) but
c     not equal to it so as to be sensitive to all logs in the check.
      ren_scale = energy/2.0d0

      write(*,*)' Insert the number of points to test'
      read(*,*) npoints
      write(*,*)'Insert the relative tolerance'
      write(*,*)' A negative number will mean use the default one: ',
     1 tolerance_default 
      read(*,*) tolerance
      if (tolerance .le. zero) then
          tolerance = tolerance_default
      else
          IRPoleCheckThreshold = tolerance
      endif

      mu_r = ren_scale
      qes2 = ren_scale**2

      do i = nincoming+1, nexternal-1
          pmass_rambo(i-nincoming) = pmass(i)
      enddo

      iconfig=1
      ichan=1
      iconfigs(1)=iconfig
c Find the nFKSprocess for which we compute the Born-like contributions,
c ie. which is a Born+g real-emission process
      do nFKSprocess=1,fks_configs
         call fks_inc_chooser()
         if (is_aorg(i_fks)) exit
      enddo
      if (nFKSprocess.gt.fks_configs) then
c If there is no fks_configuration that has a gluon or photon as i_fks
c (this might happen in case of initial state leptons with
c include_lepton_initiated_processes=False) the Born and virtuals do not
c need to be included, and we can simply quit the process.
         return
      endif
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      call setfksfactor(.false.)
      symfactvirt = 1d0

      nfail = 0
      npointsChecked = 0

c Make sure that stability checks are always used by MadLoop, even for
c initialization
      CALL FORCE_STABILITY_CHECK(.TRUE.)
      CALL COLLIER_COMPUTE_UV_POLES(.TRUE.)
      CALL COLLIER_COMPUTE_IR_POLES(.TRUE.)

200   continue
          finite=0d0
          single=0d0
          double=0d0
          calculatedborn = .false.
          if (nincoming.eq.1) then
              call rambo(0, nexternal-nincoming-1, pmass(1), 
     1         pmass_rambo, prambo)
              p_born(0,1) = pmass(1)
              p_born(1,1) = 0d0
              p_born(2,1) = 0d0
              p_born(3,1) = 0d0
          elseif (nincoming.eq.2) then
              if (nexternal - nincoming - 1 .eq.1) then
                  ! deal with the case of only one particle in the final
                  ! state
                  p_born(0,1) = pmass(3)/2d0
                  p_born(1,1) = 0d0
                  p_born(2,1) = 0d0
                  p_born(3,1) = pmass(3)/2d0
                  if (pmass(1) > 0d0) 
     1               p_born(3,1) = dsqrt(pmass(3)**2/4d0 - pmass(1)**2)
                  p_born(0,2) = pmass(3)/2d0
                  p_born(1,2) = 0d0
                  p_born(2,2) = 0d0
                  p_born(3,2) = -pmass(3)/2d0
                  if (pmass(2) > 0d0) 
     1               p_born(3,2) = -dsqrt(pmass(3)**2/4d0 - pmass(1)**2)

                  prambo(0,1) = pmass(3)
                  prambo(1,1) = 0d0
                  prambo(2,1) = 0d0
                  prambo(3,1) = 0d0

              else
                    
                  call rambo(0, nexternal-nincoming-1, energy, 
     1             pmass_rambo, prambo)
                  p_born(0,1) = energy/2d0
                  p_born(1,1) = 0d0
                  p_born(2,1) = 0d0
                  p_born(3,1) = energy/2d0
                  if (pmass(1) > 0d0) 
     1               p_born(3,1) = dsqrt(energy**2/4d0 - pmass(1)**2)
                  p_born(0,2) = energy/2
                  p_born(1,2) = 0d0
                  p_born(2,2) = 0d0
                  p_born(3,2) = -energy/2d0
                  if (pmass(2) > 0d0) 
     1               p_born(3,2) = -dsqrt(energy**2/4d0 - pmass(1)**2)
              endif
          else
              write(*,*) 'INVALID NUMBER OF INCOMING PARTICLES', 
     1          nincoming
              stop
          endif

          do j = 0, 3
            do k = nincoming+1, nexternal-1
              p_born(j,k) = prambo(j,k-nincoming)
            enddo
          enddo

          CALL UPDATE_AS_PARAM()
          call sborn(p_born, born)
          ! extra initialisation calls: skip the first point
          ! as well as any other points which is used for initialization
          ! (according to the return code)
          call BinothLHA(p_born, born, virt_wgt)
          if (npointsChecked.eq.0) then
             if (mod(ret_code_ml,100)/10.eq.3 .or.
     &            mod(ret_code_ml,100)/10.eq.4) then
              ! this is to skip initialisation points
                write(*,*) 'INITIALIZATION POINT.'
                write(*,*)
     $               'RESULTS FROM INITIALIZATION POINTS WILL NOT '/
     $               /'BE USED FOR STATISTICS'
                goto 200
             endif
          endif
          write(*,*) 'MU_R    = ', ren_scale
          write(*,*) 'ALPHA_S = ', G**2/4d0/pi
C         Otherwise, perform the check
          npointsChecked = npointsChecked +1

          do j = 0, 3
            do k = 1, nexternal - 1
              p(j,k) = p_born(j,k)
            enddo
            p(j, nexternal) = 0d0
          enddo

          if ( tolerance.lt.0.0d0 ) then
               write(*,*) 'PASSED', tolerance
          else
              if (polecheck_passed) then
                write(*,*) 'PASSED', tolerance
              else
                write(*,*) 'FAILED', tolerance
                nfail=nfail+1
              endif
          endif
          write(*,*)

      if (npointsChecked.lt.npoints) goto 200 

          write(*,*) 'NUMBER OF POINTS PASSING THE CHECK', 
     1     npoints - nfail
          write(*,*) 'NUMBER OF POINTS FAILING THE CHECK', 
     1     nfail
          write(*,*) 'TOLERANCE ', tolerance

      return
      end



      SUBROUTINE RAMBO(LFLAG,N,ET,XM,P)
c------------------------------------------------------
c
c                       RAMBO
c
c    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)
c
c    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR
c    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING
c    THIS IS VERSION 1.0 -  WRITTEN BY R. KLEISS
c    (MODIFIED BY R. PITTAU)
c
c                INPUT                 OUTPUT
c
c    LFLAG= 0:   N, ET, XM             P, (DJ)
c    LFLAG= 1:   N, ET, XM, P          (DJ)
c
c    N  = NUMBER OF PARTICLES (>1, IN THIS VERSION <101)
c    ET = TOTAL CENTRE-OF-MASS ENERGY
c    XM = PARTICLE MASSES ( DIM=100 )
c    P  = PARTICLE MOMENTA ( DIM=(4,100) )
c    DJ = 1/(WEIGHT OF THE EVENT)
c
c------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      DIMENSION XM(100),P(0:3,100),Q(4,100),Z(100),R(4),
     .   B(3),P2(100),XM2(100),E(100),V(100),IWARN(5)
      SAVE ACC,ITMAX,IBEGIN,IWARN,Z,TWOPI,PO2LOG
      DATA ACC/1.D-14/,ITMAX/10/,IBEGIN/0/,IWARN/5*0/
C
C INITIALIZATION STEP: FACTORIALS FOR THE PHASE SPACE WEIGHT
      IF(IBEGIN.NE.0) GOTO 103
      IBEGIN=1
      TWOPI=8.*DATAN(1.D0)
      PO2LOG=LOG(TWOPI/4.)
      Z(2)=PO2LOG
      DO 101 K=3,100
  101 Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,100
  102 Z(K)=(Z(K)-LOG(DFLOAT(K-1)))
C
C CHECK ON THE NUMBER OF PARTICLES
  103 IF(N.GT.1.AND.N.LT.101) GOTO 104
      PRINT 1001,N
      STOP
C
C CHECK WHETHER TOTAL ENERGY IS SUFFICIENT; COUNT NONZERO MASSES
  104 XMT=0.
      NM=0
      DO 105 I=1,N
      IF(XM(I).NE.0.D0) NM=NM+1
  105 XMT=XMT+ABS(XM(I))
      IF(XMT.LE.ET) GOTO 201
      PRINT 1002,XMT,ET
      STOP

  201 CONTINUE 
      if (lflag.eq.1) then
        w0= exp((2.*N-4.)*LOG(ET)+Z(N))
        do j= 1,N
          v(j)= sqrt(p(1,j)**2+p(2,j)**2+p(3,j)**2)
        enddo

        a1= 0.d0
        a3= 0.d0
        a2= 1.d0
        do j= 1,N
          a1= a1+v(j)/ET
          a2= a2*v(j)/p(0,j)
          a3= a3+v(j)*v(j)/p(0,j)/ET
        enddo
        wm= a1**(2*N-3)*a2/a3
        dj= 1.d0/w0/wm
        return
      endif
C
C THE PARAMETER VALUES ARE NOW ACCEPTED
C
C GENERATE N MASSLESS MOMENTA IN INFINITE PHASE SPACE

      DO 202 I=1,N
      call rans(RAN1)
      call rans(RAN2)
      call rans(RAN3)
      call rans(RAN4)
      C=2.*RAN1-1.
      S=SQRT(1.-C*C)
      F=TWOPI*RAN2
      Q(4,I)=-LOG(RAN3*RAN4)
      Q(3,I)=Q(4,I)*C
      Q(2,I)=Q(4,I)*S*COS(F)
  202 Q(1,I)=Q(4,I)*S*SIN(F)
C
C CALCULATE THE PARAMETERS OF THE CONFORMAL TRANSFORMATION
      DO 203 I=1,4
  203 R(I)=0.
      DO 204 I=1,N
      DO 204 K=1,4
  204 R(K)=R(K)+Q(K,I)
      RMAS=SQRT(R(4)**2-R(3)**2-R(2)**2-R(1)**2)
      DO 205 K=1,3
  205 B(K)=-R(K)/RMAS
      G=R(4)/RMAS
      A=1./(1.+G)
      X=ET/RMAS
C
C TRANSFORM THE Q'S CONFORMALLY INTO THE P'S
      DO 207 I=1,N
      BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
      DO 206 K=1,3
  206 P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
  207 P(0,I)=X*(G*Q(4,I)+BQ)
C
C CALCULATE WEIGHT AND POSSIBLE WARNINGS
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
      IF(WT.GE.-180.D0) GOTO 208
      IF(IWARN(1).LE.5) PRINT 1004,WT
      IWARN(1)=IWARN(1)+1
  208 IF(WT.LE. 174.D0) GOTO 209
      IF(IWARN(2).LE.5) PRINT 1005,WT
      IWARN(2)=IWARN(2)+1
C
C RETURN FOR WEIGHTED MASSLESS MOMENTA
  209 IF(NM.NE.0) GOTO 210
      WT=EXP(WT)
      DJ= 1.d0/WT
      RETURN
C
C MASSIVE PARTICLES: RESCALE THE MOMENTA BY A FACTOR X
  210 XMAX=SQRT(1.-(XMT/ET)**2)
      DO 301 I=1,N
      XM2(I)=XM(I)**2
  301 P2(I)=P(0,I)**2
      ITER=0
      X=XMAX
      ACCU=ET*ACC
  302 F0=-ET
      G0=0.
      X2=X*X
      DO 303 I=1,N
      E(I)=SQRT(XM2(I)+X2*P2(I))
      F0=F0+E(I)
  303 G0=G0+P2(I)/E(I)
      IF(ABS(F0).LE.ACCU) GOTO 305
      ITER=ITER+1
      IF(ITER.LE.ITMAX) GOTO 304
      PRINT 1006,ITMAX
      GOTO 305
  304 X=X-F0/(X*G0)
      GOTO 302
  305 DO 307 I=1,N
      V(I)=X*P(0,I)
      DO 306 K=1,3
  306 P(K,I)=X*P(K,I)
  307 P(0,I)=E(I)
C
C CALCULATE THE MASS-EFFECT WEIGHT FACTOR
      WT2=1.
      WT3=0.
      DO 308 I=1,N
      WT2=WT2*V(I)/E(I)
  308 WT3=WT3+V(I)**2/E(I)
      WTM=(2.*N-3.)*LOG(X)+LOG(WT2/WT3*ET)
C
C RETURN FOR  WEIGHTED MASSIVE MOMENTA
      WT=WT+WTM
      IF(WT.GE.-180.D0) GOTO 309
      IF(IWARN(3).LE.5) PRINT 1004,WT
      IWARN(3)=IWARN(3)+1
  309 IF(WT.LE. 174.D0) GOTO 310
      IF(IWARN(4).LE.5) PRINT 1005,WT
      IWARN(4)=IWARN(4)+1
  310 WT=EXP(WT)
      DJ= 1.d0/WT
      RETURN
C
 1001 FORMAT(' RAMBO FAILS: # OF PARTICLES =',I5,' IS NOT ALLOWED')
 1002 FORMAT(' RAMBO FAILS: TOTAL MASS =',D15.6,' IS NOT',
     . ' SMALLER THAN TOTAL ENERGY =',D15.6)
 1004 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY UNDERFLOW')
 1005 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY  OVERFLOW')
 1006 FORMAT(' RAMBO WARNS:',I3,' ITERATIONS DID NOT GIVE THE',
     . ' DESIRED ACCURACY =',D15.6)
      END


      subroutine rans(rand)
c     Just a wrapper to ran2      
      implicit none
      double precision rand, ran2
      rand = ran2()
      return 
      end


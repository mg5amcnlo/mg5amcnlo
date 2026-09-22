C routines to perform moemntum reshuffling based on those
C employed in MadSTR (1907.04898)
C One particle at the time is considered, either recoiling on
C all other FS particles, or on the initial state ones
      
      
      REAL*8 function lambda_tr(x,y,z)
C-----triangular function
      implicit none
      real*8 x,y,z
      lambda_tr=x**2+y**2+z**2-2d0*x*y-2d0*x*z-2d0*y*z
      return
      end


      double precision function lambda2(a,b,c)
      implicit none
      double precision a,b,c
      if (a.le.0d0 .or. abs(b+c).gt.abs(a) .or. abs(b-c).gt.abs(a)) then
         write (*,*) 'Error #1 in lambda2: inputs not consistent',a,b,c
         stop 1
      endif
      lambda2=sqrt(1d0-(b+c)**2/a**2)*sqrt(1d0-(b-c)**2/a**2)
      return
      end


      subroutine reshuffle_momenta_old(p,q,iresh,pdg_old,pdg_new,pass)
C A wrapper which, based on ihowresh, calls the subroutine for 
C the initial- or final-state reshuffling
      implicit none 
      include 'nexternal.inc'
C-----Arguments      
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh, pdg_old, pdg_new
      double precision mass_old, mass_new
      logical pass
C-----Local
      integer ihowresh
      parameter(ihowresh=1) ! 1->initial, 2->final

      if (iresh.gt.nincoming) then
          ! for the reshuffling of a final-state particle, 
          ! two options exist: recoil on all other FS particles,
          ! or recoil on the initial state
          if (ihowresh.eq.1) then
            call reshuffle_initial(p,q,iresh,pdg_old,pdg_new,pass)
          else if (ihowresh.eq.2) then
            call reshuffle_final(p,q,iresh,pdg_old,pdg_new,pass)
          else
            write(*,*) 'ERROR: reshuffle momenta, wrong option', ihowresh
            stop 1
          endif
      else
          ! for the reshuffling of an initial statem momentum,
          ! the only option is to recoil on all FS particles
          call reshuffle_initial_state(p,q,iresh,pdg_old,pdg_new,pass)
      endif

      return
      end


      subroutine reshuffle_momenta(p,q,iresh,pdg_old,pdg_new,pass)
C A wrapper which, based on ihowresh, calls the subroutine for 
C the initial- or final-state reshuffling
      implicit none 
      include 'nexternal.inc'
C-----Arguments      
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh(2), pdg_old(2), pdg_new(2)
      double precision mass_old, mass_new
      logical pass
C-----Local
      integer i
      ! the reshuffling strategy
      integer ihowresh
      parameter(ihowresh=1) ! 1-> recoil initial, 2-> recoil final
      ! this is whether to keep the invariant mass constant if two
      ! FS particles are reshuffled
      logical reshuffle_two
      parameter (reshuffle_two=.true.)

      ! consistency check
      do i = 1, 2 
        if (iresh(i).eq.0.or.pdg_old(i).eq.0.or.pdg_new(i).eq.0) then
          if (iresh(i).ne.0.or.pdg_old(i).ne.0.or.pdg_new(i).ne.0) then
            write(*,*) 'ERROR in reshuffling, inconsistent values',
     #         i, iresh, pdg_old, pdg_new
            stop 1
          endif
        endif
      enddo


      if (reshuffle_two.and.iresh(1).gt.nincoming.and.iresh(2).gt.nincoming) then
        call reshuffle_final_two(p,q,iresh,pdg_old,pdg_new,pass)
      else
        do i = 2, 1, -1  ! if two particles have to be reshuffled, start
                         ! from the second one
          ! skip the case where iresh/pdg_old/pdg_new are zero
          ! (just check one, consistency has been checked above)
          if (iresh(i).eq.0) continue

          if (iresh(i).gt.nincoming) then
            ! for the reshuffling of a final-state particle, 
            ! two options exist: recoil on all other FS particles,
            ! or recoil on the initial state
            if (ihowresh.eq.1) then
              call reshuffle_initial(p,q,iresh(i),pdg_old(i),pdg_new(i),pass)
            else if (ihowresh.eq.2) then
              call reshuffle_final(p,q,iresh(i),pdg_old(i),pdg_new(i),pass)
            else
              write(*,*) 'ERROR: reshuffle momenta, wrong option', ihowresh
              stop 1
            endif
          else
            ! for the reshuffling of an initial statem momentum,
            ! the only option is to recoil on all FS particles
            call reshuffle_initial_state(p,q,iresh(i),pdg_old(i),pdg_new(i),pass)
          endif
          ! exit and return if pass = False
          if (.not.pass) return
        enddo
      endif

      return
      end


      subroutine reshuffle_final_two(p,q,iresh,pdg_old,pdg_new,pass)
************************************************************************
*     Authors: Marco Zaro                                              *
*     Given momenta p(nu,nexternal-1) produce q(nu,external-1).        * 
*     Only in this function, iresh, pdg_old, pdg_new are 2-component   *
*     arrays, with the informations on the two particles whose mass    *
*     Is going to be changed.                                          *
*     Reshuffling is perfomed conserving the invariant mass and total  *
*     momentum of the two particles. Also the angles in the combined   *
*     rest frame are preserved. The other momenta are left unchanged   *
************************************************************************
      implicit none 
      include 'nexternal.inc'
C-----Arguments      
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh(2), pdg_old(2), pdg_new(2)
      double precision mass_old(2), mass_new(2)
      logical pass
C-----Local variables
      integer i,j
      double precision invm2, ptot(0:3)
      double precision pcom(0:3,2), qcom(0:3,2)
      double precision pspac2, qspac2
C-----Functions
      double precision dot, threedot, lambda_tr, get_mass_from_id
      external dot, threedot, lambda_tr, get_mass_from_id

      pass = .true.
      do i = 1, 2
        mass_old(i) = get_mass_from_id(pdg_old(i))
        mass_new(i) = get_mass_from_id(pdg_new(i))
      enddo

      ! do nothing if masses do not change
      if (mass_old(1).eq.mass_new(1).and.mass_old(2).eq.mass_new(2)) then
        q(:,:) = p(:,:)
        return
      endif

      ! copy the 'ohter' momenta
      do j = 1, nexternal-1
        if (j.eq.iresh(1).or.j.eq.iresh(2)) continue
        q(:,j) = p(:,j)
      enddo

      ! the total momentum of the two particles
      ptot(:) = p(:,iresh(1)) + p(:,iresh(2))
      ! the invariant mass
      invm2 = dot(ptot,ptot)
      ! check that the reshuffling can be done
      if (sqrt(invm2).lt.mass_new(1)+mass_new(2)) then
        pass = .false.
        return
      endif

      ! go in the rest frame of the total momentum and 
      ! compute the momenta of the two particles in that frame
      do i = 1,2
        call invboostx(p(0, iresh(i)), ptot, pcom(0,i))
      enddo
      ! the modulus of the spatial momenta for the old momenta
      pspac2 = threedot(pcom(0,1),pcom(0,1))
      ! the modulus of the spatial momenta for the new momenta
      qspac2 = lambda_tr(invm2,mass_new(1)**2,mass_new(2)**2) / 4d0 / invm2
      ! now compute the new momenta in the com frame, keeping 
      ! the same spatial direction as p, and finally boost them back in
      ! the original frame
      do i = 1,2
        do j = 1,3
          qcom(j,i) = pcom(j,i) * dsqrt(qspac2/pspac2)
        enddo
        qcom(0,i) = dsqrt(mass_new(i)**2 + qspac2)
        call boostx(qcom(0,i), ptot, q(0,iresh(i)))
      enddo

      ! check the momenta before returning
      call check_reshuffled_momenta_two(p, q, iresh, mass_new)

      return
      end
       




      subroutine reshuffle_initial_state(p,q,iresh,pdg_old,pdg_new,pass)
************************************************************************
*     Authors: Marco Zaro                                              *
*     Given momenta p(nu,nexternal-1) produce q(nu,external-1) with    * 
*     the mass of particle iresh set according to pdg_new.             *
*     In this case, iresh is in the initial stete.                     *
*     The spatial components of the IS momenta are set such that       *
*     No change is needed for the final state ones                     *
************************************************************************
      implicit none 
      include 'nexternal.inc'
C-----Arguments      
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh, pdg_old, pdg_new
      double precision mass_old, mass_new
      logical pass
C-----Local
      integer i, j, nu
      double precision preco(0:3), qreco(0:3), pcom(0:3)
      double precision qinboost(0:3,nincoming)
      double precision shat, newmom, m2_other
      double precision msq_reco, totmass
      double precision a, b
      double precision dot, threedot, sumdot, get_mass_from_id
      external dot, threedot, sumdot, get_mass_from_id
      double precision rescale_init
C--------------- 
C     BEGIN CODE                                                                   
C---------------    

      pass = .true.
      mass_old = get_mass_from_id(pdg_old)
      ! check that we start from a massless particle
      !  (should always be the case)
      if (mass_old.ne.0d0) then
          write(*,*)'ERROR reshuffle initial state with mass_old!=0', mass_old
      endif
      ! check that we are in the partonic com
      if (abs(p(3,1)+p(3,2))/(abs(p(3,1))+abs(p(3,2))).gt.1d-6) then
          write(*,*)'ERROR reshuffle initial state, no com',
     $     p(:,1), p(:,2)
      endif
      mass_new = get_mass_from_id(pdg_new)

      ! do nothing if masses do not change
      if (mass_old.eq.mass_new) then
        q(:,:) = p(:,:)
        return
      endif

      ! under these assumptions, we go in the frame where the
      ! reshuffled particle with mas = mass_new is on shell
      ! The com energy is always conserved
      shat = sumdot(p(0,1), p(0,2), 1d0)
      do i = 1,nincoming
        if (i.eq.iresh) then
            qinboost(0,i) = mass_new
            qinboost(1:3,i) = 0d0 
        else
            m2_other = dot(p(0,i),p(0,i))
            qinboost(0,i) = (shat - m2_other - mass_new**2) / 2d0 / mass_new
            qinboost(1:2,i) = 0d0
            ! check that it is physical
            if (qinboost(0,i)**2 - m2_other.lt.0d0.or.qinboost(0,i).lt.0d0) then
                pass = .false.
                return
            endif
            qinboost(3,i) = sign(dsqrt(qinboost(0,i)**2 - m2_other), p(3,i))
        endif
      enddo

      ! now boost the momenta back to the com frame
      pcom(:) = qinboost(:,1) + qinboost(:,2)

      do i=1, nincoming
        call invboostx(qinboost(0,i), pcom, q(0,i))
      enddo

      do i = nincoming+1, nexternal-1
        q(:,i) = p(:,i)
      enddo

      ! check the momenta before returning
      call check_reshuffled_momenta(p, q, iresh, mass_new)

      return     
      end




      subroutine reshuffle_final(p,q,iresh,pdg_old,pdg_new,pass)
************************************************************************
*     Authors: Marco Zaro                                              *
*     Given momenta p(nu,nexternal-1) produce q(nu,external-1) with    * 
*     the mass of particle iresh set according to pdg_new.             *
*     The reshuffling recoils against the other final-state particles. *
*     If the reshuffling is not possible, pass is set to False         *
************************************************************************
      implicit none 
      include 'nexternal.inc'
C-----Arguments      
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh, pdg_old, pdg_new
      double precision mass_old, mass_new
      logical pass
C-----Local
      integer i, j, nu
      double precision preco(0:3), qreco(0:3), ptmp(0:3)
      double precision shat
      double precision msq_reco, totmass
      double precision a, b
      double precision dot, threedot, get_mass_from_id
      external dot, threedot, get_mass_from_id
      double precision rescale_init
C--------------- 
C     BEGIN CODE                                                                   
C---------------    

      pass = .true.
      mass_old = get_mass_from_id(pdg_old)
      mass_new = get_mass_from_id(pdg_new)

      ! do nothing if masses do not change
      if (mass_old.eq.mass_new) then
        q(:,:) = p(:,:)
        return
      endif

C compute the total mass of the FS particles
      totmass = 0d0
      do i = nincoming+1, nexternal-1
        totmass = totmass + dsqrt(max(0d0, dot(p(0,i),p(0,i))))
      enddo

C the center of mass energy
      shat = 2d0 * dot(p(0,1),p(0,2))

C check if the reshuffling is feasible
      if (sqrt(shat).lt.totmass+mass_new-mass_old) then
          pass = .false.
          return
      endif

c reconstruct the recoil system (all FS particles which are not iresh)
      do nu=0,3
        preco(nu)=0d0
      enddo
      do i=nincoming+1,nexternal-1
        if (i.eq.iresh) cycle
        do nu=0,3
          preco(nu)=preco(nu)+p(nu,i)
        enddo
      enddo

      msq_reco=dot(preco,preco)

C the reshuffled momenta, q(iresh) and qreco, will have energy components
C wchic correspond to q(iresh) having the new mass and qreco keeping
C its invariant mass
      q(0,iresh) = dsqrt(shat)/2d0*(1d0+(mass_new**2-msq_reco)/shat)
      qreco(0) = dsqrt(shat)/2d0*(1d0-(mass_new**2-msq_reco)/shat)
C the other components have the same direction as q(iresh), preco, and must
C satisfy the mass-shell conditions q(iresh)^2 = mass_new^2, qreco^2=m_reco^2
      do nu=1,3
        q(nu,iresh) = p(nu,iresh)/dsqrt(threedot(p(0,iresh),p(0,iresh)))
     $                *dsqrt(q(0,iresh)*q(0,iresh)-mass_new**2)
        qreco(nu) = preco(nu)/dsqrt(threedot(preco,preco))*dsqrt(qreco(0)*qreco(0)-msq_reco)
      enddo

C *** other recoiling particles
C boost them to the preco rest frame and then back to
C the lab frame using qreco
      do i=1, nexternal-1
        if (i.eq.iresh) cycle
        call invboostx(p(0,i), preco, ptmp)
        call boostx(ptmp, qreco, q(0,i))
      enddo

C *** finally the initial state particles, do nothing
      do i=1, nincoming
        do nu=0,3
          q(nu,i)=p(nu,i)
        enddo
      enddo

      ! check the momenta before returning
      call check_reshuffled_momenta(p, q, iresh, mass_new)

      return     
      end


      subroutine reshuffle_initial(p,q,iresh,pdg_old,pdg_new,pass)
************************************************************************
*     Authors: Marco Zaro                                              *
*     Given momenta p(nu,nexternal-1) produce q(nu,external-1) with    * 
*     the mass of particle iresh set according to pdg_new.             *
*     The reshuffling is performed by changing the energy of the       *
*     initial state particles.                                         *
*     If the reshuffling is not possible, pass is set to False         *
************************************************************************
      implicit none 
      include 'nexternal.inc'
C-----Arguments      
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh, pdg_old, pdg_new
      double precision mass_old, mass_new
      logical pass
C-----Local
      integer i, j, nu
      double precision preco(0:3), qreco(0:3), ptmp(0:3)
      double precision shat
      double precision msq_reco
      double precision a, b, etot, ztot
      double precision dot, threedot, get_mass_from_id
      external dot, threedot, get_mass_from_id
      double precision rescale_init
 
C--------------- 
C     BEGIN CODE                                                                    
C---------------    
      pass = .true.
      mass_old = get_mass_from_id(pdg_old)
      mass_new = get_mass_from_id(pdg_new)

      ! do nothing if masses do not change
      if (mass_old.eq.mass_new) then
        q(:,:) = p(:,:)
        return
      endif

      etot = 0d0
      ztot = 0d0

C-----Set the energy of the particle to be reshuffled according
C     to the new mass
      do nu=1,3
        q(nu,iresh) = p(nu,iresh)
      enddo
      q(0,iresh)=dsqrt(mass_new**2+threedot(q(0,iresh),q(0,iresh)))

C-----Keep the rest of the FS momenta without any changes
      do j=nincoming+1,nexternal-1
        do nu=0,3
          if(j.ne.iresh)then
            q(nu,j) = p(nu,j)
          endif
        enddo
        etot = etot + q(0,j)
        ztot = ztot + q(3,j)
      enddo

C initial state momenta: one knows the sum and the difference of
C energies (sum of z components of FS momenta)
      q(0,1) = (etot + ztot)/2d0
      q(1,1) = 0d0
      q(2,1) = 0d0
      q(3,1) = dsign(q(0,1), p(3,1)) 

      q(0,2) = (etot - ztot)/2d0
      q(1,2) = 0d0
      q(2,2) = 0d0
      q(3,2) = dsign(q(0,2), p(3,2)) 

      ! check the momenta before returning
      call check_reshuffled_momenta(p, q, iresh, mass_new)

      return     
      end


      subroutine check_reshuffled_momenta(p, q, iresh, mass_new)
      ! performs some consistency checks on the momenta
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh
      double precision mass_new
      double precision a, b
      integer i, j
      double precision dot

      if (nincoming.ne.2) then
        write(*,*) 'ERROR IN OS_CHECK_MOMENTA:, nincoming != 2 not'//
     $   ' implemented', nincoming
        stop
      endif
      
      do i = 1, nexternal-1
C--------mass shell conditions
        if (i.ne.iresh) then
          if (dabs(dot(q(0,i),q(0,i))-dot(p(0,i),p(0,i)))
     $     .gt. 1d-3 * max(dot(p(0,i),p(0,i)), p(0,i)**2)) then
            write(*,*) 'ERROR IN CHECK_RESHUFFLED_MOMENTA: NOT ON SHELL', i
            write(*,*) 'MSQ before', dot(p(0,i),p(0,i))
            write(*,*) 'MSQ after ', dot(q(0,i),q(0,i))
            stop
          endif
        else
          if (dabs(dot(q(0,i),q(0,i))-mass_new**2)
     $     .gt. 1d-3 * max(dot(q(0,i),q(0,i)), q(0,i)**2)) then
            write(*,*) 'ERROR IN CHECK_RESHUFFLED_MOMENTA: NOT ON SHELL', i
            write(*,*) 'MSQ (iresh)', mass_new**2
            write(*,*) 'MSQ after ', dot(q(0,i),q(0,i))
            stop
          endif
        endif
      enddo

C--------momentum conservation
      do i = 0,3
        a = 0d0
        b = 0d0
        do j = 1, nexternal-1
          b = max(b, dabs(q(i,j)))
          if (j.le.nincoming) then
            a = a - q(i,j)
          else
            a = a + q(i,j)
          endif
        enddo
        if (dabs(a)/b.gt.1d-6) then
          write(*,*) 'ERROR IN CHECK_RESHUFFLED_MOMENTA: MOM. CONS',
     $      i, dabs(a), b
          do j = 1, nexternal-1
            write(*,*) q(0,j), q(1,j), q(2,j), q(3,j), dsqrt(dot(q(0,j), q(0,j)))
          enddo
          stop
        endif
      enddo

      return
      end


      subroutine check_reshuffled_momenta_two(p, q, iresh, mass_new)
      ! performs some consistency checks on the momenta
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal-1), q(0:3,nexternal-1)
      integer iresh(1:2)
      double precision mass_new(1:2)
      double precision a, b
      integer i, j
      double precision dot
      integer filoc_int

      if (nincoming.ne.2) then
        write(*,*) 'ERROR IN OS_CHECK_MOMENTA:, nincoming != 2 not'//
     $   ' implemented', nincoming
        stop
      endif

      do j =1,2
      do i = 1, nexternal-1
C--------mass shell conditions
        if (filoc_int(iresh,size(iresh),i) .eq. 0) then
          if (dabs(dot(q(0,i),q(0,i))-dot(p(0,i),p(0,i)))
     $     .gt. 1d-3 * max(dot(p(0,i),p(0,i)), p(0,i)**2)) then
            write(*,*) 'ERROR IN CHECK_RESHUFFLED_MOMENTA: NOT ON SHELL', i
            write(*,*) 'MSQ before', dot(p(0,i),p(0,i))
            write(*,*) 'MSQ after ', dot(q(0,i),q(0,i))
            stop
          endif
        else
          if (dabs(dot(q(0,i),q(0,i))-mass_new(filoc_int(iresh,size(iresh),i))**2)
     $     .gt. 1d-3 * max(dot(q(0,i),q(0,i)), q(0,i)**2)) then
            write(*,*) 'ERROR IN CHECK_RESHUFFLED_MOMENTA: NOT ON SHELL', i
            write(*,*) 'MSQ (iresh)', mass_new(j)**2
            write(*,*) 'MSQ after ', dot(q(0,i),q(0,i))
            stop
          endif
        endif
      enddo
      enddo

C--------momentum conservation
      do i = 0,3
        a = 0d0
        b = 0d0
        do j = 1, nexternal-1
          b = max(b, dabs(q(i,j)))
          if (j.le.nincoming) then
            a = a - q(i,j)
          else
            a = a + q(i,j)
          endif
        enddo
        if (dabs(a)/b.gt.1d-6) then
          write(*,*) 'ERROR IN CHECK_RESHUFFLED_MOMENTA: MOM. CONS',
     $      i, dabs(a), b
          do j = 1, nexternal-1
            write(*,*) q(0,j), q(1,j), q(2,j), q(3,j), dsqrt(dot(q(0,j), q(0,j)))
          enddo
          stop
        endif
      enddo

      return
      end


      integer function filoc_int(array,n,val)
      implicit none
      integer :: n,val,i,res
      integer, dimension(n) :: array
      res = 0
      do i=1,n
         if (array(i) .eq. val) then
            res = i
         endif
      enddo
      filoc_int = res
      end function filoc_int


      





      subroutine invboostx(p,q , pboost)
c
c This subroutine performs the Lorentz boost of a four-momentum.  The
c momenta p and q are assumed to be given in the same frame.pboost is
c the momentum p boosted to the q rest frame.  q must be a
c timelike momentum.
c it is the inverse of boostx
c
c input:
c       real    p(0:3)         : four-momentum p in the same frame as q
c       real    q(0:3)         : four-momentum q 
c
c output:
c       real    pboost(0:3)    : four-momentum p in the boosted frame
c
      implicit none
      double precision p(0:3),q(0:3),pboost(0:3),pq,qq,m,lf

      double precision rZero
      parameter( rZero = 0.0d0 )

c#ifdef HELAS_CHECK
c      integer stdo
c      parameter( stdo = 6 )
c      double precision pp
c#endif
c
      qq = q(1)**2+q(2)**2+q(3)**2

c#ifdef HELAS_CHECK
c      if (abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in boostx is zero momentum'
c      endif
c      if (abs(q(0))+qq.eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx is zero momentum'
c      endif
c      if (p(0).le.rZero) then
c         write(stdo,*)
c     &        ' helas-warn  : p(0:3) in boostx has not positive energy'
c         write(stdo,*)
c     &        '             : p(0) = ',p(0)
c      endif
c      if (q(0).le.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx has not positive energy'
c         write(stdo,*)
c     &        '             : q(0) = ',q(0)
c      endif
c      pp=p(0)**2-p(1)**2-p(2)**2-p(3)**2
c      if (pp.lt.rZero) then
c         write(stdo,*)
c     &        ' helas-warn  : p(0:3) in boostx is spacelike'
c         write(stdo,*)
c     &        '             : p**2 = ',pp
c      endif
c      if (q(0)**2-qq.le.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx is not timelike'
c         write(stdo,*)
c     &        '             : q**2 = ',q(0)**2-qq
c      endif
c      if (qq.eq.rZero) then
c         write(stdo,*)
c     &   ' helas-warn  : q(0:3) in boostx has zero spacial components'
c      endif
c#endif

      if ( qq.ne.rZero ) then
         pq = p(1)*q(1)+p(2)*q(2)+p(3)*q(3)
         m = dsqrt(max(q(0)**2-qq,1d-99))
         lf = (-(q(0)-m)*pq/qq+p(0))/m
         pboost(0) = (p(0)*q(0)-pq)/m
         pboost(1) =  p(1)-q(1)*lf
         pboost(2) =  p(2)-q(2)*lf
         pboost(3) =  p(3)-q(3)*lf
      else
         pboost(0) = p(0)
         pboost(1) = p(1)
         pboost(2) = p(2)
         pboost(3) = p(3)
      endif
c
      return
      end


      subroutine write_momenta(p)
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal)
      integer i
      do i = 1, nexternal
        write(*,*) i, p(0,i), p(1,i), p(2,i), p(3,i)
      enddo
      return
      end

      subroutine write_momenta4(p)
      implicit none
      include 'nexternal.inc'
      double precision p(0:4,nexternal)
      integer i
      do i = 1, nexternal
        write(*,*) i, p(0,i), p(1,i), p(2,i), p(3,i), p(4,i)
      enddo
      return
      end

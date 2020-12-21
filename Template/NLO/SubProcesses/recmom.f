      subroutine recombine_momenta(R, etaph, reco_l, reco_q, p_in, pdg_in, p_out, pdg_out)
      implicit none
      ! recombine photons with the closest fermion if the distance is
      ! less than R and if the rapidity of photons is < etaph (etaph < 0
      ! means no cut). Output a new set of momenta and pdgs corresponding
      ! to the recombined particles. If recombination occurs the photon
      ! disappears from the output particles
      ! arguments
      include 'nexternal.inc'
      double precision R, etaph, p_in(0:4,nexternal), p_out(0:4,nexternal)
      logical reco_l, reco_q
      integer pdg_in(nexternal), pdg_out(nexternal)
      ! local variables
      integer nq, nl
      integer id_ph
      parameter (id_ph=22)
      integer n_ph, i_ph
      integer i,j
      integer ifreco
      double precision dreco, dthis
      integer skip
      logical is_light_charged_fermion
      double precision R2, eta
      ! 
      integer times_reco
      common/to_times_reco/ times_reco
      ! reset everything
      do j=1,nexternal
        pdg_out(j)=0
        do i=0,4
          p_out(i,j)=0d0
        enddo
      enddo

      ! check if we want to recombine with leptons
      if (reco_l) then
          nl = 3
      else 
          nl = 0
      endif

      ! check if we want to recombine with quarks
      if (reco_q) then
          nq = 5
      else 
          nq = 0
      endif

      ! count the photons
      n_ph=0
      do i=nincoming+1, nexternal
        if (pdg_in(i).eq.id_ph.and.
     $   (abs(eta(p_in(0,i))).lt.etaph.or.etaph.lt.0d0)) then
            n_ph=n_ph+1
            i_ph=i
        endif
      enddo
      if (n_ph.eq.0 .or. (nl.eq.0 .and. nq.eq.0)) then
        ! do nothing
        do j=1,nexternal
          pdg_out(j)=pdg_in(j)
          do i=0,4
            p_out(i,j)=p_in(i,j)
          enddo
        enddo
        return
      elseif (n_ph.eq.1) then
        ! do nothing for initial states
        do j=1,nincoming
          pdg_out(j)=pdg_in(j)
          do i=0,4
            p_out(i,j)=p_in(i,j)
          enddo
        enddo
        ! find the closest fermion to the photon
        ifreco=0
        dreco=R
        if (i_ph.gt.0) then
          do i = nincoming+1, nexternal
            if (is_light_charged_fermion(pdg_in(i),nq,nl)) then
              dthis=dsqrt(R2(p_in(0,i_ph),p_in(0,i)))
              if (dthis.le.dreco) then
                dreco=dthis
                ifreco=i
              endif
            endif
          enddo
        endif
        if (ifreco.eq.0) then
        ! do nothing also for final states
          do j=nincoming+1,nexternal
            pdg_out(j)=pdg_in(j)
            do i=0,4
              p_out(i,j)=p_in(i,j)
            enddo
          enddo
        else
          times_reco=times_reco+1
          skip=0
          do j=nincoming+1,nexternal
            if (j.ne.i_ph.and.j.ne.ifreco) then
              pdg_out(j-skip)=pdg_in(j)
              do i=0,4
                p_out(i,j-skip)=p_in(i,j)
              enddo
            elseif (j.eq.ifreco) then
              pdg_out(j-skip)=pdg_in(j)
              do i=0,3
                p_out(i,j-skip)=p_in(i,j)+p_in(i,i_ph)
              enddo
              p_out(4,j-skip)=p_in(4,j)
            elseif (j.eq.i_ph) then
              skip=skip+1
            endif
          enddo
        endif
      else
        write(*,*) 'ERROR, too many photons', n_ph
        stop 1
      endif

      return 
      end


      logical function is_light_charged_fermion(id, nf, nl)
      implicit none
      integer id, nf, nl
      if (abs(id).le.nf) then
          is_light_charged_fermion = .true.
      elseif ((abs(id).eq.11.and.nl.ge.1).or.
     $        (abs(id).eq.13.and.nl.ge.2).or.
     $        (abs(id).eq.15.and.nl.ge.3)) then
          is_light_charged_fermion = .true.
      else
          is_light_charged_fermion = .false.
      endif
      return
      end

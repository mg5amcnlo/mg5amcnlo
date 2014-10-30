      subroutine configs_and_props_inc_chooser()
c For a given nFKSprocess, it fills the c_configs_inc common block with
c the configs.inc information (i.e. IFOREST(), SPROP(), TPRID() and
c MAPCONFIG())
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      double precision ZERO
      parameter (ZERO=0d0)
      include 'maxparticles.inc'
      include 'ngraphs.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iforest(2,-max_branch:-1,n_max_cg)
      integer sprop(-max_branch:-1,n_max_cg)
      integer tprid(-max_branch:-1,n_max_cg)
      integer mapconfig(0:n_max_cg)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig
      double precision prmass(-max_branch:nexternal,n_max_cg)
      double precision prwidth(-max_branch:-1,n_max_cg)
      integer prow(-max_branch:-1,n_max_cg)
      common/c_props_inc/prmass,prwidth,prow
      double precision pmass(nexternal)
      logical firsttime
      data firsttime /.true./
      include 'configs_and_props_decl.inc'
      save mapconfig_d, iforest_d, sprop_d, tprid_d, pmass_d, pwidth_d
     $ , pow_d
      include "pmass.inc"
c     
      if (max_branch_used.gt.max_branch) then
         write (*,*) 'ERROR in configs_and_props_inc_chooser:'/
     $        /' increase max_branch',max_branch,max_branch_used
         stop
      endif
      if (lmaxconfigs_used.gt.n_max_cg) then
         write (*,*) 'ERROR in configs_and_propsinc_chooser:'/
     $        /' increase n_max_cg' ,n_max_cg,lmaxconfigs_used
         stop
      endif

C the configurations and propagators infos are read at the first
C evaluation
      if (firsttime) then
        call read_configs_and_props_info(mapconfig_d,iforest_d,sprop_d,
     1                                   tprid_d,pmass_d,pwidth_d,pow_d)
        firsttime = .false.
      endif

c
c Fill the arrays of the c_configs_inc and c_props_inc common
c blocks. Some of the information might not be available in the
c configs_and_props_info.inc include file, but there is no easy way of skipping
c it. This will simply fill the common block with some bogus
c information.
      do i=0,MAPCONFIG_D(nFKSprocess,0)
         mapconfig(i)=MAPCONFIG_D(nFKSprocess,i)
         if (i.ne.0) then
            do j=-max_branch_used,-1
               do k=1,2
                  iforest(k,j,i)=IFOREST_D(nFKSprocess,k,j,i)
               enddo
               sprop(j,i)=SPROP_D(nFKSprocess,j,i)
               tprid(j,i)=TPRID_D(nFKSprocess,j,i)
               prmass(j,i)=PMASS_D(nFKSprocess,j,i)
               prwidth(j,i)=PWIDTH_D(nFKSprocess,j,i)
               prow(j,i)=POW_D(nFKSprocess,j,i)
            enddo
c for the mass, also fill for the external masses
            prmass(0,i)=0d0
            do j=1,nexternal
               prmass(j,i)=pmass(j)
            enddo
         endif
      enddo
c
      return
      end


      subroutine read_configs_and_props_info(mapconfig_d,iforest_d,sprop_d,
     1                                   tprid_d,pmass_d,pwidth_d,pow_d)
C read the various information from the configs_and_props_info.dat file
      implicit none
      integer i,j,k
      integer ndau, idau, dau, id
      character *200 buff
      double precision get_mass_from_id, get_width_from_id
      include 'configs_and_props_decl.inc'

      open(unit=78, file='configs_and_props_info.dat', status='old')
      do while (.true.)
        read(78,'(a)',end=999) buff
        if (buff(:1).eq.'#') cycle
        if (buff(:1).eq.'C') then
        ! mapconfig
        ! C  i   j   k -> MAPCONFIG_D(i,j)=k
          read(buff(2:),*) i,j,k
          mapconfig_d(i,j) = k
        else if (buff(:1).eq.'F') then
        ! iforest
        ! after the first line there are as many lines
        ! as the daughters
        ! F  i   j   k  ndau
        ! D dau_1
        ! D ...
        ! D dau_ndau        -> IFORREST_D(i,idau,i,k)=dau_idau
          read(buff(2:),*) i,j,k,ndau
          do idau=1,ndau
            read(78,'(a)') buff
            if (buff(:1).ne.'D') then
              write(*,*) 'ERROR #1 in read_configs_and_props_info',
     1                    i,j,k,ndau,buff
              stop 
            endif
            read(buff(2:),*) dau
            iforest_d(i,idau,j,k) = dau
          enddo
        else if (buff(:1).eq.'S') then
        ! sprop
        ! S  i   j   k  id -> SPROP_D(i,j,k)=id
          read(buff(2:),*) i,j,k,id
          sprop_d(i,j,k) = id
        else if (buff(:1).eq.'T') then
        ! tprid
        ! T  i   j   k  id -> TPRID_D(i,j,k)=id
          read(buff(2:),*) i,j,k,id
          tprid_d(i,j,k) = id
        else if (buff(:1).eq.'M') then
        ! pmass and pwidth
          read(buff(2:),*) i,j,k,id
        ! M  i   j   k  id -> gives id of particle of which 
        ! the mass/width is stored in PMASS/WIDTH_D(i,j,k)
          pmass_d(i,j,k) = get_mass_from_id(id)
          pwidth_d(i,j,k) = get_width_from_id(id)
        else if (buff(:1).eq.'P') then
        ! pow
        ! P  i   j   k  id -> POW_D(i,j,k)=id
          read(buff(2:),*) i,j,k,id
          pow_d(i,j,k) = id
        endif
      enddo
 999  continue
      close(78)

      return 
      end



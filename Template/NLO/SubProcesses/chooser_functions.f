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
      include 'maxconfigs.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig
      double precision prmass(-max_branch:nexternal,lmaxconfigs)
      double precision prwidth(-max_branch:-1,lmaxconfigs)
      integer prow(-max_branch:-1,lmaxconfigs)
      common/c_props_inc/prmass,prwidth,prow
      double precision pmass(nexternal)
      logical firsttime
      data firsttime /.true./
      include 'configs_and_props_decl.inc'
      save mapconfig_d,iforest_d,sprop_d,tprid_d,pmass_d,pwidth_d,pow_d
      include "pmass.inc"
c     
      if (max_branch_used.gt.max_branch) then
         write (*,*) 'ERROR in configs_and_props_inc_chooser:'/
     $        /' increase max_branch',max_branch,max_branch_used
         stop
      endif
      if (lmaxconfigs_used.gt.lmaxconfigs) then
         write (*,*) 'ERROR in configs_and_propsinc_chooser:'/
     $        /' increase lmaxconfigs' ,lmaxconfigs,lmaxconfigs_used
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


      subroutine fks_inc_chooser()
c For a given nFKSprocess, it fills the c_fks_inc common block with the
c fks.inc information
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      integer i,j
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision particle_charge(nexternal), particle_charge_born(nexternal-1)
      common /c_charges/particle_charge
      common /c_charges_born/particle_charge_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      integer particle_type_born(nexternal-1)
      common /c_particle_type_born/particle_type_born
      logical particle_tag(nexternal)
      common /c_particle_tag/particle_tag
      logical particle_tag_born(nexternal-1)
      common /c_particle_tag/particle_tag_born
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      integer extra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/extra_cnt, isplitorder_born, isplitorder_cnt
      include 'orders.inc'
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      logical is_aorg(nexternal)
      common /c_is_aorg/is_aorg
      logical is_charged(nexternal)
      common /c_is_charged/is_charged
c
      i_fks=fks_i_D(nFKSprocess)
      j_fks=fks_j_D(nFKSprocess)
      extra_cnt = extra_cnt_D(nFKSprocess)
      isplitorder_born = isplitorder_born_D(nFKSprocess)
      isplitorder_cnt = isplitorder_cnt_D(nFKSprocess)
      need_color_links=need_color_links_d(nFKSprocess)
      need_charge_links=need_charge_links_d(nFKSprocess)
      do i=1,nexternal
         if (fks_j_from_i_D(nFKSprocess,i,0).ge.0 .and.
     &        fks_j_from_i_D(nFKSprocess,i,0).le.nexternal) then
            do j=0,fks_j_from_i_D(nFKSprocess,i,0)
               fks_j_from_i(i,j)=fks_j_from_i_D(nFKSprocess,i,j)
            enddo
         else
            write (*,*) 'ERROR in fks_inc_chooser'
            stop
         endif
         particle_type(i)=particle_type_D(nFKSprocess,i)
         particle_tag(i)=particle_tag_D(nFKSprocess,i)
         particle_charge(i)=particle_charge_D(nFKSprocess,i)
         pdg_type(i)=pdg_type_D(nFKSprocess,i)
         ! is_aorg is true if the particle can induce soft singularities
         ! currencly it is based on the pdg id (photon or gluon)
         is_aorg(i) = abs(pdg_type(i)).eq.21.or.pdg_type(i).eq.22
         ! is_charged is true if the particle has any color or electric
         ! charge
         is_charged(i) = particle_type(i).ne.1.or.
     &                   particle_charge(i).ne.0d0
      enddo
      do i=1,nexternal
         if (i.lt.min(i_fks,j_fks)) then
            particle_type_born(i)=particle_type(i)
            particle_charge_born(i)=particle_charge(i)
            particle_tag_born(i)=particle_tag(i)
         elseif (i.gt.max(i_fks,j_fks)) then
            particle_type_born(i-1)=particle_type(i)
            particle_charge_born(i-1)=particle_charge(i)
            particle_tag_born(i-1)=particle_tag(i)
         elseif (i.eq.min(i_fks,j_fks)) then
            i_type=particle_type(i_fks)
            j_type=particle_type(j_fks)
            ch_i=particle_charge(i_fks)
            ch_j=particle_charge(j_fks)
            particle_tag_born(i) = particle_tag(j_fks)
            call get_mother_col_charge(i_type,ch_i,j_type,ch_j,m_type,ch_m) 
            particle_type_born(i)=m_type
            particle_charge_born(i)=ch_m
         elseif (i.ne.max(i_fks,j_fks)) then
            particle_tag_born(i) = particle_tag(i)
            particle_type_born(i)=particle_type(i)
            particle_charge_born(i)=particle_charge(i)
         endif
      enddo
      
      do i = 1, nsplitorders
         split_type(i) = split_type_d(nFKSprocess,i)
      enddo
      return
      end


      subroutine leshouche_inc_chooser()
c For a given nFKSprocess, it fills the c_leshouche_inc common block with the
c leshouche.inc information
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      logical firsttime
      data firsttime /.true./
      include 'leshouche_decl.inc'
      common/c_leshouche_idup_d/ idup_d
      save mothup_d, icolup_d, niprocs_d
      
c
      if (maxproc_used.gt.maxproc) then
         write (*,*) 'ERROR in leshouche_inc_chooser: increase maxproc',
     &        maxproc,maxproc_used
         stop
      endif
      if (maxflow_used.gt.maxflow) then
         write (*,*) 'ERROR in leshouche_inc_chooser: increase maxflow',
     &        maxflow,maxflow_used
         stop
      endif

      if (firsttime) then
        call read_leshouche_info(idup_d,mothup_d,icolup_d,niprocs_d)
        firsttime = .false.
      endif

      niprocs=niprocs_d(nFKSprocess)
      do j=1,niprocs
         do i=1,nexternal
            IDUP(i,j)=IDUP_D(nFKSprocess,i,j)
            MOTHUP(1,i,j)=MOTHUP_D(nFKSprocess,1,i,j)
            MOTHUP(2,i,j)=MOTHUP_D(nFKSprocess,2,i,j)
         enddo
      enddo
c
      do j=1,maxflow_used
         do i=1,nexternal
            ICOLUP(1,i,j)=ICOLUP_D(nFKSprocess,1,i,j)
            ICOLUP(2,i,j)=ICOLUP_D(nFKSprocess,2,i,j)
         enddo
      enddo
c
      return
      end


      subroutine read_configs_and_props_info(mapconfig_d,iforest_d
     $     ,sprop_d,tprid_d,pmass_d,pwidth_d,pow_d)
C read the various information from the configs_and_props_info.dat file
      implicit none
      integer i,j,k
      integer ndau, idau, dau, id
      character *200 buff
      double precision get_mass_from_id, get_width_from_id
      include 'configs_and_props_decl.inc'
      mapconfig_d=0
      iforest_d=0
      sprop_d=0
      tprid_d=0
      pmass_d=0d0
      pwidth_d=0d0
      pow_d=0
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


      subroutine read_leshouche_info(idup_d,mothup_d,icolup_d,niprocs_d)
C read the various information from the configs_and_props_info.dat file
      implicit none
      include "nexternal.inc"
      integer itmp_array(nexternal)
      integer i,j,k,l
      character *200 buff
      include 'leshouche_decl.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'born_maxamps.inc'
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'born_leshouche.inc'
      if (fks_configs.eq.1) then
         if (pdg_type_d(1,fks_i_d(1)).eq.-21) then
c SPECIAL for [LOonly=QCD] process. Simply use the information from the
c born_leshouche.inc file.
            do j=1,maxproc
               do i=1,nexternal-1
                  idup_d(1,i,j)=idup(i,j)
                  mothup_d(1,1,i,j)=mothup(1,i,j)
                  mothup_d(1,2,i,j)=mothup(2,i,j)
               enddo
               idup_d(1,nexternal,j)=-21
               mothup_d(1,1,nexternal,j)=mothup(1,fks_j_d(1),j)
               mothup_d(1,2,nexternal,j)=mothup(2,fks_j_d(1),j)
            enddo
            do j=1,maxflow
               do i=1,nexternal-1
                  icolup_d(1,1,i,j)=icolup(1,i,j)
                  icolup_d(1,2,i,j)=icolup(2,i,j)
               enddo
               icolup_d(1,1,nexternal,j)=-99999 ! should not be used
               icolup_d(1,2,nexternal,j)=-99999
            enddo
            niprocs_d(1)=maxproc_used
            return
         endif
      endif

      open(unit=78, file='leshouche_info.dat', status='old')
      do while (.true.)
        read(78,'(a)',end=999) buff
        if (buff(:1).eq.'#') cycle
        if (buff(:1).eq.'I') then
        ! idup
        ! I  i   j   id1 ..idn -> IDUP_D(i,k,j)=idk
          read(buff(2:),*) i,j,(itmp_array(k),k=1,nexternal)
          do k=1,nexternal
            idup_d(i,k,j)=itmp_array(k)
          enddo
          niprocs_d(i)=j
        else if (buff(:1).eq.'M') then
        ! idup
        ! I  i   j   l   id1 ..idn -> MOTHUP_D(i,j,k,l)=idk
          read(buff(2:),*) i,j,l,(itmp_array(k),k=1,nexternal)
          do k=1,nexternal
            mothup_d(i,j,k,l)=itmp_array(k)
          enddo
        else if (buff(:1).eq.'C') then
        ! idup
        ! I  i   j   l   id1 ..idn -> ICOLUP_D(i,j,k,l)=idk
          read(buff(2:),*) i,j,l,(itmp_array(k),k=1,nexternal)
          do k=1,nexternal
            icolup_d(i,j,k,l)=itmp_array(k)
          enddo
        endif
      enddo
 999  continue
      close(78)

      return 
      end


      subroutine get_mother_col_charge(i_type, ch_i, j_type, ch_j,
     $     m_type, ch_m)
C Given the type (color representation) and charges of i and j, return
C the type and charges of the mother particle
      implicit none
      integer i_type, j_type, m_type
      double precision ch_i, ch_j, ch_m
      include 'nexternal.inc'
      integer i_fks,j_fks

      common/fks_indices/i_fks,j_fks

      if (abs(i_type).eq.abs(j_type) .and. 
     &    abs(ch_i).eq.abs(ch_j) .and. 
     &    abs(i_type).gt.1) then
        ! neutral color octet splitting
         m_type=8
         ch_m = 0d0
         if ( (j_fks.le.nincoming .and.
     &         abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &        (j_fks.gt.nincoming .and.
     &         abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
            write(*,*)'Flavour mismatch #1col in get_mother_col_charge',
     &                 i_fks,j_fks,i_type,j_type
            stop
         endif
      elseif (abs(i_type).eq.abs(j_type) .and. 
     &        dabs(ch_i).eq.dabs(ch_j).and.abs(i_type).eq.1) then
        ! neutral color singlet splitting
         m_type=1
         ch_m = 0d0
         if ( (j_fks.le.nincoming .and.
     &         dabs(ch_i).gt.0d0 .and. ch_j.ne.ch_i) .or.
     &        (j_fks.gt.nincoming .and.
     &         dabs(ch_i).gt.0d0 .and. ch_j.ne.-ch_i)) then
            write(*,*)'Flavour mismatch #1chg in get_mother_col_charge',
     &                 i_fks,j_fks,ch_i,ch_j
            stop
         endif
      elseif ((abs(i_type).eq.3 .and. j_type.eq.8) .or.
     &        (dabs(ch_i).gt.0d0 .and. ch_j.eq.0d0) ) then
         if(j_fks.le.nincoming)then
            m_type=-i_type
            if (m_type.eq.-1) m_type=1
            ch_m = -ch_i
         else
            write(*,*) 'Error in get_mother_col_charge: (i,j)=(q,g)'
            stop
         endif
      elseif ((i_type.eq.8 .and. abs(j_type).eq.3) .or.
     &         (ch_i.eq.0d0 .and. dabs(ch_j).gt.0d0) ) then
         m_type=j_type
         ch_m= ch_j
!     special for processes without a proper NLO contribution (i.e., PDG(i_fks)=-21)
      elseif (i_type.eq.8 .and. j_type.eq.1 .and. ch_i.eq.0d0 .and.
     $        ch_j.eq.0d0) then
         m_type=0
         ch_m=0d0
         continue
      else
         write(*,*)'Flavour mismatch #2 in get_mother_col_charge',
     &      i_type,j_type,m_type
         stop
      endif
      return
      end



      subroutine set_pdg(ict,iFKS)
c fills the pdg and pdg_uborn variables. It uses only the 1st IPROC. For
c the pdg_uborn (the PDG codes for the underlying Born process) the PDG
c codes of i_fks and j_fks are combined to give the PDG code of the
c mother and the extra (n+1) parton is given the PDG code of the gluon.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      include 'genps.inc'
      integer k,ict,iFKS
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     $     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      include 'orders.inc'
      do k=1,nexternal
         pdg(k,ict)=idup(k,1)
      enddo
      do k=1,nexternal
         if (k.lt.fks_j_d(iFKS)) then
            pdg_uborn(k,ict)=pdg(k,ict)
         elseif(k.eq.fks_j_d(iFKS)) then
            if ( abs(pdg(fks_i_d(iFKS),ict)) .eq.
     &           abs(pdg(fks_j_d(iFKS),ict)) .and.
     &           abs(pdg(fks_i_d(iFKS),ict)).ne.21.and.
     &           abs(pdg(fks_i_d(iFKS),ict)).ne.22) then
c gluon splitting:  g/a -> ff
               !!!pdg_uborn(k,ict)=21
               ! check if any extra cnt is needed
               if (extra_cnt_d(iFKS).eq.0) then
                  ! if not, assign photon/gluon depending on split_type
                  if (split_type_d(iFKS,qcd_pos)) then
                    pdg_uborn(k,ict)=21
                  else if (split_type_d(iFKS,qed_pos)) then
                    pdg_uborn(k,ict)=22
                  else
                    write (*,*) 'set_pdg ',
     &                'ERROR#1 in PDG assigment for underlying Born'
                    stop 1
                  endif
               else
                  ! if there are extra cnt's, assign the pdg of the
                  ! mother in the born (according to isplitorder_born_d)
                  if (isplitorder_born_d(iFKS).eq.qcd_pos) then
                    pdg_uborn(k,ict)=21
                  else if (isplitorder_born_d(iFKS).eq.qcd_pos) then
                    pdg_uborn(k,ict)=22
                  else
                    write (*,*) 'set_pdg ',
     &                'ERROR#2 in PDG assigment for underlying Born'
                    stop 1
                  endif
               endif
            elseif (abs(pdg(fks_i_d(iFKS),ict)).eq.21.or.
     &              abs(pdg(fks_i_d(iFKS),ict)).eq.22) then
c final state gluon radiation:  X -> Xg
               pdg_uborn(k,ict)=pdg(fks_j_d(iFKS),ict)
            elseif (pdg(fks_j_d(iFKS),ict).eq.21.or.
     &              pdg(fks_j_d(iFKS),ict).eq.22) then
c initial state gluon splitting (gluon is j_fks):  g -> XX
               pdg_uborn(k,ict)=-pdg(fks_i_d(iFKS),ict)
            else
               write (*,*)
     &          'set_pdg ERROR#3 in PDG assigment for underlying Born'
               stop 1
            endif
         elseif(k.lt.fks_i_d(iFKS)) then
            pdg_uborn(k,ict)=pdg(k,ict)
         elseif(k.eq.nexternal) then
            if (split_type_d(iFKS,qcd_pos)) then
              pdg_uborn(k,ict)=21  ! give the extra particle a gluon PDG code
            elseif (split_type_d(iFKS,qed_pos)) then
              pdg_uborn(k,ict)=22  ! give the extra particle a photon PDG code
            endif
         elseif(k.ge.fks_i_d(iFKS)) then
            pdg_uborn(k,ict)=pdg(k+1,ict)
         endif
      enddo
      return
      end

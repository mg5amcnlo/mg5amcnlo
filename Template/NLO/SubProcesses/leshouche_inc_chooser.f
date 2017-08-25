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

      entry read_leshouche_info2
 
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

c$$$      subroutine read_leshouche_info2(idup_d,mothup_d,icolup_d,niprocs_d)
c$$$C read the various information from the configs_and_props_info.dat file
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$      integer itmp_array(nexternal)
c$$$      integer i,j,k,l
c$$$      character *200 buff
c$$$      include 'leshouche_decl.inc'
c$$$      include 'nFKSconfigs.inc'
c$$$      include 'fks_info.inc'
c$$$      include 'born_maxamps.inc'
c$$$c$$$      integer idup(nexternal,maxproc)
c$$$c$$$      integer mothup(2,nexternal,maxproc)
c$$$c$$$      integer icolup(2,nexternal,maxflow)
c$$$c$$$      include 'born_leshouche.inc'
c$$$c$$$      if (fks_configs.eq.1) then
c$$$c$$$         if (pdg_type_d(1,fks_i_d(1)).eq.-21) then
c$$$c$$$c SPECIAL for [LOonly=QCD] process. Simply use the information from the
c$$$c$$$c born_leshouche.inc file.
c$$$c$$$            do j=1,maxproc
c$$$c$$$               do i=1,nexternal-1
c$$$c$$$                  idup_d(1,i,j)=idup(i,j)
c$$$c$$$                  mothup_d(1,1,i,j)=mothup(1,i,j)
c$$$c$$$                  mothup_d(1,2,i,j)=mothup(2,i,j)
c$$$c$$$               enddo
c$$$c$$$               idup_d(1,nexternal,j)=-21
c$$$c$$$               mothup_d(1,1,nexternal,j)=mothup(1,fks_j_d(1),j)
c$$$c$$$               mothup_d(1,2,nexternal,j)=mothup(2,fks_j_d(1),j)
c$$$c$$$            enddo
c$$$c$$$            do j=1,maxflow
c$$$c$$$               do i=1,nexternal-1
c$$$c$$$                  icolup_d(1,1,i,j)=icolup(1,i,j)
c$$$c$$$                  icolup_d(1,2,i,j)=icolup(2,i,j)
c$$$c$$$               enddo
c$$$c$$$               icolup_d(1,1,nexternal,j)=-99999 ! should not be used
c$$$c$$$               icolup_d(1,2,nexternal,j)=-99999
c$$$c$$$            enddo
c$$$c$$$            niprocs_d(1)=maxproc_used
c$$$c$$$            return
c$$$c$$$         endif
c$$$c$$$      endif
c$$$
c$$$      open(unit=78, file='leshouche_info.dat', status='old')
c$$$      do while (.true.)
c$$$        read(78,'(a)',end=999) buff
c$$$        if (buff(:1).eq.'#') cycle
c$$$        if (buff(:1).eq.'I') then
c$$$        ! idup
c$$$        ! I  i   j   id1 ..idn -> IDUP_D(i,k,j)=idk
c$$$          read(buff(2:),*) i,j,(itmp_array(k),k=1,nexternal)
c$$$          do k=1,nexternal
c$$$            idup_d(i,k,j)=itmp_array(k)
c$$$          enddo
c$$$          niprocs_d(i)=j
c$$$        else if (buff(:1).eq.'M') then
c$$$        ! idup
c$$$        ! I  i   j   l   id1 ..idn -> MOTHUP_D(i,j,k,l)=idk
c$$$          read(buff(2:),*) i,j,l,(itmp_array(k),k=1,nexternal)
c$$$          do k=1,nexternal
c$$$            mothup_d(i,j,k,l)=itmp_array(k)
c$$$          enddo
c$$$        else if (buff(:1).eq.'C') then
c$$$        ! idup
c$$$        ! I  i   j   l   id1 ..idn -> ICOLUP_D(i,j,k,l)=idk
c$$$          read(buff(2:),*) i,j,l,(itmp_array(k),k=1,nexternal)
c$$$          do k=1,nexternal
c$$$            icolup_d(i,j,k,l)=itmp_array(k)
c$$$          enddo
c$$$        endif
c$$$      enddo
c$$$ 999  continue
c$$$      close(78)
c$$$
c$$$      return 
c$$$      end

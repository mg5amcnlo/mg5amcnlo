      subroutine leshouche_inc_chooser()
c For a given nFKSprocess, it fills the c_leshouche_inc common block with the
c leshouche.inc information
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_leshouche_inc/idup,mothup,icolup
      logical firsttime
      data firsttime /.true./
      include 'leshouche_decl.inc'
      save idup_d, mothup_d, icolup_d
      
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
        call read_leshouche_info(idup_d,mothup_d,icolup_d)
        firsttime = .false.
      endif

      do j=1,maxproc_used
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


      subroutine read_leshouche_info(idup_d,mothup_d,icolup_d)
C read the various information from the configs_and_props_info.dat file
      implicit none
      include "nexternal.inc"
      integer itmp_array(nexternal)
      integer i,j,k,l
      character *200 buff
      include 'leshouche_decl.inc'

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


      subroutine initcluster()
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'cluster.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      logical filmap
      external filmap
c   
c     initialize clustering map
c         
      do nFKSprocess=1,fks_configs
c Fill all the common blocks with global particle/diagram information
c depending on the value of nFKSprocess
         call fks_inc_chooser()
         call leshouche_inc_chooser()
         call configs_and_props_inc_chooser()
         if (.not.filmap()) then
            write(*,*) 'cuts.f: cluster map initialization failed'
            stop
         endif
         write (*,*) 'Diagram information for '/
     $        /' clustering has been set-up for nFKSprocess',
     $        nFKSprocess
      enddo
      igraphs(0)=0
      RETURN
      END


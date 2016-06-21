      subroutine find_iproc_map
c Determines which IPROC's can be combined at NLO (i.e. give the same
c flavor structure in the event file and can be summed before taking the
c absolute value).
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision dummy,dlum
      integer maxproc_found_first,i,j,ii,jj,k,kk
      integer id_current(nexternal,maxproc),id_first(nexternal,maxproc)
     $     ,nequal,equal_to(maxproc,fks_configs)
     $     ,equal_to_inverse(maxproc,fks_configs)
      character*100 buff
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     This is the common block that this subroutine fills
      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      do nFKSprocess=1,fks_configs
         call fks_inc_chooser()
         call leshouche_inc_chooser()
c Set Bjorken x's to some random value before calling the dlum() function
         xbk(1)=0.5d0
         xbk(2)=0.5d0
         dummy=dlum()
c 1. First map the IPROC's for this nFKSprocess to the underlying Born
c to get the unique IPROC's
         iproc_save(nFKSprocess)=iproc
         do j=1,iproc
            do i=1,nexternal-1
               if (i.eq.min(j_fks,i_fks)) then
                  if (abs(idup(i_fks,j)).eq.abs(idup(j_fks,j))) then
c     gluon splitting
                     id_current(i,j)=21
                  elseif(abs(idup(i_fks,j)).eq.21) then
c     final state gluon emitted
                     id_current(i,j)=idup(j_fks,j)
                  elseif(idup(j_fks,j).eq.21) then 
c     intial state g->qqbar splitting
                     id_current(i,j)=-idup(i_fks,j)
                  else
                     write (*,*) 'Error #1 in unwgt_table',nFKSprocess
     $                    ,idup(i_fks,j),idup(j_fks,j)
                     stop
                  endif
               elseif (i.lt.max(j_fks,i_fks)) then
                  id_current(i,j)=idup(i,j)
               else
                  id_current(i,j)=idup(i+1,j)
               endif
            enddo
c 1a. if IPROC not yet found, save it for checking. Also fill an array
c equal_to() that maps the IPROC to the set of unique IPROCs
            if (j.eq.1) then
               maxproc_found=1
               equal_to(j,nFKSprocess)=1
               equal_to_inverse(1,nFKSprocess)=j
            elseif (j.gt.1) then
               do jj=1,maxproc_found
                  nequal=0
                  do ii=1,nexternal-1
                     if (id_current(ii,j).eq.id_current(ii
     &                    ,equal_to_inverse(jj,nFKSprocess))) then
                        nequal=nequal+1
                     endif
                  enddo
                  if (nequal.eq.nexternal-1) then
                     equal_to(j,nFKSprocess)=jj
                     exit
                  endif
               enddo
               if (nequal.ne.nexternal-1) then
                  maxproc_found=maxproc_found+1
                  equal_to(j,nFKSprocess)=maxproc_found
                  equal_to_inverse(maxproc_found,nFKSprocess)=j
               endif
            endif
         enddo
c 2. Now that we have the unique IPROCs for a given nFKSprocess, we need
c to check that they are equal among all nFKSprocesses.
         if (nFKSprocess.eq.1) then
            maxproc_found_first=maxproc_found
            do j=1,iproc
               if (j.le.maxproc_found) then
                  do i=1,nexternal-1
                     id_first(i,j)=id_current(i,equal_to_inverse(j
     &                    ,nFKSprocess))
                  enddo
                  eto(j,nFKSprocess)=equal_to(j,nFKSprocess)
                  etoi(j,nFKSprocess)=equal_to_inverse(j,nFKSprocess)
               else
                  eto(j,nFKSprocess)=equal_to(j,nFKSprocess)
               endif
            enddo
         else
            if (maxproc_found.ne.maxproc_found_first) then
               write (*,*) 'Number of unique IPROCs not identical'/
     $              /' among nFKSprocesses',nFKSprocess,maxproc_found
     $              ,maxproc_found_first
               stop
            endif
c If order not equal: re-order them. This will fill the eto() and etoi()
c arrays which map the processes for a given FKS dir to the 1st FKS dir
            do j=1,iproc
               do jj=1,maxproc_found
                  nequal=0
                  do ii=1,nexternal-1
                     if (id_current(ii,j) .eq. id_first(ii,jj)) then
                        nequal=nequal+1
                     endif
                  enddo
                  if (nequal.eq.nexternal-1) then
                     eto(j,nFKSprocess)=jj
                     etoi(jj,nFKSprocess)=j
                  endif
               enddo
            enddo
c Should have the correct mapping now. Check that this is indeed the
c case.
            do j=1,maxproc_found
               do i=1,nexternal-1
                  if (id_current(i,etoi(j,nFKSprocess)) .ne.
     &                 id_first(i,j)) then
                     write (*,*)'Particle IDs not equal (inverse)',j
     &                    ,nFKSprocess,maxproc_found,iproc
                     do jj=1,maxproc_found
                        write (*,*) jj,etoi(jj,nFKSprocess)
     &                       ,' current:', (id_current(ii,etoi(jj
     &                       ,nFKSprocess)),ii=1,nexternal-1)
                        write (*,*) jj,jj
     &                       ,' saved  :', (id_first(ii
     &                       ,jj),ii=1,nexternal-1)
                     enddo
                     stop
                  endif
               enddo
            enddo
            do j=1,iproc
               do i=1,nexternal-1
                  if (id_current(i,j) .ne. id_first(i,eto(j
     &                 ,nFKSprocess))) then
                     write (*,*)'Particle IDs not equal',j
     &                    ,nFKSprocess,maxproc_found,iproc
                     do jj=1,iproc
                        write (*,*) jj,jj ,' current:',
     &                       (id_current(ii,jj),ii=1 ,nexternal-1)
                        write (*,*) jj,jj,' saved  :', (id_first(ii
     &                       ,eto(jj,nFKSprocess)),ii=1,nexternal-1)
                     enddo
                     stop
                  endif
               enddo
            enddo
         endif
c Print the map to the screen
         if (nFKSprocess.eq.1) 
     &        write (*,*) '================================'
         if (nFKSprocess.eq.1) write (*,*) 'process combination map '
     &        //'(specified per FKS dir):'
         write (buff(1:3),'(i3)') nFKSprocess
         write (buff(4:13),'(a)') ' map     '
         do j=1,iproc
            write (buff(10+4*j:13+4*j),'(i4)') eto(j,nFKSprocess)
         enddo
         write (*,'(a)') buff(1:13+4*iproc)
         write (buff(1:3),'(i3)') nFKSprocess
         write (buff(4:13),'(a)') ' inv. map'
         do j=1,maxproc_found
            write (buff(10+4*j:13+4*j),'(i4)') etoi(j,nFKSprocess)
         enddo
         write (*,'(a)') buff(1:13+4*maxproc_found)
         if (nFKSprocess.eq.fks_configs) 
     &        write (*,*) '================================'
      enddo
      return
      end


************************************************************************
*     The following routine sets up the flavour map that needs to be
*     feeded to APPLgrid in the initialization stage.
************************************************************************
      subroutine setup_flavourmap
*
      implicit none
*
      include 'nFKSconfigs.inc'
      include 'nexternal.inc'
      include 'genps.inc'
      include "leshouche_decl.inc"
      include "reweight_appl.inc"
      include "appl_common.inc"
*
      character*200 buffer
      integer procnum,i,l,j,ll,found_a,found_m
      logical found_appl(mxpdflumi),found_mg(maxproc)
      integer  nmatch_total

*     npdflumi is the number of pdf luminosities in this particular process
      integer npdflumi,kpdflumi,ilumi
      integer nproc(mxpdflumi)
      
      integer pdgs(2,max_nproc,mxpdflumi)
      integer flavour_map(fks_configs)
      common/c_flavour_map/flavour_map
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      logical pass
      logical flav_map_debug
      parameter(flav_map_debug=.false.)

      if(flav_map_debug) then
         write(6,*) " ---------------------------------------------- "
         write(6,*) "              in setup_flavormap                "
         write(6,*) " ---------------------------------------------- "
      endif

*     Open the file with the information on the initial states map
      open(unit=71,status="old",file="initial_states_map.dat")
      
*     Read the file using a buffer
      do
         read (71,'(a)',err=100,end=100) buffer ! Jump to line 100 when all lines read
         read (buffer,*) kpdflumi,nproc(kpdflumi),
     1        ((pdgs(i,j,kpdflumi),i=1,2),j=1,nproc(kpdflumi))
         appl_nproc(kpdflumi) = nproc(kpdflumi)
      enddo
 100  continue
      close(71)

      appl_nlumi = kpdflumi     ! Number of independent lumis for this process
      do ilumi=1,appl_nlumi     ! loop over independent lumis
         do j=1, appl_nproc(ilumi) ! loop over components of lumi
            do i=1,2            ! the two incoming parton flavors
               if(pdgs(i,j,ilumi).ne.21) then
                  appl_lumimap(i,j,ilumi) = pdgs(i,j,ilumi)
               elseif(pdgs(i,j,ilumi).eq.21) then
                  appl_lumimap(i,j,ilumi) = 0 ! Use the LHAPDF convention
               endif
            enddo
         enddo
      enddo

*     Set value of npdflumi
      if(nproc(kpdflumi).eq.0) then
         npdflumi = kpdflumi -1 ! Value of last line in initial_states_map.dat
      else
         npdflumi = kpdflumi
      endif
      if(flav_map_debug)then
         write(6,*) " kpdflumi = ",kpdflumi
         write(6,*) " npdflumi = ",npdflumi
         write(6,*) " "
      endif

*     checks
      if(flav_map_debug) then
         write(6,*) "" 
         write(6,*) " check reading of initial_state_map.data " 
         write(6,*) "" 
         write(6,*)" kpdflumi,nproc(kpdflumi),"/
     $        /"((pdgs(i,j,kpdflumi),i=1,2),j=1,nproc(kpdflumi)"
         do kpdflumi=1,npdflumi

*     Check not too large number of processes
            if (nproc(kpdflumi).gt.max_nproc) then
               write (*,*) 'Increase max_nproc in setup_flavourmap',
     1              nproc(kpdflumi)
               stop
            endif
*
            write(6,*) kpdflumi,nproc(kpdflumi),
     1           ((pdgs(i,j,kpdflumi),i=1,2),j=1,nproc(kpdflumi))
*
            if(kpdflumi.le.0.or.kpdflumi.gt.mxpdflumi) then
               write(6,*) "In iproc_map.f"
               write(6,*) "Invalid value of kpdflumi = ",kpdflumi
               stop
            endif
*
            do i=1,2
               do j=1,nproc(kpdflumi)
                  if( (pdgs(i,j,kpdflumi).ge.7.or.
     1                 pdgs(i,j,kpdflumi).le.(-7).or.
     2                 pdgs(i,j,kpdflumi).eq.0).and.
     3                 pdgs(i,j,kpdflumi).ne.21 ) then
                     write(6,*) "Invalid value for pdgs = "
                     write(6,*) kpdflumi, j, i, pdgs(i,j,kpdflumi)
                  endif
               enddo
            enddo
         enddo
      endif

*     Loop over number of fks configurations
*     These are defined in 'nFKSconfigs.inc'
      if(flav_map_debug)then
         write(6,*) "fks_configs =  ",fks_configs
      endif
      nmatch_total=0
      do nFKSprocess=1,fks_configs
*     Initialization
         flavour_map(nFKSprocess)=0
         
*     Get the relevant lumis of this fks configuration
         call leshouche_inc_chooser()

         if(flav_map_debug) then
            write(6,*) " "
            write(6,*) " nFKSprocess  = ",nFKSprocess 
            write(6,*) " maxproc_used = ",maxproc_used
            write(6,*) " niprocs      = ",niprocs
            do l=1,niprocs
               write(6,*) l,IDUP(1,l),IDUP(2,l)
            enddo
            write(6,*) " "
         endif

*     Be careful with all possible permutations in initial_parton_map!
*     Check all possible npdflumi conbinations until a match is found
         do kpdflumi=1,npdflumi

*     Initialization
            do l=1,nproc(kpdflumi)
               found_appl(l) = .false.
            enddo
            do ll=1,niprocs
               found_mg(ll) = .false.
            enddo
            found_a=0
            found_m=0

*     Look for all possible pairs of
*     a) pdgs(1,l,kpdflumi),pdgs(2,l,kpdflumi) and
*     b) IDUP(1,ll),  IDUP(2,ll)
*     with kpdflumi fixed
            do l=1,nproc(kpdflumi)
               do ll=1,niprocs
                  if ( ( pdgs(1,l,kpdflumi).eq.
     1                 IDUP(1,ll).and.pdgs(2,l,kpdflumi).
     2                 eq.IDUP(2,ll) ) ) 
     4                 then
                     found_appl(l)=.true.
                     found_mg(ll)=.true.
                     if(flav_map_debug) then
                        write(6,*) "match found!"
                        write(6,*) "pdgs = ",pdgs(1,l,kpdflumi),
     1                       pdgs(2,l,kpdflumi)
                        write(6,*) "IDUP = ",IDUP(1,ll),IDUP(2,ll)
                     endif
                  endif
               enddo
            enddo
            do l=1,nproc(kpdflumi)
               if (found_appl(l)) found_a=found_a+1
            enddo
            do ll=1,niprocs
               if (found_mg(ll)) found_m=found_m+1
            enddo
*
            if ( found_a.eq.nproc(kpdflumi) .and.
     &           found_m.eq.niprocs ) then
               if(flav_map_debug) then
                  write(6,*) " ------------------------------- "
                  write(6,*) "         Match found!"
                  write(6,*) " ------------------------------- "
                  write(6,*)  "kpdflumi = ",kpdflumi
               endif
               flavour_map(nFKSprocess)=kpdflumi
               nmatch_total = nmatch_total+1
            endif
         enddo
         if (nmatch_total.ne.nFKSprocess) then
            write(6,*) 
     1           "Problem with setup_flavourmap in iproc_map.f"
            write(6,*) "nFKSprocess = ",nFKSprocess
            write(6,*)" flavour_map(nFKSprocess)= ",
     1           flavour_map(nFKSprocess)
            stop
         endif
      enddo
      
*     Check flavor map properly initialized
*     All the entries of flavour_map(1:nFKSprocess) must be from
*     1 to npdflumi
      do nFKSprocess=1,fks_configs
         if(flav_map_debug) then
            write(6,*) "flavour_map(nFKSprocess) = ",
     1           flavour_map(nFKSprocess)
         endif
         pass=.false.
         do kpdflumi = 1, npdflumi
            if(flavour_map(nFKSprocess).eq.kpdflumi) then
*     write(6,*) "kpdflumi = ",kpdflumi
               pass=.true.
            endif
         enddo
         if(pass) then
*     write(6,*) "Flavor map ok for nFKSprocess = ",nFKSprocess
*     write(6,*) "Flavor map ok for nFKSprocess = ",nFKSprocess
         else
            write(6,*) "Problem with flavor map, stopping"
            write(6,*) " flavour_map(nFKSprocess)= ",
     1           flavour_map(nFKSprocess)
            stop
         endif
      enddo

      if(flav_map_debug) then
         write(6,*) "  -------------------------------------------- "
         write(6,*) "  "
         write (*,*) 'flavour map found:'
         write(6,*) "nFKSprocess, flavour_map(nFKSprocess)"
         do nFKSprocess=1,fks_configs
            write (*,*) nFKSprocess, flavour_map(nFKSprocess)
         enddo
         write(6,*) "  "
         write(6,*) "  -------------------------------------------- "
      endif
*
      return
      end

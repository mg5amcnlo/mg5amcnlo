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
     &     icolup(2,nexternal,maxflow)
      common /c_leshouche_inc/idup,mothup,icolup
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
                  elseif(idup(i_fks,j).eq.21) then
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
c$$$      if (maxproc_found.ne.iproc_save(nFKSprocess_used_born)) then
c$$$         write (*,*) 'ERROR #5 in unweight_table',maxproc_found
c$$$     $        ,nFKSprocess_used_born
c$$$     $        ,iproc_save(nFKSprocess_used_born)
c$$$         stop
c$$$      endif


      return
      end

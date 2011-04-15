      program genint_fks
      implicit none
      
      include "nexternal.inc"
      include "fks.inc"

      integer fksconf,i,j,same(nexternal**2)
      logical integrate

      open(unit=12,file="config.fks",status='old',err=99)
      read(12,'(I2)',err=99,end=99) fksconf
      close(12)

      do i=1,fks_configs
         same(i)=i
      enddo

      do i=1,fks_configs
         do j=i+1,fks_configs
            if (fks_i(i).le.nincoming .or. fks_i(j).le.nincoming) then
               write (*,*) 'Error #3 in genint_fks, fks_i should be '//
     &              'final state',i,j,fks_i(i),fks_i(j),nincoming
               stop
            endif
            if (fks_j(i).gt.nincoming .and. fks_j(j).gt.nincoming) then
               if(  (PDG_type(fks_i(i)).eq.PDG_type(fks_i(j)).and.
     &              PDG_type(fks_j(i)).eq.PDG_type(fks_j(j))     ).or.
     &              (PDG_type(fks_i(i)).eq.PDG_type(fks_j(j)).and.
     &              PDG_type(fks_j(i)).eq.PDG_type(fks_i(j))     ))then
c Found the same fks-contributions
                  if (fks_i(i).gt.fks_i(j)) then
                     same(j)=i
                  elseif( fks_i(i).eq.fks_i(j).and.
     &                    fks_j(i).gt.fks_j(j) )then
                     same(j)=i
                  else
                     same(i)=j
                  endif
               endif
            elseif(fks_j(i).le.nincoming.and.fks_j(j).eq.fks_j(i)) then
               if (PDG_type(fks_i(i)).eq.PDG_type(fks_i(j))) then
c Found the same fks-contributions
                  if (fks_i(i).gt.fks_i(j)) then
                     same(j)=i
                  else
                     same(i)=j
                  endif
               endif
            else
c They are not equal               
            endif
         enddo
      enddo

      integrate=same(fksconf).eq.fksconf

      open(unit=12,file="integrate.fks",status='unknown')
      if (integrate) then
         write(12,'(a)') 'Y'
      else
         write(12,'(a)') 'N'
         i=1
         do while (same(fksconf).ne.same(same(fksconf)) .and. i.le.20)
            same(fksconf)=same(same(fksconf))
            same(same(fksconf))=same(same(same(fksconf)))
            i=i+1
         enddo
         if(i.ge.20) then
            write (*,*) "Error #2 in genint_fks: cannot resolve configs"
            stop
         endif
         write(12,'(I2)') same(fksconf)
      endif
      if (particle_type(fks_i(fksconf)).ne.8) then
         write (12,'(a)') 'E'
      else
         write (12,'(a)') 'I'
      endif
      close(12)

c Also decide if we need to include this directory when using nbodyonly,
c i.e. run-mode 'born0' or 'virt0'.
      open(unit=12,file="nbodyonly.fks",status="unknown")
      if ( fks_i(fksconf).eq.nexternal .and.
     &     particle_type(fks_i(fksconf)).eq.8) then
         i=nexternal-1
         do while (particle_type(i).eq.1)
            i=i-1
            if(i.lt.1) then
               write (*,*)
     &              'WARNING: not enough colored particles in genint'
               write(12,'(a)') 'N'
            endif
         enddo
         if(fks_j(fksconf).eq.i) then
            write(12,'(a)') 'Y'
         else
            write(12,'(a)') 'N'
         endif
      else
         write(12,'(a)') 'N'
      endif
      close(12)   

      return
 99   continue
      write (*,*) "Error #1 in genint_fks: 'config.fks' not found"
      return
      end


      subroutine set_invarients(nfinal,ninvar)
c***************************************************************************
c     Calculates all of the invarients for a 2->n process
c***************************************************************************
      implicit none
c     
c     Constants
c
      include 'genps.inc'
c
c     Arguments
c
      integer nfinal,ninvar
c
c     Local
c     
      integer ip1,ip2,ipstart,ipstop,np,i
      integer ncycle
      character*10 buff
c
c     Global
c
      integer              imom(maxinvar),ninvarients
      common/to_invarients/imom          ,ninvarients
c-----
c  Begin Code
c-----

      do i=1,nfinal
         imom(i)=i
      enddo
      ipstart=1
      ipstop =nfinal
      np     =nfinal
c
c     First do all the s-channel
c
      do ncycle=2,nfinal-1
         do ip1 = ipstart,ipstop-1
            do ip2=int((real(imom(ip1))/10.-imom(ip1)/10)*10+.1)+1,
     $           nfinal
               np=np+1
               if (np .gt. maxinvar) then
                  print*,'Sorry too many invarients',np,ip1,ip2,ncycle
                  stop
               endif
               imom(np)=imom(ip1)*10+imom(ip2)
               if (imom(np) .lt. 10) then
                  write(buff,'(a2,i1)') 'S?',imom(np)
               elseif (imom(np) .lt. 100) then
                  write(buff,'(a2,i2)') 'S?',imom(np)
               elseif (imom(np) .lt. 1000) then
                  write(buff,'(a2,i3)') 'S?',imom(np)
               elseif (imom(np) .lt. 10000) then
                  write(buff,'(a2,i4)') 'S?',imom(np)
               elseif (imom(np) .lt. 100000) then
                  write(buff,'(a2,i5)') 'S?',imom(np)
               else
                  write(buff,'(a2,i6)') 'S?',imom(ip1)
               endif
c               call hbook1(100+np-nfinal,buff,100,0.,1.,0.)
c               write(*,'(i4,i6)') np-nfinal,imom(np)
               write(*,'(i4,a1,a6)') np-nfinal,'=',buff
               if ((np-nfinal)/7 .eq. real(np-nfinal)/7.) write(*,*)' '
            enddo
         enddo
         ipstart=ipstop+1
         ipstop = np
      enddo
c
c     Now do the t-channel
c
      ipstop = np
      do ip1 = 1,ipstop
c         write(*,'(i4,a2,i6)') np-nfinal+ip1,'a-',imom(ip1)
         if (imom(ip1) .lt. 10) then
            write(buff,'(a2,i1)') 'T?',imom(ip1)
         elseif (imom(ip1) .lt. 100) then
            write(buff,'(a2,i2)') 'T?',imom(ip1)
         elseif (imom(ip1) .lt. 1000) then
            write(buff,'(a2,i3)') 'T?',imom(ip1)
         elseif (imom(ip1) .lt. 10000) then
            write(buff,'(a2,i4)') 'T?',imom(ip1)
         elseif (imom(ip1) .lt. 100000) then
            write(buff,'(a2,i5)') 'T?',imom(ip1)
         else
            write(buff,'(a2,i6)') 'T?',imom(ip1)
         endif
c         call hbook1(100+np-nfinal+ip1,buff,100,0.,1.,0.)
c         write(*,*) np-nfinal+ip1,buff
         write(*,'(i4,a1,a6)') np-nfinal+ip1,'=',buff
         if ((np-nfinal+ip1)/7 .eq. real(np-nfinal+ip1)/7.) write(*,*)
      enddo
      write(*,*)
      print*,'Particles, Invarients',nfinal,np-nfinal+np
      ninvarients=np-nfinal+np
      ninvar=ninvarients
      if (ninvarients .gt. maxinvar) then
         print*,'Error too many invarients to map'
c         stop
      endif
      end


      subroutine fill_invarients(nfinal,p1,s,xx)
c***************************************************************************
c     Calculates all of the invarients for a 2->n process
c***************************************************************************
      implicit none
c     
c     Constants
c     
      include 'genps.inc'
c
c     Arguments
c
      integer nfinal
      double precision p1(0:3,nfinal+2),s,xx(55)
c
c     Local
c     
      integer ip1,ip2,ipstart,ipstop,np,i,j
      integer imom(maxinvar)
      integer ncycle
      character*10 buff
      double precision p(0:3,maxinvar),ptemp(0:3)
c
c     External
c
      double precision dot
      external         dot
c-----
c  Begin Code
c-----

      do i=1,nfinal
         imom(i) = i
         do j=0,3
            p(j,i)=p1(j,i+2)
         enddo
c         write(*,'(i3,4f17.8)') i,(p(j,i),j=0,3)
      enddo
      ipstart=1
      ipstop =nfinal
      np     =nfinal
c
c     First do all the s-channel
c
      do ncycle=2,nfinal-1
         do ip1 = ipstart,ipstop-1
            do ip2=int((real(imom(ip1))/10.-imom(ip1)/10)*10+.1)+1
     $           ,nfinal
               np=np+1
               if (np .gt. maxinvar) then
                  print*,'Sorry too many invarients',np,ip1,ip2,ncycle
                  stop
               endif
               imom(np)=imom(ip1)*10+imom(ip2)
               do j=0,3
                  p(j,np) = p(j,ip1)+p(j,ip2)
               enddo
               xx(np-nfinal) = dot(p(0,np),p(0,np))/s
c               call hfill(100+np-nfinal,
c     &              real(dot(p(0,np),p(0,np))/s),0.,wgt)
c               write(*,'(i4,3f20.8)') np-nfinal,
c     &              real(dot(p(0,np),p(0,np))/s)
            enddo
         enddo
         ipstart=ipstop+1
         ipstop = np
      enddo
c
c     Now do the t-channel
c
      ipstop = np
      do ip1 = 1,ipstop
         do j = 0,3
            ptemp(j)=p1(j,1)-p(j,ip1)
         enddo
         xx(np-nfinal+ip1)= .5d0*(dot(ptemp,ptemp)/s+1d0)
c         call hfill(100+np-nfinal+ip1,real(-dot(ptemp,ptemp)/s),0.,wgt)
c         write(*,'(i4,3f20.8)') np-nfinal+ip1,
c     &         real(-dot(ptemp,ptemp)/s)
      enddo
      end


      subroutine map_invarients(Minvar,nconfigs,ninvar,mincfig,maxcfig,nexternal,nincoming)
c****************************************************************************
c     Determines mappings for each structure of invarients onto integration
c     variables.  Input: Ninvar, iforest.  Output: Minvar, ninvar
c****************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
c
c     Arguments
c
      integer Minvar(maxdim,lmaxconfigs),nconfigs,ninvar,nexternal,nincoming
      integer mincfig,maxcfig
c
c     Local
c
      integer iconfig, jgrid,j, nbranch
      logical found,tchannel
c
c     Global
c
      integer              imom(maxinvar),ninvarients
      common/to_invarients/imom          ,ninvarients
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest

c-----
c  Begin Code
c----

      nbranch = nexternal-2
      jgrid=0
c
c     
c     Try simple mapping if nconfigs = 1
c

      if (nconfigs .eq. 1) then
c         do j=1,3*nbranch-4+2
         do j=1,maxdim
            minvar(j,mincfig)=j
         enddo
         jgrid=j-1
      else
c      if (ep) jgrid=1
c      if (pp) jgrid=2
      do iconfig=mincfig,maxcfig
         tchannel = .false.         
         do j=1,nbranch-1
            if (iforest(1,-j,iconfig) .eq. 1) then
               tchannel=.true.
            endif
            jgrid=jgrid+1
            minvar(j,iconfig) = jgrid
            if (tchannel .and. j .lt. nbranch-1) then
               jgrid=jgrid+1            
               minvar(nbranch-1+2*j,iconfig)=jgrid
            endif
         enddo             !Each Branch
         if (.not. tchannel .and. nincoming.eq.2) then          !Don't need last s-channel
            jgrid=jgrid-1
            minvar(nbranch-1,iconfig)=0
         endif
c         if (pp) then
c            jgrid=jgrid+1
c            minvar(3*nbranch-3,iconfig)=jgrid
c            jgrid=jgrid+1
c            minvar(3*nbranch-2,iconfig)=jgrid
c         elseif (ep) then
c            jgrid=jgrid+1
c            minvar(3*nbranch-3,iconfig)=jgrid
c         endif
      enddo  !Each configurations
      endif
      ninvar = jgrid
      end

      subroutine sortint(n,ra)
      integer ra(n)
      l=n/2+1
      ir=n
10    continue
        if(l.gt.1)then
          l=l-1
          rra=ra(l)
        else
          rra=ra(ir)
          ra(ir)=ra(1)
          ir=ir-1
          if(ir.eq.1)then
            ra(1)=rra
            return
          endif
        endif
        i=l
        j=l+l
20      if(j.le.ir)then
          if(j.lt.ir)then
            if(ra(j).lt.ra(j+1))j=j+1
          endif
          if(rra.lt.ra(j))then
            ra(i)=ra(j)
            i=j
            j=j+j
          else
            j=ir+1
          endif
        go to 20
        endif
        ra(i)=rra
      go to 10
      end



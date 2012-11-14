      subroutine read_event(lun,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed,done)
c********************************************************************
c     Reads one event from data file #lun
c     ic(*,1) = Particle ID
c     ic(*,2) = Mothup(1)
c     ic(*,3) = Mothup(2)
c     ic(*,4) = ICOLUP(1)
c     ic(*,5) = ICOLUP(2)
c     ic(*,6) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(*,7) = Helicity
c********************************************************************
      implicit none
c
c     parameters
c
      integer    MaxParticles
      parameter (MaxParticles=15)
      double precision pi
      parameter (pi = 3.1415926d0)
c
c     Arguments
c      
      integer lun
      integer nexternal, ic(7,MaxParticles)
      logical done
      double precision P(0:3,MaxParticles),wgt,aqcd,aqed,scale
      integer ievent
c
c     Local
c
      integer i,j,k
      character*(132) buff
c
c     Global
c
c      include 'coupl.inc'
c      real*8          scale
c      logical               fixed_ren_scale,fixed_fac_scale
c      common/to_scale/scale,fixed_ren_scale,fixed_fac_scale

      logical banner_open
      integer lun_ban
      common/to_banner/banner_open, lun_ban

      data lun_ban/37/
      data banner_open/.false./
c-----
c  Begin Code
c-----     
      done=.false.
      if (.not. banner_open) then
         open (unit=lun_ban, status='scratch')
         banner_open=.true.
      endif
 11   read(lun,'(a132)',end=99,err=99) buff
      do while(index(buff,"#") .ne. 0)
         write(lun_ban,'(a)') buff
         read(lun,'(a132)',end=99,err=99) buff
      enddo
      read(buff,*,err=11, end=11) nexternal,k,wgt,scale,aqed,aqcd
      do j=1,7
         read(lun,*,err=99,end=99) (ic(j,i),i=1,nexternal)!This is info
      enddo      
      do j=1,nexternal
         read(lun,55,err=99,end=99) k,(p(i,j),i=0,3)
      enddo
c      gal(1) = sqrt(4d0*pi*aqed)
c      g      = sqrt(4d0*pi*aqcd)
      return
 99   done=.true.
      return
 55   format(i3,4e19.11)         
      end

      subroutine write_event(lun,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed)
c********************************************************************
c     Writes one event from data file #lun according to LesHouches
c     ic(*,1) = Particle ID
c     ic(*,2) = Mothup(1)
c     ic(*,3) = Mothup(2)
c     ic(*,4) = ICOLUP(1)
c     ic(*,5) = ICOLUP(2)
c     ic(*,6) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(*,7) = Helicity
c********************************************************************
      implicit none
c
c     parameters
c
      integer    MaxParticles
      parameter (MaxParticles=15)
      double precision pi
      parameter (pi = 3.1415926d0)
c
c     Arguments
c      
      integer lun, ievent
      integer nexternal, ic(7,MaxParticles)
      double precision P(0:3,MaxParticles),wgt
      double precision aqcd, aqed, scale
c
c     Local
c
      integer i,j,k
c
c     Global
c

c-----
c  Begin Code
c-----     
c      aqed= gal(1)*gal(1)/4d0/pi
c      aqcd = g*g/4d0/pi
      write(lun,'(2i8,4e15.7)') nexternal,ievent,wgt,scale,aqed,aqcd
      do j=1,7
         write(lun,51) (ic(j,i),i=1,nexternal)  !This is info
      enddo
      do j=1,nexternal
         write(lun,55) j,(p(i,j),i=0,3)
      enddo
      return
 51   format(19i5)
 55   format(i3,4e19.11)         
      end

      subroutine write_comments(lun)
c********************************************************************
c     Outputs all of the banner comment lines back at the top of
c     the file lun.
c********************************************************************
      implicit none
c
c     Arguments
c
      integer lun
c
c     Local
c
      character*(80) buff
c
c     Global
c
      logical banner_open
      integer lun_ban
      common/to_banner/banner_open, lun_ban

c-----
c  Begin Code
c-----     
c      write(*,*) 'Writing comments'
      if (banner_open) then
         rewind(lun_ban)
         do while (.true.) 
            read(lun_ban,'(a79)',end=99,err=99) buff
            write(lun,'(a79)') buff
c            write(*,*) buff
         enddo
 99      close(lun_ban)
         banner_open = .false.
      endif
      end


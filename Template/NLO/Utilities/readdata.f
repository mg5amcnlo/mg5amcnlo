      program readdata

      implicit none
      integer i
      double precision value(0:1,2),error(0:1,2)
      double precision eabs,etot,dabs,dtot
      character*140 buffer
      character*30 dir_name


      open(unit=30,file='res_0',status='old')
      open(unit=31,file='res_1',status='old')
      open(unit=32,file='dirname',status='old')


      read(*,*)dir_name

      do i=1,2
         read (30,'(a)',err=11,end=11) buffer
         read(buffer(index(buffer,":")+3:
     &               index(buffer,"+/-")-1),*),value(0,i)
         read(buffer(index(buffer,"+/-")+3:140),*),error(0,i)
      enddo
      do i=1,2
         read (31,'(a)',err=11,end=11) buffer
         read(buffer(index(buffer,":")+3:
     &               index(buffer,"+/-")-1),*),value(1,i)
         read(buffer(index(buffer,"+/-")+3:140),*),error(1,i)
      enddo
 11   continue

      eabs=sqrt(error(0,1)**2+error(1,1)**2)
      etot=sqrt(error(0,2)**2+error(1,2)**2)
      dabs=abs(value(0,1)-value(1,1))
      dtot=abs(value(0,2)-value(1,2))

      if(dabs.gt.3d0*eabs)write(*,*)dir_name,'ABS ',dabs/eabs,' sigma'
      if(dtot.gt.3d0*etot)write(*,*)dir_name,'TOT ',dtot/etot,' sigma'

      close(30)
      close(31)
      close(32)

      stop
      end

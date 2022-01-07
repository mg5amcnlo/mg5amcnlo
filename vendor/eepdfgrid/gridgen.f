      program main
      implicit none
! number of components
      integer ncom
      parameter (ncom=4)
! number of particles
      integer npart
      parameter (npart=2)
! Detail of particles
      integer partlst(npart)
      data partlst/-11,11/
! number of beam type
      integer nbeam
      parameter (nbeam=2)
! detail of beam id
      integer beamlst(nbeam)
      data beamlst/-11,11/
! nodes of the y grid
      integer nny
      parameter (nny=100)
! nodes of the z grid
      integer nnz
      parameter (nnz=20)
! the reference scale
      real*8 Qref
! the scale (temporary variable)
      real*8 Q
! the electron mass
      real*8 me
      parameter (me=0.511d-3)
! the electric coupling
      real*8 aem
! this is the Gmu scheme
      parameter (aem=1.d0/132.2332d0)
! grid nodes and grids
      real*8 yv(nny),zv(nnz)
      real*8,dimension(:,:,:,:,:), allocatable :: grids
c
      integer iny,inz,icom,ipart,ibeam
      integer i1,i2,i,j
c     integer ifakegrid
      real*8 ylow,yupp,zlow,zupp
c     real*8 alst,q0st,alxm,q0xm
      real*8 jkb,x
      real*8 eepdf_tilde_calc,eepdf_tilde_factor
!     temporary variables
      real*8 tv1,tv2,tvy
c
      ylow=1.d-6
      yupp=1d0-1.d-8
      zlow=log(1.d0/me)
      zupp=log(10000.d0/me)
      Qref=1d0
      allocate(grids(nny,nnz,ncom,npart,nbeam))
c     electron part
      do iny=1,nny
c     Using linear grid as the first step
c     More points at the end
      tvy=dfloat(iny-1)/dfloat(nny-1)
c     yv(iny)=(1.d0-(1.d0-tvy)**3.d0)*(yupp-ylow)+ylow
      yv(iny)=tvy*tvy*(6.d0+tvy*(-8.d0+3.d0*tvy))*(yupp-ylow)+ylow
c     yv(iny)=tvy*(3.d0-tvy*2.d0)*(yupp-ylow)+ylow
      do inz=1,nnz
      zv(inz)=dfloat(inz-1)/dfloat(nnz-1)*(zupp-zlow)+zlow
      do icom=1,ncom
      do ipart=1,npart
      do ibeam=1,nbeam
      Q=exp(zv(inz))*me
      tv1=eepdf_tilde_calc(yv(iny),Q*Q,icom
     #     ,partlst(ipart),beamlst(ibeam))
      tv2=eepdf_tilde_factor(yv(iny),Q*Q,icom
     #     ,partlst(ipart),beamlst(ibeam))
      grids(iny,inz,icom,ipart,ibeam)=tv1/tv2
!     print *,yv(iny),tv1,tv2,tv1/tv2
      enddo
      enddo
      enddo
      enddo
      enddo
c
      open(unit=10,file='eepdf.f',status='unknown')
c
      write(10,'(a)')
     #'      function eepdf_tilde(y,Q2,icom,ipart,ibeam)',
     #'      implicit none',
     #'      real*8 eepdf_tilde',
     #'      real*8 Q2,Qref,me',
     #'      integer icom,ipart,ibeam',
     #'      real*8 tmp,cstmin,cxmmin,cxmmax',
     #'      integer i,id0,listmin,lixmmin,lixmmax',
     #'      logical firsttime,check,T,F,grid(21)',
     #'      parameter (T=.true.)',
     #'      parameter (F=.false.)',
     #'      real*8 eepdf_tilde_factor'
      write(10,'(a)')
     #'      real*8 y,z',
     #'      real*8 ylow,yupp,zlow,zupp',
     #'      real*8 jkb'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (ylow=',ylow,',yupp=',yupp,')'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (zlow=',zlow,',zupp=',zupp,')'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (Qref=',Qref,',me=',me,')'
      do ipart=1,npart
      do ibeam=1,nbeam
      do icom=1,ncom
!     Use ipart instead of the real PID, as we have negative PID
        write(10,'(a,i0,a,i0,a,i0)')
     #'      real*8 eepdf_',icom,'_',ipart,'_',ibeam
      enddo
      enddo
      enddo
      write(10,'(a)')
     #'      z=0.5d0*log(Q2/me/me)'
!     No check now
      do icom=1,ncom
      if(icom.eq.1)then
        write(10,'(a)')
     #'      if(icom.eq.1)then'
      else
        write(10,'(a,i0,a)')
     #'      else if(icom.eq.',icom,')then'
      endif
        do ipart=1,npart
        if(ipart.eq.1)then
          write(10,'(a,i0,a)')
     #'        if(ipart.eq.',partlst(ipart),')then'
        else
          write(10,'(a,i0,a)')
     #'        else if(ipart.eq.',partlst(ipart),')then'
        endif
        do ibeam=1,nbeam
        if(ibeam.eq.1)then
          write(10,'(a,i0,a)')
     #'          if(ibeam.eq.',beamlst(ibeam),')then'
        else
          write(10,'(a,i0,a)')
     #'          else if(ibeam.eq.',beamlst(ibeam),')then'
        endif
        write(10,'(a,i0,a,i0,a,i0,a)')
     #'          tmp=eepdf_',icom,'_',ipart,'_',ibeam,'(y,z)'
        enddo
        write(10,'(a)')
     #'          else',
     #'            tmp=0d0',
     #'          endif'

        enddo
        write(10,'(a)')
     #'        else',
     #'          tmp=0d0',
     #'        endif'
        enddo
        write(10,'(a)')
     #'      else',
     #'        tmp=0d0',
     #'      endif',
     #'      eepdf_tilde=tmp*eepdf_tilde_factor(y,Q2,icom,ipart,ibeam)',
     #'      end'
        write(10,'(a)')
     #'c',
     #'c',
     #'cccc',
     #'c',
     #'c'
 
         do icom=1,ncom
         do ipart=1,npart
         do ibeam=1,nbeam

         write(10,'(a,i0,a,i0,a,i0,a)')
     #'      function eepdf_',icom,'_',ipart,'_',ibeam,'(y,z)'
         write(10,'(a)')
     #'      implicit none'
         write(10,'(a,i1,a,i1,a,i0,a)')
     #'      real*8 eepdf_',icom,'_',ipart,'_',ibeam,',y,z'
         write(10,'(a)')
     #'      integer narg,nny,nnz',
     #'      parameter (narg=2)'
        write(10,'(a,i0,a)')
     #'      parameter (nny=',nny,')'
        write(10,'(a,i0,a)')
     #'      parameter (nnz=',nnz,')'
        write(10,'(a)')
     #'      integer iny,inz,nent(narg)',
     #'      real*8 tmp,dfint,ymap,zmap',
     #'      real*8 arg(narg),ent(nny+nnz)',
     #'      real*8 yv(nny),zv(nnz),gridv(nny,nnz)',
     #'      logical firsttime',
     #'      external dfint,ymap,zmap'
c
        write(10,'(a)')
     #'      data yv/'
        i1=nny/4
        i2=mod(nny,4)
        if(i2.eq.0)i1=i1-1
        do i=1,i1
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(yv(j),j=1+(i-1)*4,i*4)
        enddo
        write(10,'(a,4(d15.8,a))')
     #'     #',(yv(j),',',j=1+i1*4,nny-1),yv(nny),'/'
c
        write(10,'(a)')
     #'      data zv/'
        i1=nnz/4
        i2=mod(nnz,4)
        if(i2.eq.0)i1=i1-1
        do i=1,i1
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(zv(j),j=1+(i-1)*4,i*4)
        enddo
        write(10,'(a,4(d15.8,a))')
     #'     #',(zv(j),',',j=1+i1*4,nnz-1),zv(nnz),'/'
c
        do inz=1,nnz
        write(10,'(a,i3,a,i3,a)')
     #'      data (gridv(iny,',inz,'),iny=1,',nny,')/'
        i1=nny/4
        i2=mod(nny,4)
        if(i2.eq.0)i1=i1-1
        do i=1,i1
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(grids(j,inz,icom,ipart,ibeam),j=1+(i-1)*4,i*4)
        enddo
        write(10,'(a,4(d15.8,a))')
     #'     #',(grids(j,inz,icom,ipart,ibeam),',',j=1+i1*4,nny-1),
     #    grids(nny,inz,icom,ipart,ibeam),'/'
        enddo
c
        write(10,'(a)')
     #'      data firsttime/.true./',
     #'      save',
     #'c',
     #'      if(firsttime)then',
     #'        firsttime=.false.',
     #'        nent(1)=nny',
     #'        nent(2)=nnz',
     #'        do iny=1,nny',
     #'          ent(iny)=ymap(yv(iny))',
     #'        enddo',
     #'        do inz=1,nnz',
     #'          ent(nny+inz)=zmap(zv(inz))',
     #'        enddo',
     #'      endif',
     #'      arg(1)=ymap(y)',
     #'      arg(2)=zmap(z)',
     #'      tmp=dfint(narg,arg,nent,ent,gridv)'
        write(10,'(a,i0,a,i0,a,i0,a)')
     #'      eepdf_',icom,'_',ipart,'_',ibeam,'=tmp'
        write(10,'(a)')
     #'      return',
     #'      end'
c
        write(10,'(a)')
     #'c',
     #'c',
     #'cccc',
     #'c',
     #'c'
c end of loops over itype and ipart
        enddo
        enddo
        enddo
c
        write(10,'(a)')
     #'      function ymap(st)',
     #'c Use this function to interpolate by means of',
     #'c   stnode_i=ymap(stnode_stored_i).',
     #'c Example (to be used below): tmp=log10(st)',
     #'      implicit none',
     #'      real*8 ymap,st,tmp',
     #'c',
     #'      tmp=st',
     #'      ymap=tmp',
     #'      return',
     #'      end',
     #'  ',
     #'  ',
     #'      function zmap(xm)',
     #'c Use this function to interpolate by means of',
     #'c   xmnode_i=zmap(xmnode_stored_i).',
     #'c Example (to be used below): tmp=log10(xm)',
     #'      implicit none',
     #'      real*8 zmap,xm,tmp',
     #'c',
     #'      tmp=xm',
     #'      zmap=tmp',
     #'      return',
     #'      end'
c
        end program

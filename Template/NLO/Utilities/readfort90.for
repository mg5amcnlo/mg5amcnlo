c Compile with:
c  gfortran -ffixed-line-length-132 -o readfort90 readfort90.for 
c                                       mcatnlo_hbook_gfortran.f
      implicit real*8(a-h,o-z)
      parameter (nexternal=5)
      real*8 p(0:3,nexternal)
      character*3 ch1
      character*2 ch2
c
      write(6,*)'enter number of events'
      read(5,*)nevts
      write(6,*)'enter iunit'
      read(5,*)iunit
      ch2='  '
      call fk88strnum(ch2,iunit)
      open(unit=99,file='sudakov'//ch2//'.top',status='unknown')
      call initplot
      do i=1,nevts
        read(iunit,554,err=999,end=999)ch1,sud,bogus
        do kk=1,nexternal
          read(iunit,555,err=999,end=999)
     #      kk0,p(0,kk),p(1,kk),p(2,kk),p(3,kk)
          if(kk0.ne.kk)then
            write(6,*)'error in input:',i,kk,kk0,sud,bogus
          endif
        enddo
        call outfun(p,sud,bogus)
      enddo
 999  continue
      call topout
      close(99)
 554  format(1x,a,2(1x,e14.8))
 555  format(1x,i2,5(1x,e14.8))
      end


      function xpt(en,px,py,pz)
      implicit real*8(a-h,o-z)
c
      tmp=sqrt(px**2+py**2)
      xpt=tmp
      return
      end


      function p3(en,px,py,pz)
      implicit real*8(a-h,o-z)
c
      tmp=sqrt(px**2+py**2+pz**2)
      p3=tmp
      return
      end


      function angle(en1,px1,py1,pz1,en2,px2,py2,pz2)
      implicit real*8(a-h,o-z)
c
      xl1=p3(en1,px1,py1,pz1)
      xl2=p3(en2,px2,py2,pz2)
      if(xl1.eq.0.or.xl2.eq.0)then
        angle=-1.d8
        return
      endif
      c12=px1*px2+py1*py2+pz1*pz2
      c12=c12/(xl1*xl2)
      tmp=acos(c12)
      angle=tmp
      return
      end


      subroutine initplot
      implicit real*4(a-h,o-z)
      character*4 cc(2)
      data cc/' sdk',' prb'/
c 
      pi=acos(-1.d0)
      call inihist 
c
      k=0
      call mbook(k+ 1,'pt'//' #ev',2.e0,0.e0,200.e0)
      call mbook(k+ 2,'energy'//' #ev',2.e0,0.e0,200.e0)
      call mbook(k+ 3,'theta1'//' #ev',pi/50.e0,0.e0,pi)
      call mbook(k+ 4,'theta2'//' #ev',pi/50.e0,0.e0,pi)
      k=4
      call mbook(k+ 1,'log10[pt]'//' #ev',0.05e0,-3.e0,2.e0)
      call mbook(k+ 2,'log10[energy]'//' #ev',0.05e0,-3.e0,2.e0)
      call mbook(k+ 3,'log10[theta1]'//' #ev',0.05e0,-3.e0,0.2e0)
      call mbook(k+ 4,'log10[theta2]'//' #ev',0.05e0,-3.e0,0.2e0)
c
      do i=1,3
      do j=1,2
      k=8+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'energy'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 3,'theta1'//cc(j),pi/50.e0,0.e0,pi)
      call mbook(k+ 4,'theta2'//cc(j),pi/50.e0,0.e0,pi)
      k=12+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'log10[pt]'//cc(j),0.05e0,-3.e0,2.e0)
      call mbook(k+ 2,'log10[energy]'//cc(j),0.05e0,-3.e0,2.e0)
      call mbook(k+ 3,'log10[theta1]'//cc(j),0.05e0,-3.e0,0.2e0)
      call mbook(k+ 4,'log10[theta2]'//cc(j),0.05e0,-3.e0,0.2e0)
      enddo
      enddo
c
      return
      end


      subroutine topout
      implicit real*4(a-h,o-z)
      parameter (szero=0.e0)
      parameter (sone=1.e0)
c
      i=1
      do j=1,2
      k=8+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      k=12+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      enddo
      do i=1,100
        call mfinal(i)
      enddo
      do i=1,8
        call mnorm(i,sone)
      enddo
c
      k=0
      call multitop(k+ 1,99,2,2,'pt',' ','LOG')
      call multitop(k+ 2,99,2,2,'energy',' ','LOG')
      call multitop(k+ 3,99,2,2,'theta1',' ','LOG')
      call multitop(k+ 4,99,2,2,'theta2',' ','LOG')
c
      i=2
      do j=1,2
      k=8+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'pt',' ','LOG')
      call multitop(k+ 2,99,2,2,'energy',' ','LOG')
      call multitop(k+ 3,99,2,2,'theta1',' ','LOG')
      call multitop(k+ 4,99,2,2,'theta2',' ','LOG')
      enddo
c
      k=4
      call multitop(k+ 1,99,2,2,'log10[pt]',' ','LOG')
      call multitop(k+ 2,99,2,2,'log10[energy]',' ','LOG')
      call multitop(k+ 3,99,2,2,'log10[theta1]',' ','LOG')
      call multitop(k+ 4,99,2,2,'log10[theta2]',' ','LOG')
c
      do j=1,2
      k=12+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'log10[pt]',' ','LOG')
      call multitop(k+ 2,99,2,2,'log10[energy]',' ','LOG')
      call multitop(k+ 3,99,2,2,'log10[theta1]',' ','LOG')
      call multitop(k+ 4,99,2,2,'log10[theta2]',' ','LOG')
      enddo                
      end


      subroutine outfun(p,sud,bogus)
      implicit real*8(a-h,o-z)
      real*4 one
      parameter (pi=3.1415926535897932385d0)
      parameter (one=1.e0)
      parameter (nexternal=5)
      parameter (ione=1)
      parameter (itwo=2)
      real*8 p(0:3,nexternal)
c
      pt=xpt(p(0,nexternal),p(1,nexternal),
     #       p(2,nexternal),p(3,nexternal))
      th1=angle(p(0,nexternal),p(1,nexternal),
     #          p(2,nexternal),p(3,nexternal),
     #          p(0,ione),p(1,ione),
     #          p(2,ione),p(3,ione))
      th2=angle(p(0,nexternal),p(1,nexternal),
     #          p(2,nexternal),p(3,nexternal),
     #          p(0,itwo),p(1,itwo),
     #          p(2,itwo),p(3,itwo))
      energy=p(0,nexternal)
c
      kk=0
      call mfill(kk+1,sngl(pt),one)
      call mfill(kk+2,sngl(energy),one)
      call mfill(kk+3,sngl(th1),one)
      call mfill(kk+4,sngl(th2),one)
      kk=4
      if(pt.gt.0.d0)call mfill(kk+1,sngl(log10(pt)),one)
      if(energy.gt.0.d0)call mfill(kk+2,sngl(log10(energy)),one)
      xth1=-1.d8
      if(th1.lt.pi/2.d0)then
        if(th1.gt.0.d0)xth1=log10(th1)
      else
        if((pi-th1).gt.0.d0)xth1=log10(pi-th1)
      endif
      call mfill(kk+3,sngl(xth1),one)
      xth2=-1.d8
      if(th2.lt.pi/2.d0)then
        if(th2.gt.0.d0)xth2=log10(th2)
      else
        if((pi-th2).gt.0.d0)xth2=log10(pi-th2)
      endif
      call mfill(kk+4,sngl(xth2),one)
      do i=1,3
        ipow=i-1
        kk=8+(i-1)*16
        call mfill(kk+1,sngl(pt),one*sngl(sud**ipow))
        call mfill(kk+2,sngl(energy),one*sngl(sud**ipow))
        call mfill(kk+3,sngl(th1),one*sngl(sud**ipow))
        call mfill(kk+4,sngl(th2),one*sngl(sud**ipow))
c
        kk=12+(i-1)*16
        if(pt.gt.0.d0)
     #    call mfill(kk+1,sngl(log10(pt)),one*sngl(sud**ipow))
        if(energy.gt.0.d0)
     #    call mfill(kk+2,sngl(log10(energy)),one*sngl(sud**ipow))
        call mfill(kk+3,sngl(xth1),one*sngl(sud**ipow))
        call mfill(kk+4,sngl(xth2),one*sngl(sud**ipow))
c
        kk=16+(i-1)*16
        call mfill(kk+1,sngl(pt),one*sngl(bogus**ipow))
        call mfill(kk+2,sngl(energy),one*sngl(bogus**ipow))
        call mfill(kk+3,sngl(th1),one*sngl(bogus**ipow))
        call mfill(kk+4,sngl(th2),one*sngl(bogus**ipow))
c
        kk=20+(i-1)*16
        if(pt.gt.0.d0)
     #    call mfill(kk+1,sngl(log10(pt)),one*sngl(bogus**ipow))
        if(energy.gt.0.d0)
     #    call mfill(kk+2,sngl(log10(energy)),one*sngl(bogus**ipow))
        call mfill(kk+3,sngl(xth1),one*sngl(bogus**ipow))
        call mfill(kk+4,sngl(xth2),one*sngl(bogus**ipow))
      enddo
c
      return
      end


      subroutine fk88strnum(string,num)
c- writes the number num on the string string starting at the blank
c- following the last non-blank character
      character * (*) string
      character * 20 tmp
      l = len(string)
      write(tmp,'(i15)')num
      j=1
      dowhile(tmp(j:j).eq.' ')
        j=j+1
      enddo
      ipos = ifk88istrl(string)
      ito = ipos+1+(15-j)
      if(ito.gt.l) then
         write(*,*)'error, string too short'
         write(*,*) string
         stop
      endif
      string(ipos+1:ito)=tmp(j:)
      end


      function ifk88istrl(string)
c returns the position of the last non-blank character in string
      character * (*) string
      i = len(string)
      dowhile(i.gt.0.and.string(i:i).eq.' ')
         i=i-1
      enddo
      ifk88istrl = i
      end

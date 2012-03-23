c Sums plots from input files. This code is obtained from read11
      implicit real*4 (a-h,o-z)
      logical flag,titfnd
      integer ndata(1:40),ioff(1:40)
      real * 8 xfacts(1:40),step(1:40)
      real * 8 xv(1:40,1:1000),yv(1:40,1:1000),er(1:40,1:1000)
      real * 8 xa(1:1000),ya(1:1000),ea(1:1000)
      character * 70 file,file1,filename(1:40)
      character * 70 line,vartype,vartype2
      character * 70 strscale,strlimx,strlimy,strtit
      character * 11 chref1,chref7
      character * 14 chref2,chref3
      character * 18 chref4,chref6
      character * 7 chref5,chref8
      character * 9 chref9,chref10
      character * 10 chref11
      integer iverbose
      parameter (iverbose=0)
c
999   format(3(d16.6))
      chref1='  SET SCALE '
      chref2='  SET LIMITS X'
      chref3='  SET LIMITS Y'
      chref4='  SET ORDER X Y DY'
      chref5='   PLOT'
      chref6='  ( SET FONT DUPLE'
      chref7='   NEW PLOT'
      chref8='  HIST '
      chref9=' new plot'
      chref10=' (       '
      chref11='  TITLE   '
c
      strtit(1:17)='  TITLE    TOP   '
      ifound=-1
      open(unit=88,status='unknown',file='fort.88',form='formatted')
      open(unit=99,status='unknown',file='read40.out',form='formatted')
      if(iverbose.eq.1)
     #  write(6,*)'enter number of files (less than 40)'
      read(5,*)nfil
      if(nfil.ge.100)stop
      if(iverbose.eq.1)write(6,*)'enter filenames'
      read(5,'(a)')file
      filename(1)=file
      write(99,*)nfil
      write(99,*)"10 ",file
      open(unit=10,status='old',file=file,form='formatted')
      do i=1,nfil-1
        iunit=10+i
        read(5,'(a)')file1
        filename(i+1)=file1
        write(99,*)i," ",file1
        open(unit=iunit,status='old',file=file1,form='formatted')
      enddo
c Multiplicative factors may be entered here if need be
      do i=1,nfil
        xfacts(i)=1.d0
      enddo
c
      iend=0
      imtfill=0
      vartype2='&&&&&&&&'
 111  continue
      titfnd=.false.
      write(88,'(a)')' ('
      write(88,'(a)')' new plot'
      write(88,'(a)')' ('
      flag=.true.
      do j=1,1000
        do i=1,nfil
          ndata(i)=0
          xv(i,j)=-1.d8
          yv(i,j)=0.d0
          er(i,j)=0.d0
        enddo
        ya(j)=0.d0
        ea(j)=0.d0
      enddo
      do while(flag)
        if(imtfill.eq.1)then
          flag=.false.
          write(88,'(a)')strtit
          write(88,'(a)')strscale
          write(88,'(a)')strlimx
          write(88,'(a)')strlimy
          vartype=vartype2
          write(88,'(a)')vartype
          read(10,'(a)',err=200,end=200)line
          if(line(1:7).ne.' ( INT=')then
            write(99,*)'Bad format in input file'
            write(99,*)'READ40 ERROR READ40 ERROR'
            stop
          else
            write(88,'(a)')line
          endif
        else
          read(10,'(a)',err=200,end=200)line
          if(line(1:10) .eq. chref11 .and. .not.titfnd) then
             istrt=index(line,'"')
             istp =index(line(istrt+1:70),'"')
             if (istrt.ne.0 .and. istp.ne.0) then
                write(strtit(18:18+istp),'(a)')
     &               line(istrt:istp+istrt)
                do i=19+istp,70
                   write(strtit(i:i),'(a)')' '
                enddo
                write(88,'(a)')strtit
                titfnd=.true.
             endif
          endif
          if(line(1:11).eq.chref1)then
            write(88,'(a)')line
            strscale=line
          elseif(line(1:14).eq.chref2)then
            write(88,'(a)')line
            strlimx=line
c RF: comment out the limits for y-axes
          elseif(line(2:14).eq.chref3(2:14))then
            write (line(1:1),'(a)') '('
            write(88,'(a)')line
            strlimy=line
          elseif(line(1:18).eq.chref4)then
            flag=.false.
            write(88,'(a)')line
            read(10,'(a)',err=200,end=200)line
            vartype=line
            write(88,'(a)')line
            read(10,'(a)',err=200,end=200)line
            if(line(1:7).ne.' ( INT=')then
              write(99,*)'Bad format in input file'
              write(99,*)'READ40 ERROR READ40 ERROR'
              stop
            else
              write(88,'(a)')line
            endif
          else
            continue
          endif
        endif
      enddo
      write(99,*)"Doing ",vartype
c now read the numbers
 222  continue
      do jj=1,100000
        read(10,'(a)',err=200,end=200)line
        read(unit=line,fmt=*,err=10)x,y,dy
        ndata(1)=ndata(1)+1
        xv(1,jj)=x
        yv(1,jj)=y
        er(1,jj)=dy
      enddo
 10   continue
      if(line(1:7).ne.chref8)then
        write(99,*)'No HIST statement at the end of the histogram'
        write(99,*)'READ40 ERROR READ40 ERROR'
        stop
      endif
      read(10,'(a)',err=200,end=200)line
      if(line(1:7).ne.chref5)then
        write(99,*)'No PLOT statement at the end of the histogram'
        write(99,*)'READ40 ERROR READ40 ERROR'
        stop
      endif
      read(10,'(a)',err=200,end=333)line
      if(line(1:18).eq.chref6.or.line(1:11).eq.chref7.or.
     #   line(1:9).eq.chref9.or.line(1:9).eq.chref10)then
        imtfill=0
        vartype2='&&&&&&&&'
      else
        imtfill=1
        vartype2=line
      endif
      goto 444
 333  continue
      iend=1
 444  continue
      write(99,*)"0 done ",filename(1)
c now the other files
      do i=1,nfil-1
        iunit=10+i
        flag=.true.
        ifound=0
        do while(flag)
          read(iunit,'(a)',err=200,end=200)line
          if(line.eq.vartype)then
            flag=.false.
            ifound=1
            read(iunit,'(a)',err=200,end=200)line
            if(line(1:7).ne.' ( INT=')then
              write(99,*)'Bad format in input file'
              write(99,*)'READ40 ERROR READ40 ERROR'
              stop
            endif
          else
            continue
          endif
        enddo
c now read the numbers
        do jj=1,100000
          read(iunit,'(a)',err=200,end=200)line
          read(unit=line,fmt=*,err=11)x,y,dy
          ndata(i+1)=ndata(i+1)+1
          xv(i+1,jj)=x
          yv(i+1,jj)=y
          er(i+1,jj)=dy
        enddo
 11     continue
        if(line(1:7).ne.chref8)then
          write(99,*)'No HIST statement at the end of the histogram'
          write(99,*)'READ40 ERROR READ40 ERROR'
          stop
        endif
        if(iend.eq.0)then
          read(iunit,'(a)',err=200,end=200)line
          if(line(1:7).ne.chref5)then
            write(99,*)'No PLOT statement at the end of the histogram'
            write(99,*)'READ40 ERROR READ40 ERROR'
            stop
          endif
        endif
        write(99,*)i," done ",filename(iunit-9)
      enddo
c
      xstep=1.d10
      xmin=1.d10
      xmax=-1.d10
      do i=1,nfil
        if(ndata(i).ge.2)then
          xmin=min(xmin,xv(i,1),xv(i,2))
          xmax=max(xmax,xv(i,1),xv(i,2))
          step(i)=xv(i,2)-xv(i,1)
          do j=3,ndata(i)
            tmp=xv(i,j)-xv(i,j-1)
            if(abs(tmp-step(i)).lt.1.d-2*step(i))then
              step(i)=((j-2)*step(i)+tmp)/dfloat(j-1)
            elseif(tmp.lt.step(i)*0.97d0)then
              step(i)=tmp
            elseif( tmp.gt.step(i)*0.97d0 .and.
     #              tmp.lt.step(i)*1.03d0 )then
              step(i)=(step(i)+tmp)/2.d0
            endif
            if(xv(i,j).lt.xmin)xmin=xv(i,j)
            if(xv(i,j).gt.xmax)xmax=xv(i,j)
          enddo
        else
          xmin=xv(i,1)
          xmax=xv(i,1)
          step(i)=1.d-10
        endif
        if(step(i).lt.xstep)xstep=step(i)
      enddo
      do i=1,nfil
        if(abs(step(i)-xstep).gt.1.d-3*xstep)then
          write(99,*)'mismatch in input',xstep,step(i)
          write(99,*)'in file: ',
     #      filename(i)(1:istrl(filename(i)))
          write(99,*)'continue with the next plot'
          write(99,*)'READ40 ERROR READ40 ERROR'
          goto 111
        endif
      enddo
      nn=nint((xmax-xmin)/xstep)
      if(abs(xmax-xmin-nn*xstep)/dfloat(nn).gt.1.d-2*xstep)then
        write(99,*)'Fatal error:',xmax,xmin,xstep,nn
        write(99,*)'READ40 ERROR READ40 ERROR'
        stop
      endif
      do i=1,nfil
        ioff(i)=1
      enddo
      xpt=xmin-xstep
      do j=0,nn
        xpt=xpt+xstep
        xa(j+1)=xpt
        do i=1,nfil
c$$$          if(i.eq.1)print*,xpt,xv(i,j+1),xv(i,ioff(i))
          if(abs(xv(i,ioff(i))-xpt).lt.3.d-2*xstep)then
            xa(j+1)=xv(i,ioff(i))
            ya(j+1)=ya(j+1)+yv(i,ioff(i))*xfacts(i)
            ea(j+1)=ea(j+1)+( er(i,ioff(i))*xfacts(i) )**2
            ioff(i)=ioff(i)+1
            xpt=xa(j+1)
          endif
        enddo
      enddo
      do i=1,nfil
        if(ndata(i)+1.ne.ioff(i))then
          write(99,*)'ioff is wrong:',ndata(i),ioff(i),i,xstep
          write(99,*)'READ40 ERROR READ40 ERROR'
          stop
        endif
      enddo
      do j=0,nn
        write(88,999)xa(j+1),ya(j+1),sqrt(ea(j+1))
      enddo
      write(88,'(a)')'  HIST SOLID'
      write(88,'(a)')'   PLOT'
      write(99,*)vartype," done"
      if(iend.eq.0)goto 111
200   continue
      if( (nfil.eq.1.and.ifound.ne.-1) .or.
     #    (nfil.gt.1.and.ifound.eq.0) )then
        write(6,*)'Error: plot not found '
        write(6,*)' ID String: ',vartype(1:istrl(vartype))
        write(6,*)' In file: ',
     #    filename(iunit-9)(1:istrl(filename(iunit-9)))
        write(99,*)'Error: plot not found '
        write(99,*)' ID String: ',vartype(1:istrl(vartype))
        write(99,*)' In file: ',
     #    filename(iunit-9)(1:istrl(filename(iunit-9)))
        write(99,*)'READ40 ERROR READ40 ERROR'
        stop
      endif
      close(88)
      close(99)
      do i=0,nfil-1
        iunit=10+i
        close(iunit)
      enddo
      end


      function istrl(string)
c returns the position of the last non-blank character in string
      character * (*) string
      i = len(string)
      dowhile(i.gt.0.and.string(i:i).eq.' ')
         i=i-1
      enddo
      istrl = i
      end


      subroutine strcat(str1,str2,str)
c concatenates str1 and str2 into str. Ignores trailing blanks of str1,str2
      character *(*) str1,str2,str
      l1=istrl(str1)
      l2=istrl(str2)
      l =len(str)
      if(l.lt.l1+l2) then
          write(*,*) 'error: l1+l2>l in strcat'
          write(*,*) 'l1=',l1,' str1=',str1
          write(*,*) 'l2=',l2,' str2=',str2
          write(*,*) 'l=',l
          stop
      endif
      if(l1.ne.0) str(1:l1)=str1(1:l1)
      if(l2.ne.0) str(l1+1:l1+l2)=str2(1:l2)
      if(l1+l2+1.le.l) str(l1+l2+1:l)= ' '
      end

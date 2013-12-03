c Identical to read7, except for the fact that the cross sections can
c be multiplied by factors entered in input
      implicit real*4 (a-h,o-z)
      real * 4 xfacts(1:4)
      logical flag,titfnd
      character * 70 file,file1,filename(1:4)
      character * 70 line,vartype,vartype2
      character * 70 strscale,strlimx,strlimy,strtit
      character * 11 chref1,chref7
      character * 14 chref2,chref3
      character * 18 chref4,chref6
      character * 7 chref5,chref8
      character * 9 chref9,chref10
      character * 10 chref11
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
      write(6,*)'enter number of files (less than 5)'
      read(5,*)nfil
      if(nfil.ge.5)stop
      write(6,*)'enter filenames'
      read(5,'(a)')file
      filename(1)=file
      open(unit=50,type='old',name=file,form='formatted')
      do i=1,nfil-1
        iunit=50+i
        read(5,'(a)')file1
        filename(i+1)=file1
        open(unit=iunit,type='old',name=file1,form='formatted')
      enddo
      write(6,*)'enter multiplicative factors'
      do i=1,nfil
        read(5,*)xfacts(i)
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
      do while(flag)
        if(imtfill.eq.1)then
          flag=.false.
          write(88,'(a)')strtit
          write(88,'(a)')strscale
          write(88,'(a)')strlimx
          write(88,'(a)')strlimy
          vartype=vartype2
          write(88,'(a)')vartype
          read(50,'(a)',err=200,end=200)line
          if(line(1:7).ne.' ( INT=')then
            write(6,*)'Bad format in input file'
            stop
          else
            write(88,'(a)')line
          endif
        else
          read(50,'(a)',err=200,end=200)line
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
          elseif(line(1:14).eq.chref3)then
            write(88,'(a)')line
            strlimy=line
          elseif(line(1:18).eq.chref4)then
            flag=.false.
            write(88,'(a)')line
            read(50,'(a)',err=200,end=200)line
            vartype=line
            write(88,'(a)')line
            read(50,'(a)',err=200,end=200)line
            if(line(1:7).ne.' ( INT=')then
              write(6,*)'Bad format in input file'
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
        read(50,'(a)',err=200,end=200)line
        read(unit=line,fmt=*,err=10)x,y,dy
        write(88,999)x,y*xfacts(1),dy
      enddo
 10   continue
      if(line(1:7).ne.chref8)then
        write(6,*)'No HIST statement at the end of the histogram'
        stop
      else
        write(88,'(a)')'  hist solid'
      endif
      read(50,'(a)',err=200,end=200)line
      if(line(1:7).ne.chref5)then
        write(6,*)'No PLOT statement at the end of the histogram'
        stop
      endif
      read(50,'(a)',err=200,end=333)line
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
c now the other files
      do i=1,nfil-1
        iunit=50+i
        flag=.true.
        do while(flag)
          read(iunit,'(a)',err=200,end=200)line
          if(line.eq.vartype)then
            flag=.false.
            write(88,'(a)')line
            read(iunit,'(a)',err=200,end=200)line
            if(line(1:7).ne.' ( INT=')then
              write(6,*)'Bad format in input file'
              stop
            else
              write(88,'(a)')line
            endif
          else
            continue
          endif
        enddo
c now read the numbers
        do jj=1,100000
          read(iunit,'(a)',err=200,end=200)line
          read(unit=line,fmt=*,err=11)x,y,dy
          write(88,999)x,y*xfacts(i+1),dy
        enddo
 11     continue
        if(line(1:7).ne.chref8)then
          write(6,*)'No HIST statement at the end of the histogram'
          stop
        else
          if(i.eq.1)then
            write(88,'(a)')' set pattern .05 .07'
          elseif(i.eq.2)then
            write(88,'(a)')' set pattern .01 .07'
          elseif(i.eq.3)then
            write(88,'(a)')' set pattern .01 .07 .05 .07'
          else
            write(6,*)'Fatal error'
            stop
          endif
          write(88,'(a)')' hist patterned'
        endif
        if(iend.eq.0)then
          read(iunit,'(a)',err=200,end=200)line
          if(line(1:7).ne.chref5)then
            write(6,*)'No PLOT statement at the end of the histogram'
            stop
          endif
        endif
      enddo
c
      if(iend.eq.0)goto 111
200   continue
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

c Identical to vegas.for, except for the fact that a larger number of
c integration variables (58 rather than 10) is allowed here
c
c
c-
c Integration-histogramming package using vegas.
c Calls:
c      subroutine delete(fname)         ! deletes the file fname
c      character * (*) fname
c
c      subroutine time(ctime)           ! returns a string with the time
c      character * 8 ctime
c
c      subroutine date(cdate)           ! returns a string with the date
c      character * 9 cdate
c

      subroutine strnum(string,num)
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
      ipos = istrl(string)
      ito = ipos+1+(15-j)
      if(ito.gt.l) then
         write(*,*)'error, string too short'
         write(*,*) string
         stop
      endif
      string(ipos+1:ito)=tmp(j:)
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

      block data skiphistos
c If integrate will be allowed to save and restore histograms
c (the default mode of operation till Nov 2013), then set:
c  skiph=.false.
c here. Otherwise, if skiph=.true., no information on the histograms
c will be saved in the .sv1 and .sv2 by the routine integrate()
      implicit none
      logical skiph
      common/cskiph/skiph
      data skiph/.true./
      end

      subroutine
     # integrate(init,sig,str,n,inew,idim,icalls,avgi,sd,chi2a,isave)
      implicit real * 8 (a-h,o-z)
      external init,sig
      common/bveg1/xl(58),xu(58),acc,ndim,ncall,itmx,nprn
      common/bveg2/xi(50,58),si,si2,swgt,schi,ndo,it
      common/newver/newver
      character * (*) str
      character * 7 statf,newver
      character * 70 fname
      logical isave
c- if inew = 0 ignore
      if(inew.eq.0) return
c- add str to list of save files
      if(isave) then
         fname = str
         call addfil(fname)
      endif
c- initialize histograms
      call init
c
      if(inew.ge.2) then
         call resume(str,init,'IN')
c also fixes the value of it
         call vegas4(sig,avgi,sd,chi2a)
      endif
c
      if(inew.eq.10) then
c- save the file '.sv2' in ASCII format
         call strcat(str,'.sv2',fname)
c- ndim = idim for upward compatibility with old restart files
         ndim = idim
         call savea(fname,newver)
         return
      endif
c
      if(inew.eq.11) then
c- save the file '.sv1' in standard format
         call strcat(str,'.sv1',fname)
         call save(fname,newver)
         return
      endif
c
c Initialize parameters for Vegas
      do j=1,58
          xl(j) = 0
          xu(j) = 1
      enddo
      acc = -1
      ndim = idim
      ncall = icalls
      itmx = 1
c
c for negative n abs(n) is the number of iterations to reach,
c for positive n, n is the number of iterations to perform
      if(n.lt.0) then
         if(inew.eq.1.or.inew.eq.3) then
            nsteps = abs(n)
         elseif(inew.eq.2) then
            nsteps = abs(n) - it
         endif
      else
         nsteps = n
      endif
c create new versions of save file
cRF
c$$$      statf = newver
      statf = 'UNKNOWN'
c----------------------------------------------------------
      do j=1,nsteps
         if(j.eq.1) then
            if(inew.eq.1) then
                  call vegas(sig,avgi,sd,chi2a)
            elseif(inew.eq.2) then
                  call vegas2(sig,avgi,sd,chi2a)
            elseif(inew.eq.3) then
                  call init
                  call vegas1(sig,avgi,sd,chi2a)
            else
                  write(*,*) 'irregular inew flag ',inew
                  stop
            endif
         else
            call vegas3(sig,avgi,sd,chi2a)
         endif
         call accum
         if(isave) then
            call strcat(str,'.sv2',fname)
            call save(fname,statf)
            call strcat(str,'.sv1',fname)
            call save(fname,statf)
            call strcat(str,'.sv2',fname)
            call delete(fname)
            statf = 'UNKNOWN'
         endif
      enddo
      return
      entry integrms1
c # of iterations message
      write(*,*)
      write(*,*)'if you enter a number of iterations n>0'
      write(*,*)'the program will execute n iterations'
      write(*,*)'if n<0, it will execute iterations until'
      write(*,*)'total number of iterations performed = abs(n)'
      write(*,*)'(it makes a difference only if in restart mode)'
      write(*,*)
      return
      entry integrms2
c inew values
      write(*,*)
      write(*,*) 'enter 0 to exclude,1 for new run, 2 to restart'
      write(*,*) '3 to restart keeping the grid but reset histo''s '
      write(*,*) '10 to produce ASCII save files from standard ones'
      write(*,*) '11 to produce standard save files from ASCII ones'
      write(*,*) '(10/11 used to transport save files across machines)'
      write(*,*)
      return
      entry nopr(nnn)
      nprn = nnn
      end

      subroutine resume(string,init,fl)
      character * (*) string, fl *2
      character * 70 fname
      if(fl.eq.'IN') call init
      call strcat(string,'.sv1',fname)
      call restart(fname,itmp)
      if(itmp.eq.1.and.fl.eq.'NO') goto 99
      if(itmp.eq.0) return
c- retry with spare save file
      if(fl.eq.'IN') call init
      call strcat(string,'.sv2',fname)
      call restart(fname,itmp)
      if(itmp.eq.1.and.fl.eq.'NO') goto 99
      if(itmp.eq.0) return
c- retry with ASCII spare save file
      if(fl.eq.'IN') call init
      call strcat(string,'.sv2',fname)
      call restara(fname,itmp)
      if(itmp.eq.1.and.fl.eq.'NO') goto 99
      if(itmp.eq.0) return
c- give up.
      write(*,*) 'unusable restart file'
      stop
99    continue
c In this case the accumulated values in the histograms common block
c could have been altered in the attempt to read the save file.
c Abort. (if itmp=2 this is not the case)
      write(*,*) 'corrupted file',fname,'.'
      write(*,'(a,/)')
     # 'The attempt to read the corrupted file may have corrupted',
     # 'the histogram common block. Delete the corrupted file, or',
     # 'rerun one more iteration, or run with nit=11.'
      stop
      end

      subroutine savea(name,statf)
      include "dbook.inc"
      PARAMETER (NMB=NPLOTS)
      character * 80 runstr,string
      character * 3 books(nmb)
      real * 8 xl,xu,acc
      common/bveg1/xl(58),xu(58),acc,ndim,ncall,itmx,nprn
      real * 8 xi,si,si2,swgt,schi
      common/bveg2/xi(50,58),si,si2,swgt,schi,ndo,it

      common/runstr/runstr
      character * (*) name, statf
      logical skiph
      common/cskiph/skiph
c
c Salva gli istogrammi in uso (marcati 'YES' in book(j))
c e tutte le informazioni per vegas, il seme del numero
c casuale, etc.
c Marchia la fine del file con i numeri acos(-1.),exp(1.),log(10.),
c
      open(unit=97,file=name,status=statf)
101   format(a)
102   format(25a)
103   format(6d13.7)
104   format(6i12)
105   format(4d20.14)
c version of save file
      string = 'SAVEA,VER 8-3-92'
      write(97,101) string
c runstring
      write(97,101)  runstr
c booked histograms
      if(.not.skiph)then
      write(97,104)  nmb
      write(97,102)  (book(j),j=1,nmb)
c Mbook block
      do j=1,nmb
         if(book(j).eq.'YES') then
            write(97,103)
     &      (hist(j,k),k=1,nbin(j)),uscore(j),oscore(j)
            write(97,104)
     &      (ihis(j,k),k=1,nbin(j)),iuscore(j),ioscore(j),ient(j)
         endif
      enddo
      endif
c Vegas block
      write(97,104)
     & ndo,it,num1,num2,ndim
      write(97,105)
     & ((xi(j,k),j=1,ndo),k=1,ndim),si,si2,swgt,schi

      a=acos(-1.)
      b=exp(1.)
      c=log(10.)
c end mark
      write(97,103)
     & a,b,c
      close(97)
      return
c
c Loads the histograms found in the (ascii) save file into the common block.
c Previous values are overwritten.
c Does not modify histograms not present in the file.
c If all goes well, it riturns iflag= 0.
c If there are errors (e.g. IO errors, the mark at the end of the file
c is not found, etc.) and in the input operation the common blocks
c may have been corrupted, returns iflag=1.
c If there are errors, but no common block has been altered, returns iflag=2.
c
      entry restara(name,iflag)
      open(unit=97,file=name,status='OLD',err=2)
c version of save file
      string = ' '
      read(97,101,err=2,end=2) string
      if(string.ne.'SAVEA,VER 8-3-92') goto 2
c run string
      string = ' '
      read(97,101,err=2,end=2) string
c marked histograms
      if(.not.skiph)then
      read(unit=97,fmt=104,err=2,end=2) nhs
      if(nhs.gt.nmb) then
         write(*,*) 'Error: save files had nmb=',nhs
         write(*,*) 'Running program has nmb=',nmb,'<',nhs
         write(*,*) 'Make nmb >= ',nhs,' in running program'
         stop
      endif
      read(unit=97,fmt=102,err=1,end=1) (books(j),j=1,nhs)
c Mbook block
      do j=1,nhs
         if(books(j).eq.'YES') then
            read(unit=97,fmt=103,err=1,end=1)
     &      (hist(j,k),k=1,nbin(j)),uscore(j),oscore(j)
            read(unit=97,fmt=104,err=1,end=1)
     &      (ihis(j,k),k=1,nbin(j)),iuscore(j),ioscore(j),ient(j)
         endif
      enddo
      endif
c vegas block
      read(unit=97,fmt=104,err=1,end=1)
     & ndo,it,num1,num2,ndim
      read(unit=97,fmt=105,err=1,end=1)
     & ((xi(J,k),j=1,ndo),k=1,ndim),si,si2,swgt,schi
c end mark
      read(unit=97,fmt=103,err=1,end=1)
     & a0,b0,c0
c
      tiny = .1e-5
      a = abs(a0-acos(-1.))
      b = abs(b0-exp(1.))
      c = abs(c0-log(10.))
      if( a.gt.tiny .or. b.gt.tiny .or. c.gt.tiny ) goto 1
      close(unit=97,iostat=ios)
      iflag = 0
      if(string.ne.runstr) then
         write(6,*)
     #  'You are using save files belonging to a different run,'
         write(6,*) string
         write(6,*) 'instead of'
         write(6,*) runstr
      endif
      write(6,*) string
      return
 1    continue
      iflag = 1
      goto 3
 2    continue
      iflag = 2
      goto 3
 3    continue
      close(unit=97,iostat=ios)
      end

      subroutine save(name,statf)
      include "dbook.inc"
      PARAMETER (NMB=NPLOTS)
      character * 80 runstr,string
      character * 3 books(nmb)
      real * 8 xl,xu,acc
      common/bveg1/xl(58),xu(58),acc,ndim,ncall,itmx,nprn
      real * 8 xi,si,si2,swgt,schi
      common/bveg2/xi(50,58),si,si2,swgt,schi,ndo,it

      common/runstr/runstr
      character * (*) name, statf
      logical skiph
      common/cskiph/skiph
c
c Salva gli istogrammi in uso (marcati 'YES' in book(j))
c e tutte le informazioni per vegas, il seme del numero
c casuale, etc.
c Marchia la fine del file con i numeri acos(-1.),exp(1.),log(10.),
c
      open(unit=97,file=name,form='UNFORMATTED',status=statf)
c version of save program
      string = 'SAVE,VER 8-3-92'
      write(97) string
c run string
      write(97)  runstr
c Mbook Block
      if(.not.skiph)then
      write(97) nmb
      write(97)  (book(j),j=1,nmb)
      do j=1,nmb
         if(book(j).eq.'YES') then
           write(97)
     &     (hist(j,k),k=1,nbin(j))
     &     ,uscore(j),oscore(j)
     &     ,(ihis(j,k),k=1,nbin(j))
     &     ,iuscore(j),ioscore(j)
     &     ,ient(j)
         endif
      enddo
      endif
c Vegas Block
      write(97)
     & ndo,it,num1,num2,ndim
      write(97)
     & ((xi(J,k),j=1,ndo),k=1,ndim),si,si2,swgt,schi
c
      a=acos(-1.)
      b=exp(1.)
      c=log(10.)
      write(97) a,b,c
      close(97)
      return
c
c Loads the histograms found in the file into the common block.
c Previous values are overwritten.
c Does not modify histograms not present in the file.
c If all goes well, it riturns iflag= 0.
c If there are errors (e.g. IO errors, the mark at the end of the file
c is not found, etc.) and in the input operation the common blocks
c may have been corrupted, returns iflag=1.
c If there are errors, but no common block has been altered, returns iflag=2.
c
      entry restart(name,iflag)
      open(unit=97,file=name,form='unformatted',status='OLD',err=2)
c version of save file
      string = ' '
      read(unit=97,err=2,end=2) string
      if(string.ne.'SAVE,VER 8-3-92') goto 2
c run string
      string = ' '
      read(unit=97,err=2,end=2) string
c Mbook block
      if(.not.skiph)then
      read(97,err=2,end=2) nhs
      if(nhs.gt.nmb) then
         write(*,*) 'Error: save files had nmb=',nhs
         write(*,*) 'Running program has nmb=',nmb,'<',nhs
         write(*,*) 'Make nmb >= ',nhs,' in running program'
         stop
      endif
      read(97,err=1,end=1) (books(j),j=1,nhs)
      do j=1,nhs
         if(books(j).eq.'YES') then
            read(97,err=1,end=1)
     &      (hist(j,k),k=1,nbin(j))
     &      ,uscore(j),oscore(j)
     &      ,(ihis(j,k),k=1,nbin(j))
     &      ,iuscore(j),ioscore(j)
     &      ,ient(j)
         endif
      enddo
      endif
c Vegas Block
      read(97,err=1,end=1)
     & ndo,it,num1,num2,ndim
      read(97,err=1,end=1)
     & ((xi(J,k),j=1,ndo),k=1,ndim),si,si2,swgt,schi
c
      a0 = 0
      b0 = 0
      c0 = 0
      read(97,err=1,end=1)   a0,b0,c0
      a=acos(-1.)
      b=exp(1.)
      c=log(10.)
      if( a.ne.a0 .or. b.ne.b0 .or. c.ne.c0 ) goto 1
      close(97)
      iflag = 0
      if(string.ne.runstr) then
         write(6,*)
     #   'You are using save files belonging to a different run,'
         write(6,*) string
         write(6,*) 'instead of'
         write(6,*) runstr
      endif
      write(6,*) string
      return
 1    continue
      iflag = 1
      goto 3
 2    continue
      iflag = 2
      goto 3
 3    continue
      close(unit=97,iostat=ios)
      end

       block data vegas0
       implicit  real*8 (a-h,o-z)
       common/bveg1/xl(58),xu(58),acc,ndim,ncall,itmx,nprn
       common/bveg2/xi(50,58),si,si2,swgt,schi,ndo,it
       data ncall/10000/,itmx/10/,nprn/ 1000/,acc/-1.d0/,
     1   xl/58*1.d-3/,
     2   xu/58*1.d0/
       data xi/2900*1.d0/
       end
C
C
C      NCALL IS THE NUMBER OF CALLS TO VEGAS.
C      NPRN >  0 VEGAS PRINTS THE RESULTS OF EACH ITERATION.
C      NPRN 0 VEGAS PRINTS NOTHING.
C      NPRN < 0 VEGAS PRINTS ALL.
C      XL(I) IS LOWER INTEGRATION LIMIT ON I TH AXIS.
C      XU(I) IS UPPER INTEGRATION LIMIT ON I THE AXIS.
c
         subroutine vegas(fxn,avgi,sd,chi2a)
c
c   routine performs n dim monte carlo inte
c written by p lepage
c
         implicit real*8 (a-h,o-z)
         implicit integer*4 (i-n)
         common/bveg1/xl(58),xu(58),acc,ndim,ncall,itmx,nprn
         common/bveg2/xi(50,58),si,si2,swgt,schi,ndo,it
         dimension d(50,58),di(50,58),xin(50),r(50),dx(58),dt(58),
     1   x(58),kg(58),ia(58)
       dimension RAND(58)
         data ndmx/50/,alph/1.5d0/,one/1.d0/,mds/1/
         ndo=1
         do 1 j=1,ndim
 1       xi(1,j)=one
c
         entry vegas1(fxn,avgi,sd,chi2a)
c       initialises  cumulative  variables but not grid
         it=0
         si=0.
         si2=si
         swgt=si
         schi=si
c
         entry vegas2(fxn,avgi,sd,chi2a)
c        no initialisation
         nd=ndmx
         ng=1
         if(mds.eq.0)go to 2
         ng=(ncall/2.)**(1./ndim)
         mds=1
         if((2*ng-ndmx).lt.0)go to 2
         mds=-1
         npg=ng/ndmx+1
         nd=ng/npg
         ng=npg*nd
 2       k=ng**ndim
         npg=ncall/k
         if(npg.lt.2)npg=2
         calls=npg*k
         dxg=one/ng
         dv2g=(calls*dxg**ndim)**2/npg/npg/(npg-one)
         xnd=nd
         ndm=nd-1
         dxg=dxg*xnd
         xjac=one/calls
         do 3 j=1,ndim
         dx(j)=xu(j)-xl(j)
 3       xjac=xjac*dx(j)
c
c    rebin preserving bin density
c
         if(nd.eq.ndo)go to 8
         rc=ndo/xnd
         do 7 J=1,ndim
         k=0
         xn=0.
         dr=xn
         i=k
 4       k=k+1
         dr=dr+one
         xo=xn
         xn=xi(k,j)
 5       if(rc.gt.dr)go to 4
         i=i+1
         dr=dr-rc
         xin(i)=xn-(xn-xo)*dr
         if(i.lt.ndm)go to 5
         do 6 i=1,ndm
 6       xi(i,j)=xin(i)
 7       xi(nd,j)=one
         ndo=nd
c
 8       if(nprn.ne.0)write(6,200)ndim,calls,it,itmx,acc
     1   ,mds,nd,(xl(j),xu(j),j=1,ndim)
c
         entry vegas3(fxn,avgi,sd,chi2a)
c         main integration loop
 9       it=it+1
          ti=0.
         tsi=ti
         do 10 j=1,ndim
         kg(j)=1
         do 10 i=1,nd
         d(i,j)=ti
 10      di(i,j)=ti
c
 11      fb=0.
         f2b=fb
         k=0
 12      k=k+1
       call randa(ndim,rand)
         wgt=xjac
         do 15 j=1,ndim
         xn=(kg(j)-rand(j))*dxg+one
         ia(j)=xn
         if(ia(j).gt.1)go to 13
         xo=xi(ia(j),j)
         rc=(xn-ia(j))*xo
         go to 14
13       xO=xi(ia(j),j)-xi(ia(j)-1,j)
         rc=xi(ia(j)-1,j)+(xn-ia(j))*xo
 14      x(j)=xl(j)+rc*dx(j)
 15      wgt=wgt*xo*xnd
c
         f=wgt
         f=f*fxn(x,wgt)
         f2=f*f
         fb=fb+f
         f2b=f2b+f2
         do 16 j=1,ndim
         di(ia(j),j)=di(ia(j),j)+f
 16      if(mds.ge.0)d(ia(j),J)=d(ia(j),J)+f2
         if(k.lt.npg) go to 12
c
888    FORMAT(1X,'F',G14.6,'F2',G14.6,'FB',G14.6,'F2B',G14.6)
         f2b= sqrt(f2b*      NPG)
         f2b=(f2b-fb)*(f2b+fb)
1661   FORMAT(1X,'F2B',G14.6,'NPG',  I10)
         ti=ti+fb
         tsi=tsi+f2b
33     FORMAT(1X,'TSI',G14.6,'F2B',G14.6)
         if(mds.ge.0)go to 18
         do 17 j=1,ndim
 17      d(ia(j),j)=d(ia(j),j)+f2b
 18      k=ndim
 19      kg(k)=mod(kg(k),ng)+1
         if(kg(k).ne.1)go to 11
         k=k-1
         if(k.gt.0)go to 19
c
c final results for this iteration
c
        tsi=tsi*dv2g
        ti2=ti*ti
88     format(1x,'tsi',g14.6)
       if(tsi.eq.0.d0)then 
         rpp=1.d15
       else
         rpp=abs(ti2/tsi)
       endif
       if(rpp.lt.1.d14) then
        wgt=ti2/tsi
        si=si+ti*wgt
        si2=si2+ti2
        swgt=swgt+wgt
        schi=schi+ti2*wgt
995    FORMAT(1X,'SWGT',G14.6,'SI2',G14.6)
        avgi=si/swgt
        sd=swgt*it/si2
        chi2a=sd*(schi/swgt-avgi*avgi)/(it-.999)
        sd=dsqrt(one/sd)
       else
        write(*,*) '***VEGAS WARNING***: zero error!'
        write(*,*) ' we guess that integral is exact '
        avgi=ti
        si=ti
        si2=0.d0
        sd=0
        chi2a=-1
        return
       endif
c
        if(nprn.eq.0)go to 21
        tsi=dsqrt(tsi)
        write(6,201)it,ti,tsi,avgi,sd,chi2a
cRF: also update the grids for the integer sum
        call regrid_MC_integer
        if(nprn.ge.0)go to 21
        do 20 j=1,ndim
 20     write(6,202) j,(xi(i,j),di(i,j),d(i,j),i=1,nd)
c
c      refine grid
c
 21     do 23 j=1,ndim
        xo=d(1,j)
        xn=d(2,j)
        d(1,j)=(xo+xn)/2.
        dt(j)=d(1,j)
        do 22 i=2,ndm
        d(i,j)=xo+xn
        xo=xn
        xn=d(i+1,j)
        d(i,j)=(d(i,j)+xn)/3.
 22     dt(j)=dt(j)+d(i,j)
        d(nd,j)=(xn+xo)/2.
 23     dt(j)=dt(j)+d(nd,j)
c
        do 28 j=1,ndim
        rc=0.
        do 24 i=1,nd
        r(i)=0.
        if(d(i,j).le.0.)go to 24
        xo=dt(j)/d(i,j)
        r(i)=((xo-one)/xo/dlog(xo))**alph
 24     rc=rc+r(i)
        rc=rc/xnd
        k=0
        xn=0.
        dr=xn
        i=k
 25     k=k+1
        dr=dr+r(k)
        xo=xn
        xn=xi(k,j)
 26     if(rc.gt.dr)go to 25
        i=i+1
        dr=dr-rc
        xin(i)=xn-(xn-xo)*dr/r(k)
        if(i.lt.ndm)go to 26
        do 27 i=1,ndm
 27     xi(i,j)=xin(i)
 28     xi(nd,j)=one
c
        if(it.lt.itmx.and.acc*dabs(avgi).lt.sd)go to 9
 200    format(1X,'0input parameters for vegas:  ndim=',i3,
     1  '   ncall=',f8.0/28x,'  it=',i5,'    itmx=',i5/28x,
     2  '  acc=',g9.3/28x,'  mds=',i3,'     nd=',i4/28x,
     3  '  (xl,xu)=',(t40,'( ',g12.6,' , ',g12.6,' )'))
 201    format(///' integration by vegas' / '0iteration no.',i5,
     1  ':  integral=',g14.8/21x,'std dev =',g10.4 /
     2  ' accumulated results:   integral=',g14.8/
     3  24x,'std dev =',g10.4 / 24x,'chi**2 per it''n =',g10.4)
 202    format(1X,'0data for axis',i2,/,' ',6x,'x',7x,'  delt i ',
     1  2x,'conv','ce   ',11x,'x',7x,'  delt i ',2x,'conv','ce  '
     2  ,11x,'x',7x,'   delt i ',2x,'conv','CE  ',/,
     3  (1X,' ',3g12.4,5x,3g12.4,5x,3g12.4))
        return
        entry vegas4(fxn,avgi,sd,chi2a)
        if(si2.eq.0.d0)then
          avgi=si
          sd=0
          chi2a=-1
        else
          avgi=si/swgt
          sd=swgt*it/si2
          chi2a=sd*(schi/swgt-avgi*avgi)/(it-.999)
          sd=dsqrt(one/sd)
        endif
        if(nprn.ne.0)then
           write(6,201)it,0.d0,0.d0,avgi,sd,chi2a
        endif
        return
        end

c        subroutine save(ndim)
c        implicit real*8 (a-h,o-z)
c       implicit integer*4 (i-n)
c        common/bveg2/xi(50,58),si,si2,swgt,schi,ndo,it
c
c      stores vegas data   (unit 7) for later initialisation
c
c        write(7,200) ndo,it,si,si2,swgt,schi,
c     1       ((xi(i,j),i=1,ndo),j=1,ndim)
c        return
c        entry restr(ndim)
c
c         enters initialisation data for vegas
c
c        read(7,200) ndo,it,si,si2,swgt,schi,
c     1    ((xi(i,j),i= 1,ndo),j=1,ndim)
c 200    format(2i8,4z16/(5z16))
c        return
c        end

      

      subroutine randa(n,rand)
      implicit double precision (a-h,o-z)
      common/caso/caso(5)
      dimension rand(58)
      real*8 ran2
      external ran2
      do 1 i=1,n
      rand(i)=ran2()
1     continue
      do 2 i=1,5
      caso(i)=ran2()
2     continue
      return
      end


      subroutine sysdep
c use newver='NEW' for vaxes, 'UNKNOWN' in other machines,
      character * 7 newver
      common/newver/newver
      newver = 'UNKNOWN'
      end

      subroutine delete(fname)
      character * 80 str
      character * (*) fname
      l = len(fname)
      k = 1
      dowhile(fname(k:k).eq.' '.and.k.lt.l)
         k = k+1
      enddo
      dowhile(fname(l:l).eq.' '.and.l.gt.k)
         l = l-1
      enddo
      if(l-k.gt.70) then
         write(*,*) 'delete: filename > 70 chars not allowed'
         stop
      endif
      if(l.eq.k) then
         write(*,*) 'delete: void filename'
         stop
      endif
      str(1:) = 'rm -f '
      str(7:) = fname(k:l)
      call system(str)
      end


c
c
c A simple integration wrapper
c
c
      subroutine xinteg(sig,indim,initn,incalls,avg,stddev)
      implicit real * 8 (a-h,o-z)
      common/bveg1/xl(58),xu(58),acc,ndim,ncall,itmx,nprn
      common/bveg2/xi(50,58),si,si2,swgt,schi,ndo,it
      common/cwww/w1max,w1vgs,w1evt
      common/ciweight/iweight
      external sig
c
      do j=1,58
        xl(j)=0.d0
        xu(j)=1.d0
      enddo
c Set nprn equal to zero to print nothing
      nprn=1
      acc=-1.d0
      ndim=indim
      ncall=incalls
      itmx=1
      do j=1,initn
        if(j.eq.1)then
          call vegas(sig,avgi,sd,chi2a)
        else
          call vegas3(sig,avgi,sd,chi2a)
        endif
      enddo
      avg=avgi
      stddev=sd
      return
      end



c Compile with:
c  gfortran -ffixed-line-length-132 -o readfort90n readfort90n.for 
c                                       mcatnlo_hbook_gfortran.f
      implicit real*8(a-h,o-z)
      include 'nexternal.inc'
      real*8 p(0:3,nexternal),psave(nexternal)
      integer ifksmat(nexternal),jfksmat(nexternal)
      character*3 ch1,ch3,ch1save
      character*2 ch2
      logical check,plot,dead
      common/cdead/dead
c
      dead=.false.
      write(6,*)'enter number of events'
      read(5,*)nevts
      write(6,*)'enter iunit'
      read(5,*)iunit
      write(6,*)'enter 1 to plot SDK'
      write(6,*)'      2 to plot SD3'
      write(6,*)'      3 to plot SD1'
      write(6,*)'      4 to plot EVT'
      read(5,*)ipl1
      if(ipl1.eq.4)then
        write(6,*)'enter 0 to plot S+H events'
        write(6,*)'      1 to plot S events'
        write(6,*)'      2 to plot H events'
        read(5,*)ipl2
        write(6,*)'enter 0 to plot all events'
        write(6,*)'      1 to plot w>0 events'
        write(6,*)'      2 to plot w<0 events'
        read(5,*)ipl3
        write(6,*)'enter 0 to exclude ev/w sc1>sc0'
        write(6,*)'      1 otherwise'
        read(5,*)ipl4
        if(ipl4.eq.0)dead=.true.
      endif
c See the setting of itype in outfun() to see what
c is plotted, in particular pt vs pt(relative)
      write(6,*)'enter -2 to keep j_fks<=2'
      write(6,*)'      -1 to keep j_fks>2'
      write(6,*)'       0 to keep all'
      write(6,*)'       n>0 to keep j_fks=n'
      read(5,*)njfks
      write(6,*)'enter 0 for events generated without folding'
      write(6,*)'      1 otherwise'
      read(5,*)ifold
      ch2='  '
      call fk88strnum(ch2,iunit)
      open(unit=99,file='sudakov'//ch2//'.top',status='unknown')
      call initplot
      itot=0
      i1=0
      i2=0
      i3=0
      i4=0
      j1=0
      j2=0
      j3=0
      js=0
      jsn=0
      jsp=0
      jh=0
      jhp=0
      jhn=0
      j1sn=0
      j1sp=0
      j1hp=0
      j1hn=0
      j2sn=0
      j2sp=0
      j2hp=0
      j2hn=0
      j3sn=0
      j3sp=0
      j3hp=0
      j3hn=0
      do i=1,nexternal
        ifksmat(i)=0
        jfksmat(i)=0
      enddo
      ch3='EVT'
      do i=1,nevts
        check=.false.
        read(iunit,554,err=999,end=999)ch1,sud,bogus,scltarget,
     #                                 sclstart,sudpdffact
        read(iunit,557)i_fks,j_fks
        if( i_fks.le.0.or.i_fks.gt.nexternal .or.
     #      j_fks.le.0.or.j_fks.gt.nexternal )then
          write(*,*)'FKS error',i,i_fks,j_fks
          stop
        endif
        ifksmat(i_fks)=ifksmat(i_fks)+1
        jfksmat(j_fks)=jfksmat(j_fks)+1
        if(ch1.ne.'EVT')then
          ch1save=ch1
          i_fkssave=i_fks
          j_fkssave=j_fks
          sudsave=sud
          bogsave=bogus
          sc0save=sclstart
          sc1save=scltarget
          pdfsave=sudpdffact
        else
          check=check .or.
     #          i_fkssave .ne. i_fks .or.
     #          j_fkssave .ne. j_fks
          if(ch1save.ne.'SD1')
     #      check=check .or.
     #          abs(sudsave-sud).gt.1.d-6 .or.
     #          abs(bogsave-bogus).gt.1.d-6 .or.
     #          abs(sc0save-sclstart).gt.1.d-6 .or.
     #          abs(sc1save-scltarget).gt.1.d-6 .or.
     #          abs(pdfsave-sudpdffact).gt.1.d-6
        endif
        plot=(ipl1.eq.1.and.ch1.eq.'SDK') .or.
     #       (ipl1.eq.2.and.ch1.eq.'SD3') .or.
     #       (ipl1.eq.3.and.ch1.eq.'SD1') .or.
     #       (ipl1.eq.4.and.ch1.eq.'EVT')
        plot=plot .and. (
     #       (njfks.eq.-2.and.j_fks.le.2) .or.
     #       (njfks.eq.-1.and.j_fks.gt.2) .or.
     #        njfks.eq.0 .or.
     #       (njfks.gt.0.and.j_fks.eq.njfks) )
        itot=itot+1
        if(ch1.eq.'SDK')then
          i1=i1+1
        elseif(ch1.eq.'SD3')then
          i2=i2+1
        elseif(ch1.eq.'SD1')then
          i3=i3+1
        elseif(ch1.eq.'EVT')then
          if(iunit.ne.92.or.(iunit.eq.92.and.ch3.eq.ch1))then
            write(6,*)'error #1:',i,ch1,ch3
            stop
          endif
          i4=i4+1
          jtype=0
          if(ch3.eq.'SDK')then
            j1=j1+1
            jtype=1
          elseif(ch3.eq.'SD3')then
            j2=j2+1
            jtype=2
          elseif(ch3.eq.'SD1')then
            j3=j3+1
            jtype=3
          endif
        else
          write(6,*)'error #2:',i,ch1
          stop
        endif
        if(ch1.eq.'EVT')then
          read(iunit,556,err=999,end=999)izoo,evtsgn
          plot=plot .and.
     #         ( ipl2.eq.0 .or.
     #           (ipl2.eq.1.and.izoo.eq.0) .or.
     #           (ipl2.eq.2.and.izoo.eq.1) ) .and.
     #         ( ipl3.eq.0 .or.
     #           (ipl3.eq.1.and.evtsgn.gt.0.d0) .or.
     #           (ipl3.eq.2.and.evtsgn.lt.0.d0) )
          if(izoo.eq.0)then
            js=js+1
            if(evtsgn.gt.0.d0)then
              jsp=jsp+1
              if(jtype.eq.1)then
                j1sp=j1sp+1
              elseif(jtype.eq.2)then
                j2sp=j2sp+1
              elseif(jtype.eq.3)then
                j3sp=j3sp+1
              endif
            else
              jsn=jsn+1
              if(jtype.eq.1)then
                j1sn=j1sn+1
              elseif(jtype.eq.2)then
                j2sn=j2sn+1
              elseif(jtype.eq.3)then
                j3sn=j3sn+1
              endif
            endif
          elseif(izoo.eq.1)then
            jh=jh+1
            if(evtsgn.gt.0.d0)then
              jhp=jhp+1
              if(jtype.eq.1)then
                j1hp=j1hp+1
              elseif(jtype.eq.2)then
                j2hp=j2hp+1
              elseif(jtype.eq.3)then
                j3hp=j3hp+1
              endif
            else
              jhn=jhn+1
              if(jtype.eq.1)then
                j1hn=j1hn+1
              elseif(jtype.eq.2)then
                j2hn=j2hn+1
              elseif(jtype.eq.3)then
                j3hn=j3hn+1
              endif
            endif
          else
            write(6,*)'error #5:',i,izoo,evtsgn
            stop
          endif
        endif
        ch3=ch1
        do kk=1,nexternal
          read(iunit,555,err=999,end=999)
     #      kk0,p(0,kk),p(1,kk),p(2,kk),p(3,kk)
          if(kk0.ne.kk)then
            write(6,*)'error #3:',i,kk,kk0,sud,bogus
            stop
          endif
          if(ch1.ne.'EVT')then
            psave(kk)=p(0,kk)
          else
            if(ifold.eq.0)check=check.or.abs(psave(kk)-p(0,kk)).gt.1.d-6
          endif
        enddo
        if(check)then
          write(6,*)'error #4:',i
          write(6,*)'error #4:',sudsave,sud,bogsave,bogus
          write(6,*)'error #4:',sc0save,sclstart,sc1save,scltarget
          write(6,*)'error #4:',pdfsave,sudpdffact
          write(6,*)'error #4:',psave(1),p(0,1),psave(2),p(0,2)
          write(6,*)'error #4:',i_fks,i_fkssave,j_fks,j_fkssave
          stop
        endif
        if(plot)call outfun(p,sud,bogus,scltarget,sclstart,sudpdffact,
     #                      i_fks,j_fks)
      enddo
 999  continue
      call topout
      close(99)
      write(6,*)'total number of events:',itot
      write(6,*)'of which SDK:',i1
      write(6,*)'of which SD3:',i2
      write(6,*)'of which SD1:',i3
      write(6,*)'of which EVT:',i4
      if(i4.ne.0)then
        write(6,*)'S evts:',js,' w>0:',jsp,'   w<0:',jsn
        write(6,*)'H evts:',jh,' w>0:',jhp,'   w<0:',jhn
        write(6,*)'# previous SDK event (Sw>0,Sw<0,Hw>0,Hw<0):',j1,
     #    j1sp,j1sn,j1hp,j1hn
        write(6,*)'# previous SD3 event (Sw>0,Sw<0,Hw>0,Hw<0):',j2,
     #    j2sp,j2sn,j2hp,j2hn
        write(6,*)'# previous SD1 event (Sw>0,Sw<0,Hw>0,Hw<0):',j3,
     #    j3sp,j3sn,j3hp,j3hn
      endif
      do i=1,nexternal
        write(6,600)'#',i,' --> ifks:',ifksmat(i),'  jfks:',jfksmat(i)
      enddo
 554  format(1x,a,5(1x,e14.8))
 555  format(1x,i2,5(1x,e14.8))
 556  format(1x,i2,1x,e14.8)
 557  format(2(1x,i2))
 600  format(1x,a1,i2,a10,i8,a7,i8)
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


      function xptrel(en1,px1,py1,pz1,en2,px2,py2,pz2)
      implicit real*8(a-h,o-z)
c
      xl1=p3(en1,px1,py1,pz1)
      xl2=p3(en2,px2,py2,pz2)
      if(xl1.eq.0)then
        xptrel=0.d0
        return
      endif
      if(xl2.eq.0)then
        xptrel=-1.d8
        return
      endif
      en3=0.d0
      px3=py1*pz2-pz1*py2
      py3=pz1*px2-px1*pz2
      pz3=px1*py2-py1*px2
      tmp=p3(en3,px3,py3,pz3)/xl2
      xptrel=tmp
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
      character*4 cc(2),dd(2),ee(1)
      data cc/' sdk',' prb'/
      data dd/' sc0',' sc1'/
      data ee/' PDF'/
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
      k=500
      call mbook(k+ 1,'start',2.e0,0.e0,200.e0)
      call mbook(k+ 2,'target',2.e0,0.e0,200.e0)
      call mbook(k+ 3,'start',0.8e0,0.e0,80.e0)
      call mbook(k+ 4,'target',0.8e0,0.e0,80.e0)
      call mbook(k+ 5,'start-target',2.e0,-50.e0,150.e0)
      call mbook(k+ 6,'start-target',1.e0,-10.e0,90.e0)
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
      do i=1,3
      do j=1,2
      k=208+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'pt'//dd(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'energy'//dd(j),2.e0,0.e0,200.e0)
      call mbook(k+ 3,'theta1'//dd(j),pi/50.e0,0.e0,pi)
      call mbook(k+ 4,'theta2'//dd(j),pi/50.e0,0.e0,pi)
      k=212+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'log10[pt]'//dd(j),0.05e0,-3.e0,2.e0)
      call mbook(k+ 2,'log10[energy]'//dd(j),0.05e0,-3.e0,2.e0)
      call mbook(k+ 3,'log10[theta1]'//dd(j),0.05e0,-3.e0,0.2e0)
      call mbook(k+ 4,'log10[theta2]'//dd(j),0.05e0,-3.e0,0.2e0)
      enddo
      enddo
c
      do i=1,3
      do j=1,1
      k=308+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'start'//cc(j),0.8e0,0.e0,80.e0)
      call mbook(k+ 2,'target'//cc(j),0.8e0,0.e0,80.e0)
      call mbook(k+ 3,'start-target'//cc(j),2.e0,-50.e0,150.e0)
      call mbook(k+ 4,'start-target'//cc(j),1.e0,-10.e0,90.e0)
      enddo
      do j=2,2
      k=308+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'start'//dd(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'start'//dd(j),0.8e0,0.e0,80.e0)
      enddo
      enddo
c
      do i=1,3
      do j=1,1
      k=608+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'pt'//ee(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'energy'//ee(j),2.e0,0.e0,200.e0)
      call mbook(k+ 3,'theta1'//ee(j),pi/50.e0,0.e0,pi)
      call mbook(k+ 4,'theta2'//ee(j),pi/50.e0,0.e0,pi)
      k=612+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'log10[pt]'//ee(j),0.05e0,-3.e0,2.e0)
      call mbook(k+ 2,'log10[energy]'//ee(j),0.05e0,-3.e0,2.e0)
      call mbook(k+ 3,'log10[theta1]'//ee(j),0.05e0,-3.e0,0.2e0)
      call mbook(k+ 4,'log10[theta2]'//ee(j),0.05e0,-3.e0,0.2e0)
      enddo
      enddo
c
      do i=1,3
      do j=1,1
      k=708+(j-1)*8+(i-1)*16
      call mbook(k+ 1,'start'//ee(j),0.8e0,0.e0,80.e0)
      call mbook(k+ 2,'target'//ee(j),0.8e0,0.e0,80.e0)
      call mbook(k+ 3,'start-target'//ee(j),2.e0,-50.e0,150.e0)
      call mbook(k+ 4,'start-target'//ee(j),1.e0,-10.e0,90.e0)
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
      do j=1,2
      k=208+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      k=212+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      enddo
      do j=1,1
      k=308+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      enddo
      do j=2,2
      k=308+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      enddo
      do j=1,1
      k=608+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      k=612+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      enddo
      do j=1,1
      k=708+(j-1)*8+(i-1)*16
      call mopera(k+ 1,'M',k+17,k+33,szero,szero)
      call mopera(k+ 2,'M',k+18,k+34,szero,szero)
      call mopera(k+ 3,'M',k+19,k+35,szero,szero)
      call mopera(k+ 4,'M',k+20,k+36,szero,szero)
      enddo
      do i=1,1000
        call mfinal(i)
      enddo
      do i=1,8
        call mnorm(i,sone)
      enddo
      do i=501,506
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
c
      do j=1,2
      k=208+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'pt',' ','LIN')
      call multitop(k+ 2,99,2,2,'energy',' ','LIN')
      call multitop(k+ 3,99,2,2,'theta1',' ','LIN')
      call multitop(k+ 4,99,2,2,'theta2',' ','LIN')
      enddo
c
      do j=1,2
      k=212+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'log10[pt]',' ','LOG')
      call multitop(k+ 2,99,2,2,'log10[energy]',' ','LOG')
      call multitop(k+ 3,99,2,2,'log10[theta1]',' ','LOG')
      call multitop(k+ 4,99,2,2,'log10[theta2]',' ','LOG')
      enddo                
c
      k=500
      call multitop(k+ 1,99,2,2,'mu',' ','LOG')
      call multitop(k+ 2,99,2,2,'tij',' ','LOG')
      call multitop(k+ 3,99,2,2,'mu',' ','LOG')
      call multitop(k+ 4,99,2,2,'tij',' ','LOG')
      call multitop(k+ 5,99,2,2,'mu-tij',' ','LOG')
      call multitop(k+ 6,99,2,2,'mu-tij',' ','LOG')
c
      do j=1,1
      k=308+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'mu',' ','LOG')
      call multitop(k+ 2,99,2,2,'tij',' ','LOG')
      call multitop(k+ 3,99,2,2,'mu-tij',' ','LOG')
      call multitop(k+ 4,99,2,2,'mu-tij',' ','LOG')
      enddo
c
      do j=2,2
      k=308+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'mu',' ','LIN')
      call multitop(k+ 2,99,2,2,'mu',' ','LIN')
      enddo
c
      i=2
      do j=1,1
      k=608+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'pt',' ','LOG')
      call multitop(k+ 2,99,2,2,'energy',' ','LOG')
      call multitop(k+ 3,99,2,2,'theta1',' ','LOG')
      call multitop(k+ 4,99,2,2,'theta2',' ','LOG')
      enddo
c
      do j=1,2
      k=612+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'log10[pt]',' ','LOG')
      call multitop(k+ 2,99,2,2,'log10[energy]',' ','LOG')
      call multitop(k+ 3,99,2,2,'log10[theta1]',' ','LOG')
      call multitop(k+ 4,99,2,2,'log10[theta2]',' ','LOG')
      enddo                
c
      do j=1,1
      k=708+(j-1)*8+(i-1)*16
      call multitop(k+ 1,99,2,2,'mu',' ','LOG')
      call multitop(k+ 2,99,2,2,'tij',' ','LOG')
      call multitop(k+ 3,99,2,2,'mu-tij',' ','LOG')
      call multitop(k+ 4,99,2,2,'mu-tij',' ','LOG')
      enddo
c
      end


      subroutine outfun(p,sud,bogus,sc1,sc0,pdf,i_fks,j_fks)
      implicit real*8(a-h,o-z)
      real*4 one
      parameter (pi=3.1415926535897932385d0)
      parameter (one=1.e0)
      include 'nexternal.inc'
      parameter (ione=1)
      parameter (itwo=2)
      real*8 p(0:3,nexternal),q(0:3)
      integer itype
c itype=0 -> pt, angles wrt parton #1
c itype=1 -> pt, angles wrt parton j_fks
      parameter (itype=1)
      logical pscales,dead
      common/cdead/dead
c
      if(sc1.gt.sc0.and.dead)return
      if(sc0.lt.0.d0)then
        if( sc1.gt.0.d0 .or.
     #      (sc1.lt.0.d0 .and. sud.ne.1.d0) )then
          write(*,*)'scales are negative while sud#1'
          write(*,*)sud,sc0,sc1
          stop
        endif
        pscales=.false.
      else
        pscales=.true.
      endif
      if(itype.eq.0)then
        pt=xpt(p(0,i_fks),p(1,i_fks),
     #         p(2,i_fks),p(3,i_fks))
        th1=angle(p(0,i_fks),p(1,i_fks),
     #            p(2,i_fks),p(3,i_fks),
     #            p(0,ione),p(1,ione),
     #            p(2,ione),p(3,ione))
        th2=angle(p(0,i_fks),p(1,i_fks),
     #            p(2,i_fks),p(3,i_fks),
     #            p(0,itwo),p(1,itwo),
     #            p(2,itwo),p(3,itwo))
      elseif(itype.eq.1)then
        if(j_fks.le.2)then
          do i=0,3
            q(i)=p(i,j_fks)
          enddo
        else
          do i=0,3
            q(i)=p(i,i_fks)+p(i,j_fks)
          enddo
        endif
        pt=xptrel(p(0,i_fks),p(1,i_fks),
     #            p(2,i_fks),p(3,i_fks),
     #            q(0),q(1),q(2),q(3))
        th1=angle(p(0,i_fks),p(1,i_fks),
     #            p(2,i_fks),p(3,i_fks),
     #            q(0),q(1),q(2),q(3))
        th2=th1
      else
        write(*,*)'error in outfun: itype=',itype
        stop
      endif
      energy=p(0,i_fks)
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
c
      kk=500
      if(pscales)then
      call mfill(kk+1,sngl(sc0),one)
      call mfill(kk+2,sngl(sc1),one)
      call mfill(kk+3,sngl(sc0),one)
      call mfill(kk+4,sngl(sc1),one)
      call mfill(kk+5,sngl(sc0-sc1),one)
      call mfill(kk+6,sngl(sc0-sc1),one)
      endif
c
      do i=1,3
        ipow=i-1
c
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
        kk=608+(i-1)*16
        call mfill(kk+1,sngl(pt),one*sngl(pdf**ipow))
        call mfill(kk+2,sngl(energy),one*sngl(pdf**ipow))
        call mfill(kk+3,sngl(th1),one*sngl(pdf**ipow))
        call mfill(kk+4,sngl(th2),one*sngl(pdf**ipow))
c
        kk=612+(i-1)*16
        if(pt.gt.0.d0)
     #    call mfill(kk+1,sngl(log10(pt)),one*sngl(pdf**ipow))
        if(energy.gt.0.d0)
     #    call mfill(kk+2,sngl(log10(energy)),one*sngl(pdf**ipow))
        call mfill(kk+3,sngl(xth1),one*sngl(pdf**ipow))
        call mfill(kk+4,sngl(xth2),one*sngl(pdf**ipow))
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
c
        if(pscales)then
        kk=208+(i-1)*16
        call mfill(kk+1,sngl(pt),one*sngl(sc0**ipow))
        call mfill(kk+2,sngl(energy),one*sngl(sc0**ipow))
        call mfill(kk+3,sngl(th1),one*sngl(sc0**ipow))
        call mfill(kk+4,sngl(th2),one*sngl(sc0**ipow))
c
        kk=212+(i-1)*16
        if(pt.gt.0.d0)
     #    call mfill(kk+1,sngl(log10(pt)),one*sngl(sc0**ipow))
        if(energy.gt.0.d0)
     #    call mfill(kk+2,sngl(log10(energy)),one*sngl(sc0**ipow))
        call mfill(kk+3,sngl(xth1),one*sngl(sc0**ipow))
        call mfill(kk+4,sngl(xth2),one*sngl(sc0**ipow))
c
        kk=216+(i-1)*16
        call mfill(kk+1,sngl(pt),one*sngl(sc1**ipow))
        call mfill(kk+2,sngl(energy),one*sngl(sc1**ipow))
        call mfill(kk+3,sngl(th1),one*sngl(sc1**ipow))
        call mfill(kk+4,sngl(th2),one*sngl(sc1**ipow))
c
        kk=220+(i-1)*16
        if(pt.gt.0.d0)
     #    call mfill(kk+1,sngl(log10(pt)),one*sngl(sc1**ipow))
        if(energy.gt.0.d0)
     #    call mfill(kk+2,sngl(log10(energy)),one*sngl(sc1**ipow))
        call mfill(kk+3,sngl(xth1),one*sngl(sc1**ipow))
        call mfill(kk+4,sngl(xth2),one*sngl(sc1**ipow))
c
        kk=308+(i-1)*16
        call mfill(kk+1,sngl(sc0),one*sngl(sud**ipow))
        call mfill(kk+2,sngl(sc1),one*sngl(sud**ipow))
        call mfill(kk+3,sngl(sc0-sc1),one*sngl(sud**ipow))
        call mfill(kk+4,sngl(sc0-sc1),one*sngl(sud**ipow))
c
        kk=708+(i-1)*16
        call mfill(kk+1,sngl(sc0),one*sngl(pdf**ipow))
        call mfill(kk+2,sngl(sc1),one*sngl(pdf**ipow))
        call mfill(kk+3,sngl(sc0-sc1),one*sngl(pdf**ipow))
        call mfill(kk+4,sngl(sc0-sc1),one*sngl(pdf**ipow))
c
        kk=316+(i-1)*16
        call mfill(kk+1,sngl(sc0),one*sngl(sc1**ipow))
        call mfill(kk+2,sngl(sc0),one*sngl(sc1**ipow))
        endif
c
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

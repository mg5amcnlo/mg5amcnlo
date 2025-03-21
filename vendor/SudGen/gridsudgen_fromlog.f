      implicit none
      integer nl,nf,npart,ntype
c Assume five light flavours, the top, and the gluon
      parameter (nl=5)
      parameter (nf=nl+1)
      parameter (npart=nf+1)
c 1=II, 2=FF, 3=IF, 4=FI
      parameter (ntype=4)
c values of grid nodes: Q(j), 1<=j<=jmax, Q being either st or xm
c (or their squares if need be). They are defined as follows:
c   Q(j) = b^[ Q0*(j/jmax)^k - alpha ]
c and by imposing 
c   Q(1)    = Qmin
c   Q(jmax) = Qmax
c with Qmin and Qmax given. These imply the following consistency relations:
c   alpha*log(b)*(jmax^k-1) = log(Qmax) - jmax^k*log(Qmin)
c   Q0 = alpha + log(Qmax)/log(b)
c These are best exploited by choosing k, jmax, and b, and thus solving
c for alpha and Q0. 
c At given k and jmax, the values of the grid nodes are independent
c of b. Thus, we can just choose b=10 or b=e. Consider that:
c  - more points towards Qmin -> increase k
c  - less points towards Qmin -> decrease k
c In any case, it is sensible to choose k of order one.
c
c nodes of the st grid (jmax for st)
      integer nnst
c$$$      parameter (nnst=26)
      parameter (nnst=100)
c nodes of the xm grid (jmax for xm)
      integer nnxm
c$$$      parameter (nnxm=20)
      parameter (nnxm=50)
c b of the formulae above
      real*8 base
      parameter (base=10.d0)
c k power for the st grid
      real*8 xkst
      parameter (xkst=1.d0)
c k power for the xm grid
      real*8 xkxm
      parameter (xkxm=1.d0)
c grid nodes and grids
      real*8 st(nnst),xm(nnxm),grids(nnst,nnxm,npart,ntype)
c
      integer i,j,id,inst,inxm,ipart,itype,i1,i2,ipmap(100)
      integer ifakegrid
      real*8 stlow,stupp,xmlow,xmupp,alst,q0st,alxm,q0xm,
     # qnodeval
c$$$      external SUDAKOV FUNCTION
      external py_compute_sudakov
      double precision py_compute_sudakov,restmp
      double precision tolerance,xlowthrs
      parameter (tolerance=1.d-2)
      integer iunit1,iunit2,iunit3,maxseed,iseed,iseedtopy,ifk88seed
      parameter (iunit1=20)
      parameter (iunit2=30)
      parameter (iunit3=40)
      parameter (maxseed=100000)
      common/cifk88seed/ifk88seed
      integer infHloop,infLloop,maxloop
      parameter (maxloop=5)
      integer ieHoob,ieLoob,ieHoob2,ieLoob2,ieHfail,ieLfail
      real*8 fk88random,rnd
      real*8 mcmass(21)
      logical grid(21)
      character*200 buff,filelog,cdum1,cdum2,cdum3,cdum4
c
      do i=1,21
        mcmass(i)=0.d0
        grid(i)=.false.
      enddo
      include 'MCmasses_PYTHIA8.inc'
c
      call dire_init(mcmass)
      open(unit=iunit1,file='sudakov.log',status='unknown')
      open(unit=iunit2,file='sudakov.err',status='unknown')

      write(*,*)'enter lower and upper bounds of st range'
      read(*,*)stlow,stupp
      write(*,*)'enter lower and upper bounds of M range'
      read(*,*)xmlow,xmupp

      call getalq0(nnst,xkst,stlow,stupp,base,alst,q0st)
      call getalq0(nnxm,xkxm,xmlow,xmupp,base,alxm,q0xm)
c
      do ipart=1,npart
        if(ipart.lt.npart)then
          ipmap(ipart)=ipart
        else
          ipmap(ipart)=21
        endif
        grid(ipmap(ipart))=.true.
      enddo
      ieHoob=0
      ieLoob=0
      ieHoob2=0
      ieLoob2=0
      ieHfail=0
      ieLfail=0
      do itype=1,4
c        if (itype .ne. 3) cycle
         write(*,*)'===>Doing itype=',itype
         write(iunit1,*)'===>Doing itype=',itype
         write(iunit2,*)'===>Doing itype=',itype
         do ipart=1,npart
c          if (ipart .ne. 7) cycle
            write(*,*)'   --->Doing ipart=',ipart
            write(iunit1,*)'   --->Doing ipart=',ipart
            write(iunit2,*)'   --->Doing ipart=',ipart
            do inxm=1,nnxm
c$$$            if (inxm .le. 15) cycle
               xm(inxm)=qnodeval(inxm,nnxm,xkxm,base,alxm,q0xm)

               filelog='sudakov_'
               write (cdum1,*) itype
               write (cdum2,*) ipart
               write (cdum3,*) inxm
               filelog='sudakov_'//trim(adjustl(cdum1))//'_'
     $              //trim(adjustl(cdum2))//'_'//trim(adjustl(cdum3))
     $              //'.log'

               open(unit=iunit3,file=filelog,status='unknown')
               read(iunit3,'(a)') buff
               read(iunit3,'(a)') buff
               
               do inst=1,nnst
c$$$            if (inst .le. 20) cycle
                  st(inst)=qnodeval(inst,nnst,xkst,base,alst,q0st)
                  read(iunit3,'(a)') buff
                  read(buff(108:200),*) restmp
                  grids(inst,inxm,ipart,itype)= restmp
               enddo
               close (iunit3)
            enddo
            write(*,*)'   <---Done ipart= ',ipart
            write(iunit1,*)'   <---Done ipart= ',ipart
            write(iunit2,*)'   <---Done ipart= ',ipart
         enddo
         write(*,*)'<===Done itype= ',itype
         write(iunit1,*)'<===Done itype= ',itype
         write(iunit2,*)'<===Done itype= ',itype
      enddo
c
      open(unit=10,file='sudakov.f',status='unknown')
c
      write(10,'(a)')
     #'      function pysudakov(st,xm,id,itype,xmpart)',
     #'      implicit none',
     #'      real*8 pysudakov,st,xm,xmpart(21)',
     #'      integer id,itype',
     #'      real*8 tmp,cstmin,cxmmin,cxmmax',
     #'      real*8 mcmass(21)',
     #'      integer i,id0,listmin,lixmmin,lixmmax',
     #'      logical firsttime,check,T,F,grid(21)',
     #'      parameter (T=.true.)',
     #'      parameter (F=.false.)'
      write(10,'(a)')
     #'      real*8 stlow,stupp,xmlow,xmupp'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (stlow=',stlow,',stupp=',stupp,')'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (xmlow=',xmlow,',xmupp=',xmupp,')'
      write(10,'(a)')
     #'      real*8 cstlow,cstupp,cxmlow,cxmupp',
     #'      common/cstxmbds/cstlow,cstupp,cxmlow,cxmupp'
      do itype=1,4
        do ipart=1,npart
          id=abs(ipmap(ipart))
          if(id.lt.10)then
            write(10,'(a,i1,a,i1)')
     #'      real*8 pysudakov_',id,'_',itype
          elseif(id.lt.100)then
            write(10,'(a,i2,a,i1)')
     #'      real*8 pysudakov_',id,'_',itype
          else
            write(6,*)'parton id too large',id
            stop
          endif
        enddo
      enddo
      write(10,'(a)')
     #'      data mcmass/'
      do i=1,5
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(mcmass(j),j=1+4*(i-1),4*i)
      enddo
      write(10,'(a,d15.8,a)')
     #'     #',mcmass(21),'/'
      write(10,'(a,20(l1,1h,),l1,a)')
     #'      data grid/',(grid(j),j=1,20),grid(21),'/'
      write(10,'(a)')
     #'      data firsttime/.true./',
     #'      save'
c
      write(10,'(a)')
     #'c',
     #'      if(firsttime)then',
     #'        firsttime=.false.',
     #'        check=.false.',
     #'        do i=1,21',
     #'          if(grid(i))check=check .or.',
     #'     #      abs(xmpart(i)-mcmass(i)).gt.1.d-6',
     #'        enddo',
     #'        if(check)then',
     #'          write(*,*)''input masses inconsistent''',
     #'     #              //'' with grid setup''',
     #'          stop',
     #'        endif',
     #'        cstlow=stlow',
     #'        cstupp=stupp',
     #'        cxmlow=xmlow',
     #'        cxmupp=xmupp',
     #'        listmin=0',
     #'        lixmmin=0',
     #'        lixmmax=0',
     #'        cstmin=0.d0',
     #'        cxmmin=0.d0',
     #'        cxmmax=0.d0',
     #'      endif',
     #'      if(st.gt.stupp)then',
     #'        write(*,*)''st is too large:'',st',
     #'        write(*,*)''largest value allowed is:'',stupp',
     #'        stop',
     #'      endif',
     #'      if(st.lt.stlow)then',
     #'        cstmin=cstmin+1.d0',
     #'        if(log10(cstmin).gt.listmin)then',
     #'          write(*,*)''==========================''',
     #'          write(*,*)''Warning in pysudakov:''',
     #'          write(*,*)''  st<stlow more than 10**'',',
     #'     #      listmin,'' times''',
     #'          write(*,*)''==========================''',
     #'          listmin=listmin+1',
     #'        endif',
     #'      endif',
     #'      if(xm.lt.xmlow)then',
     #'        cxmmin=cxmmin+1.d0',
     #'        if(log10(cxmmin).gt.lixmmin)then',
     #'          write(*,*)''==========================''',
     #'          write(*,*)''Warning in pysudakov:''',
     #'          write(*,*)''xm<xmlow more than 10**'',',
     #'     #      lixmmin,'' times''',
     #'          write(*,*)''==========================''',
     #'          lixmmin=lixmmin+1',
     #'        endif',
     #'      endif',
     #'      if(xm.gt.xmupp)then',
     #'        cxmmax=cxmmax+1.d0',
     #'        if(log10(cxmmax).gt.lixmmax)then',
     #'          write(*,*)''==========================''',
     #'          write(*,*)''Warning in pysudakov:''',
     #'          write(*,*)''xm>xmupp more than 10**'',',
     #'     #      lixmmax,'' times''',
     #'          write(*,*)''==========================''',
     #'          lixmmax=lixmmax+1',
     #'        endif',
     #'      endif',
     #'      id0=abs(id)'
      do itype=1,4
        if(itype.eq.1)then
          write(10,'(a)')
     #'      if(itype.eq.1)then'
          do ipart=1,npart
            id=abs(ipmap(ipart))
            if(id.eq.1)then
              write(10,'(a)')
     #'        if(id0.eq.1)then',
     #'          tmp=pysudakov_1_1(st,xm)'
            elseif(id.lt.10)then
              write(10,'(a,i1,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i1,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm)'
            else
              write(10,'(a,i2,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i2,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm)'
            endif
          enddo
          write(10,'(a)')
     #'        else',
     #'          write(*,*)''unknown parton ID:'',id',
     #'          stop',
     #'        endif'
        else
          write(10,'(a,i1,a)')
     #'      elseif(itype.eq.',itype,')then'
          do ipart=1,npart
            id=abs(ipmap(ipart))
            if(id.eq.1)then
              write(10,'(a)')
     #'        if(id0.eq.1)then'
              write(10,'(a,i1,a)')
     #'          tmp=pysudakov_1_',itype,'(st,xm)'
            elseif(id.lt.10)then
              write(10,'(a,i1,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i1,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm)'
            else
              write(10,'(a,i2,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i2,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm)'
            endif
          enddo
          write(10,'(a)')
     #'        else',
     #'          write(*,*)''unknown parton ID:'',id',
     #'          stop',
     #'        endif'
        endif
      enddo
      write(10,'(a)')
     #'      else',
     #'        write(*,*)''unknown type:'',itype',
     #'        stop',
     #'      endif',
     #'      pysudakov=tmp',
     #'      return',
     #'      end'
c
      write(10,'(a)')
     #'c',
     #'c',
     #'cccc',
     #'c',
     #'c'
c
      do itype=1,4
      do ipart=1,npart
      id=abs(ipmap(ipart))
c
      if(id.lt.10)then
        write(10,'(a,i1,a,i1,a)')
     #'      function pysudakov_',id,'_',itype,'(st,xm)'
        write(10,'(a)')
     #'      implicit none'
        write(10,'(a,i1,a,i1,a)')
     #'      real*8 pysudakov_',id,'_',itype,',st,xm'
      elseif(id.lt.100)then
        write(10,'(a,i2,a,i1,a)')
     #'      function pysudakov_',id,'_',itype,'(st,xm)'
        write(10,'(a)')
     #'      implicit none'
        write(10,'(a,i2,a,i1,a)')
     #'      real*8 pysudakov_',id,'_',itype,',st,xm'
      endif
      write(10,'(a)')
     #'      integer narg,nnst,nnxm',
     #'      parameter (narg=2)'
      write(10,'(a,i3,a)')
     #'      parameter (nnst=',nnst,')'
      write(10,'(a,i3,a)')
     #'      parameter (nnxm=',nnxm,')'
      write(10,'(a)')
     #'      integer inst,inxm,nent(narg)',
     #'      real*8 tmp,dfint,stmap,xmmap',
     #'      real*8 arg(narg),ent(nnst+nnxm)',
     #'      real*8 stv(nnst),xmv(nnxm),gridv(nnst,nnxm)',
     #'      logical firsttime',
     #'      external dfint,stmap,xmmap'
c
      write(10,'(a)')
     #'      data stv/'
      i1=nnst/4
      i2=mod(nnst,4)
      if(i2.eq.0)i1=i1-1
      do i=1,i1
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(st(j),j=1+(i-1)*4,i*4)
      enddo
      write(10,'(a,4(d15.8,a))')
     #'     #',(st(j),',',j=1+i1*4,nnst-1),st(nnst),'/'
c
      write(10,'(a)')
     #'      data xmv/'
      i1=nnxm/4
      i2=mod(nnxm,4)
      if(i2.eq.0)i1=i1-1
      do i=1,i1
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(xm(j),j=1+(i-1)*4,i*4)
      enddo
      write(10,'(a,4(d15.8,a))')
     #'     #',(xm(j),',',j=1+i1*4,nnxm-1),xm(nnxm),'/'
c
      do inxm=1,nnxm
        write(10,'(a,i3,a,i3,a)')
     #'      data (gridv(inst,',inxm,'),inst=1,',nnst,')/'
        i1=nnst/4
        i2=mod(nnst,4)
        if(i2.eq.0)i1=i1-1
        do i=1,i1
          write(10,'(a,4(d15.8,1h,))')
     #'     #',(grids(j,inxm,ipart,itype),j=1+(i-1)*4,i*4)
        enddo
        write(10,'(a,4(d15.8,a))')
     #'     #',(grids(j,inxm,ipart,itype),',',j=1+i1*4,nnst-1),
     #    grids(nnst,inxm,ipart,itype),'/'
      enddo
c
      write(10,'(a)')
     #'      data firsttime/.true./',
     #'      save',
     #'c',
     #'      if(firsttime)then',
     #'        firsttime=.false.',
     #'        nent(1)=nnst',
     #'        nent(2)=nnxm',
     #'        do inst=1,nnst',
     #'          ent(inst)=stmap(stv(inst))',
     #'        enddo',
     #'        do inxm=1,nnxm',
     #'          ent(nnst+inxm)=xmmap(xmv(inxm))',
     #'        enddo',
     #'      endif',
     #'      arg(1)=stmap(st)',
     #'      arg(2)=xmmap(xm)',
     #'      tmp=dfint(narg,arg,nent,ent,gridv)'
      if(id.lt.10)then
        write(10,'(a,i1,a,i1,a)')
     #'      pysudakov_',id,'_',itype,'=tmp'
      elseif(id.lt.100)then
        write(10,'(a,i2,a,i1,a)')
     #'      pysudakov_',id,'_',itype,'=tmp'
      endif
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
c
      write(10,'(a)')
     #'      function stmap(st)',
     #'c Use this function to interpolate by means of',
     #'c   stnode_i=stmap(stnode_stored_i).',
     #'c Example (to be used below): tmp=log10(st)',
     #'      implicit none',
     #'      real*8 stmap,st,tmp',
     #'c',
     #'      tmp=st',
     #'      stmap=tmp',
     #'      return',
     #'      end',
     #'  ',
     #'  ',
     #'      function xmmap(xm)',
     #'c Use this function to interpolate by means of',
     #'c   xmnode_i=xmmap(xmnode_stored_i).',
     #'c Example (to be used below): tmp=log10(xm)',
     #'      implicit none',
     #'      real*8 xmmap,xm,tmp',
     #'c',
     #'      tmp=xm',
     #'      xmmap=tmp',
     #'      return',
     #'      end'
c
      close(10)
      close(iunit1)
      write(iunit2,*)'     '
      write(iunit2,*)'Sudakovs>1 within tolerance: ',ieHoob
      write(iunit2,*)'Sudakovs<0 within tolerance: ',ieLoob
      write(iunit2,*)'Sudakovs>1 outside tolerance: ',ieHoob2
      write(iunit2,*)'Sudakovs<0 outside tolerance: ',ieLoob2
      write(iunit2,*)'Sudakovs>>1 set to 1: ',ieHfail
      write(iunit2,*)'Sudakovs<<0 set to 0: ',ieLfail
      close(iunit2)
      end


      subroutine getalq0(jmax,xk,qmin,qmax,b,alpha,q0)
      implicit none
      integer jmax
      real*8 xk,qmin,qmax,b,alpha,q0
c
      alpha=log(qmax)-jmax**xk*log(qmin)
      alpha=alpha/((jmax**xk-1)*log(b))
      q0=alpha+log(qmax)/log(b)
      return
      end


      function qnodeval(j,jmax,xk,b,alpha,q0)
      implicit none
      real*8 qnodeval
      integer j,jmax
      real*8 xk,b,alpha,q0,tmp
c
      tmp=b**( q0*(j/dfloat(jmax))**xk - alpha )
      qnodeval=tmp
      return
      end


      subroutine invqnodeval(qnode,jmax,xk,b,alpha,q0,j1,j2)
c "Inverse" of qnodeval -- returns the two nearest integers
c corresponding to the given note qnode
      implicit none
      real*8 qnode
      integer jmax,j1,j2
      real*8 xk,b,alpha,q0,tmp
c
      tmp=jmax*q0**(-1/xk)*(log(qnode)/log(b)+alpha)**(1/xk)
      j1=int(tmp)
      j2=j1+1
      return
      end


      function ifakegrid(jst,jxm,id,itype)
      implicit none
      integer ifakegrid,jst,jxm,id,itype
c
      ifakegrid=jst+100*jxm+10000*id+1000000*itype
      return
      end


      function py_compute_sudakov(stlow,md,id,itype,mcmass,stupp,
     #                            iseed,min_py_sudakov,iunit)
      implicit none
      double precision py_compute_sudakov, stlow, stupp, md
      double precision min_py_sudakov
      integer id, itype,iseed,iunit
      real*8 mcmass(21)
      double precision temp
c
      call dire_get_no_emission_prob(temp, stupp,
     #     stlow, md, id, itype, iseed, min_py_sudakov)
      py_compute_sudakov=temp
      write(iunit,*) 'md=', md, ' start=', stupp,
     #           ' stop=', stlow, ' --> sud=', temp
c      write(*,*) 'md=', md, ' start=', stupp,
c     #           ' stop=', stlow, ' --> sud=', temp
c
      return
      end


      FUNCTION FK88RANDOM(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      REAL*8 MINV,FK88RANDOM
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      FK88RANDOM = SEED*MINV
      END


      function iseedtopy()
      implicit none
      integer iseedtopy,itmp
      integer maxseed,iseed,ifk88seed
      parameter (maxseed=100000)
      common/cifk88seed/ifk88seed
      real*8 rnd,fk88random
c
      if(ifk88seed.eq.-1.or.ifk88seed.eq.0)then
        itmp=ifk88seed
      else
        rnd=fk88random(ifk88seed)
        itmp=int(maxseed*rnd)+1
      endif
      iseedtopy=itmp
      return
      end


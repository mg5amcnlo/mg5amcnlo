c This code generalises gridsudgen.f, giving one three options in input:
c   inode=1 -> same as gridsudgen.f
c   inode=2 -> a possibly different nodal-point spacing wrt that of inode=1,
c              depending on the choices of base and k exponent
c   inode=3 -> nodal points in t may depend on a non-constant upper bound,
c              defined by the function stbound(), thus increasing efficiency
c While stbound() is written in the body of sudakov.f regardless of the
c choice of inode, in the case of inode=1,2 it is unused (as is the
c variable stmax).
c
c WARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNING
c
c The use of inode=3 requires some care, since the definitions of stbound()
c in this file, and in the body of sudakov.f, written by this code, MUST
c COINCIDE. Thus look for stbound in this file, and make sure that the
c two definitions (one explicit, and one implicit in write(10,*) statements)
c are identical
c
c WARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNING
      implicit none
      integer nl,nf,npart,ntype
c Assume five light flavours, the top, and the gluon
      parameter (nl=5)
      parameter (nf=nl+1)
      parameter (npart=nf+1)
c 1=II, 2=FF, 3=IF, 4=FI
      parameter (ntype=4)
c When inode=1:
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
      parameter (nnst=100)
c nodes of the xm grid (jmax for xm)
      integer nnxm
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
      integer i,j,id,inst,inxm,ipart,itype,i1,i2,inode,ipmap(100)
      integer ifakegrid
      real*8 stlow,stupp,xmlow,xmupp,alst,q0st,alxm,q0xm,stval,xmval,
     # stmax,stbound,qnodeval,qnodeval2,qnodeval3,aux1_qnode3
c$$$      external SUDAKOV FUNCTION
      external py_compute_sudakov
      double precision py_compute_sudakov,restmp
      double precision tolerance,xlowthrs
      parameter (tolerance=1.d-2)
      integer iunit1,iunit2,maxseed,iseed,iseedtopy,ifk88seed
      parameter (iunit1=20)
      parameter (iunit2=30)
      parameter (maxseed=100000)
      common/cifk88seed/ifk88seed
      integer infHloop,infLloop,maxloop
      parameter (maxloop=5)
      integer ieHoob,ieLoob,ieHoob2,ieLoob2,ieHfail,ieLfail
      real*8 fk88random,rnd
      real*8 mcmass(21)
      logical grid(21)
c
      do i=1,21
        mcmass(i)=0.d0
        grid(i)=.false.
      enddo
      include 'MCmasses_PYTHIA8.inc'
c
      call pythia_init(mcmass)
      open(unit=iunit1,file='sudakov.log',status='unknown')
      open(unit=iunit2,file='sudakov.err',status='unknown')

      write(*,*)'enter lower and upper bounds of st range'
      read(*,*)stlow,stupp
      write(*,*)'enter lower and upper bounds of M range'
      read(*,*)xmlow,xmupp
      write(*,*)'enter Sudakov lower threshold'
      write(*,*)' Sudakov will be set to zero if below threshold'
      read(*,*)xlowthrs
c$$$      xlowthrs=0.0001
      write(*,*)'enter 1 for choice of nodes #1'
      write(*,*)'      2 for choice of nodes #2'
      write(*,*)'      3 for choice of nodes #3'
      read(*,*)inode

      write(*,*)'enter -1 to use Pythia default seed'
      write(*,*)'       0 to use Pythia timestamp'
      write(*,*)'       >=1 to use random seeds'
      read(5,*)ifk88seed
c Discard first (tends to be extremely small)
      if(ifk88seed.ge.1)rnd=fk88random(ifk88seed)

      if(inode.eq.1)then
        call getalq0(nnst,xkst,stlow,stupp,base,alst,q0st)
        call getalq0(nnxm,xkxm,xmlow,xmupp,base,alxm,q0xm)
      endif
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
c$$$        if (itype .ne. 3) cycle
        write(*,*)'===>Doing itype=',itype
        write(iunit1,*)'===>Doing itype=',itype
        write(iunit2,*)'===>Doing itype=',itype
        do ipart=1,npart
c$$$          if (ipart .ne. 7) cycle
          write(*,*)'   --->Doing ipart=',ipart
          write(iunit1,*)'   --->Doing ipart=',ipart
          write(iunit2,*)'   --->Doing ipart=',ipart
          do inxm=1,nnxm
c$$$            if (inxm .le. 15) cycle
            if(inode.eq.1)then
              xm(inxm)=qnodeval(inxm,nnxm,xkxm,base,alxm,q0xm)
            elseif(inode.eq.2)then
              xm(inxm)=qnodeval2(inxm,nnxm,xkxm,xmlow,xmupp)
            elseif(inode.eq.3)then
              xm(inxm)=qnodeval3(inxm,nnxm,xkxm)
            endif
            do inst=1,nnst
c$$$            if (inst .le. 20) cycle
              if(inode.eq.1)then
                st(inst)=qnodeval(inst,nnst,xkst,base,alst,q0st)
              elseif(inode.eq.2)then
                st(inst)=qnodeval2(inst,nnst,xkst,stlow,stupp)
              elseif(inode.eq.3)then
                st(inst)=qnodeval3(inst,nnst,xkst)
              endif
              infHloop=0
              infLloop=0
 111          continue
              if(infHloop.eq.maxloop)then
                write(iunit2,*)'Failure (too large): st, xm=',
     #                         st(inst),xm(inxm)
                restmp=1.d0
                ieHfail=ieHfail+1
                goto 112
              endif
              if(infLloop.eq.maxloop)then
                write(iunit2,*)'Failure (too small): st, xm=',
     #                         st(inst),xm(inxm)
                restmp=0.d0
                ieLfail=ieLfail+1
                goto 112
              endif
c$$$              restmp = ifakegrid(inst,inxm,ipmap(ipart),itype)
c Compute the Sudakov relevant to parton line ipmap(ipart),
c for dipole type itype with mass xm(inxm), between the scales
c st(inst) and stupp
              iseed=iseedtopy()
              if(inode.eq.1.or.inode.eq.2)then
                stval=st(inst)
                xmval=xm(inxm)
              else
                xmval=aux1_qnode3(xm(inxm),xmlow,xmupp)
                stmax=stbound(xmval,stupp,mcmass(ipmap(ipart)),
     #                        itype,ipmap(ipart))
                stval=aux1_qnode3(st(inst),stlow,stmax)
              endif
              restmp = py_compute_sudakov(
     #          stval,xmval,ipmap(ipart),itype,
     #          mcmass,stupp,iseed,xlowthrs,iunit1)
              if(restmp.gt.1.d0)then
                if(restmp.le.(1.d0+tolerance))then
                  write(iunit2,*)'Out of bounds (>1): ',restmp
                  write(iunit2,*)'  for st, xm=',st(inst),xm(inxm)
                  write(iunit2,*)'  seed=',iseed
                  ieHoob=ieHoob+1
                  restmp=1.d0
                  goto 112
                else
                  write(iunit2,*)'Out of bounds (>>1): ',restmp
                  write(iunit2,*)'  for st, xm=',st(inst),xm(inxm)
                  write(iunit2,*)'  seed=',iseed
                  ieHoob2=ieHoob2+1
                  infHloop=infHloop+1
                  goto 111
                endif
              endif
              if(restmp.lt.xlowthrs)then
                if(restmp.ge.(-tolerance))then
                  if(restmp.lt.0.d0)then
                    write(iunit2,*)'Out of bounds (<0): ',restmp
                    write(iunit2,*)'  for st, xm=',st(inst),xm(inxm)
                    write(iunit2,*)'  seed=',iseed
                    ieLoob=ieLoob+1
                  endif
                  restmp=0.d0
                  goto 112
                else
                  write(iunit2,*)'Out of bounds (<<0): ',restmp
                  write(iunit2,*)'  for st, xm=',st(inst),xm(inxm)
                  write(iunit2,*)'  seed=',iseed
                  ieLoob2=ieLoob2+1
                  infLloop=infLloop+1
                  goto 111
                endif
              endif
 112          continue
              grids(inst,inxm,ipart,itype)= restmp
            enddo
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
     #'c This file parametrises the Sudakov Delta(t,T) with:',
     #'c   stlow<= t <=stupp,   T=stupp',
     #'c and dipole masses Md in the range:',
     #'c   xmlow<= Md <=xmupp',
     #'c The nodal values are computed with the parameters above and:',
     #'c   k=xkst,  jmax=nnst    --> for t',
     #'c   k=xkxm,  jmax=nnxm    --> for Md',
     #'c The actual values of stlow, stupp, xmlow, xmupp, xkst, nnst, ',
     #'c xkxm, and nnxm used in the construction of the grids are ',
     #'c included in this file as parameters',
     #'      function pysudakov(st,xm,id,itype,xmpart)',
     #'      implicit none',
     #'      real*8 pysudakov,st,xm,xmpart(21)',
     #'      integer id,itype',
     #'      real*8 tmp,cstmin,cxmmin,cxmmax,stmax,stbound',
     #'      real*8 mcmass(21)',
     #'      integer i,id0,listmin,lixmmin,lixmmax',
     #'      logical firsttime,check,T,F,grid(21)',
     #'      parameter (T=.true.)',
     #'      parameter (F=.false.)'
      write(10,'(a)')
     #'      real*8 stlow,stupp,xmlow,xmupp,xkst,xkxm'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (stlow=',stlow,',stupp=',stupp,')'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (xmlow=',xmlow,',xmupp=',xmupp,')'
      write(10,'(a)')
     #'      real*8 cstlow,cstupp,cxmlow,cxmupp',
     #'      common/cstxmbds/cstlow,cstupp,cxmlow,cxmupp'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (xkst=',xkst,',xkxm=',xkxm,')'
      write(10,'(a)')
     #'      integer nnst,nnxm'
      write(10,'(a,i3,a)')
     #'      parameter (nnst=',nnst,')'
      write(10,'(a,i3,a)')
     #'      parameter (nnxm=',nnxm,')'
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
     #'      external stbound',
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
     #'      id0=abs(id)',
     #'      stmax=stbound(xm,stupp,xmpart(id0),itype,id0)'
      do itype=1,4
        if(itype.eq.1)then
          write(10,'(a)')
     #'      if(itype.eq.1)then'
          do ipart=1,npart
            id=abs(ipmap(ipart))
            if(id.eq.1)then
              write(10,'(a)')
     #'        if(id0.eq.1)then',
     #'          tmp=pysudakov_1_1(st,xm,stmax)'
            elseif(id.lt.10)then
              write(10,'(a,i1,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i1,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm,stmax)'
            else
              write(10,'(a,i2,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i2,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm,stmax)'
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
     #'          tmp=pysudakov_1_',itype,'(st,xm,stmax)'
            elseif(id.lt.10)then
              write(10,'(a,i1,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i1,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm,stmax)'
            else
              write(10,'(a,i2,a)')
     #'        elseif(id0.eq.',id,')then'
              write(10,'(a,i2,a,i1,a)')
     #'          tmp=pysudakov_',id,'_',itype,'(st,xm,stmax)'
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
     #'      function pysudakov_',id,'_',itype,'(st,xm,stmax)'
        write(10,'(a)')
     #'      implicit none'
        write(10,'(a,i1,a,i1,a)')
     #'      real*8 pysudakov_',id,'_',itype,',st,xm,stmax'
      elseif(id.lt.100)then
        write(10,'(a,i2,a,i1,a)')
     #'      function pysudakov_',id,'_',itype,'(st,xm,stmax)'
        write(10,'(a)')
     #'      implicit none'
        write(10,'(a,i2,a,i1,a)')
     #'      real*8 pysudakov_',id,'_',itype,',st,xm,stmax'
      endif
      write(10,'(a)')
     #'      integer narg,nnst,nnxm',
     #'      parameter (narg=2)'
      write(10,'(a,i3,a)')
     #'      parameter (nnst=',nnst,')'
      write(10,'(a,i3,a)')
     #'      parameter (nnxm=',nnxm,')'
      write(10,'(a)')
     #'      real*8 stlow,stupp,xmlow,xmupp'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (stlow=',stlow,',stupp=',stupp,')'
      write(10,'(a,d15.8,a,d15.8,a)')
     #'      parameter (xmlow=',xmlow,',xmupp=',xmupp,')'
      write(10,'(a)')
     #'      integer inst,inxm,nent(narg)',
     #'      real*8 tmp,dfint,stmap,xmmap,aux2_qnode3',
     #'      real*8 arg(narg),ent(nnst+nnxm)',
     #'      real*8 stv(nnst),xmv(nnxm),gridv(nnst,nnxm)',
     #'      logical firsttime',
     #'      external dfint,stmap,xmmap,aux2_qnode3'
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
     #'      endif'
      if(inode.eq.1.or.inode.eq.2)then
        write(10,'(a)')
     #'      arg(1)=stmap(st)',
     #'      arg(2)=xmmap(xm)',
     #'      tmp=dfint(narg,arg,nent,ent,gridv)'
      else
        write(10,'(a)')
     #'      if(stmax.gt.stupp)then',
     #'        write(*,*)''Error in function'''
        if(id.lt.10)then
          write(10,'(a,i1,a,i1,a)')
     #'        write(*,*)''  pysudakov_',id,'_',itype,'(st,xm,stmax):'''
        elseif(id.lt.100)then
          write(10,'(a,i2,a,i1,a)')
     #'        write(*,*)''  pysudakov_',id,'_',itype,'(st,xm,stmax):'''
        endif
        write(10,'(a)')
     #'        write(*,*)''stmax>stupp:'',stmax,stupp',
     #'        stop',
     #'      endif',
     #'      if(st.le.stmax)then',
     #'        arg(1)=stmap(aux2_qnode3(st,stlow,stmax))',
     #'        arg(2)=xmmap(aux2_qnode3(xm,xmlow,xmupp))',
     #'        tmp=dfint(narg,arg,nent,ent,gridv)',
     #'      else',
     #'        tmp=0.d0',
     #'      endif'
      endif
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
     #'      end',
     #'  ',
     #'  ',
     #'      function aux2_qnode3(q,qmin,qmax)',
     #'      implicit none',
     #'      real*8 aux2_qnode3',
     #'      real*8 q,qmin,qmax,tmp',
     #'c',
     #'      tmp=(log(q)-log(qmin))/(log(qmax)-log(qmin))',
     #'      aux2_qnode3=tmp',
     #'      return',
     #'      end',
     #'  ',
     #'  ',
     #'      function stbound(xm,stupp,xmass,itype,id)',
     #'c When st>stbound(...) the Sudakov will be set equal to one.',
     #'c The value of stbound is limited from above by stupp',
     #'      implicit none',
     #'      real*8 stbound,xm,stupp,xmass,tmp',
     #'      integer itype,id',
     #'c',
     #'c Insert here the actual MC bound on st, which may depend on',
     #'c xm (the dipole mass), xmass (the particle mass), itype',
     #'c (the type of connection, 1..4), and id (the particle PDG id)',
     #'c$$$      tmp=1.d10',
     #'      tmp=xm',
     #'c Do NOT remove the following line',
     #'      tmp=min(stupp,tmp)',
     #'      stbound=tmp',
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


      function qnodeval2(j,jmax,xk,qmin,qmax)
      implicit none
      real*8 qnodeval2
      integer j,jmax
      real*8 xk,qmin,qmax,ff,tmp
c
      if(j.eq.1)then
        tmp=qmin
      else
        ff=( (j-1)/dfloat(jmax-1) )**xk
        tmp=qmin*(qmax/qmin)**ff
      endif
      qnodeval2=tmp
      return
      end


      function qnodeval3(j,jmax,xk)
      implicit none
      real*8 qnodeval3
      integer j,jmax
      real*8 xk,tmp
c
      if(j.eq.1)then
        tmp=0
      else
        tmp=( (j-1)/dfloat(jmax-1) )**xk
      endif
      qnodeval3=tmp
      return
      end


      function aux1_qnode3(ff,qmin,qmax)
      implicit none
      real*8 aux1_qnode3
      real*8 ff,qmin,qmax,tmp
c
      tmp=log(qmin)+ff*(log(qmax)-log(qmin))
      aux1_qnode3=exp(tmp)
      return
      end


      function aux2_qnode3(q,qmin,qmax)
      implicit none
      real*8 aux2_qnode3
      real*8 q,qmin,qmax,tmp
c
      tmp=(log(q)-log(qmin))/(log(qmax)-log(qmin))
      aux2_qnode3=tmp
      return
      end


      function stbound(xm,stupp,xmass,itype,id)
c When st>stbound(...) the Sudakov will be set equal to one.
c The value of stbound is limited from above by stupp
      implicit none
      real*8 stbound,xm,stupp,xmass,tmp
      integer itype,id
c
c Insert here the actual MC bound on st, which may depend on
c xm (the dipole mass), xmass (the particle mass), itype
c (the type of connection, 1..4), and id (the particle PDG id)
c$$$      tmp=1.d10
      tmp=xm
c Do NOT remove the following line
      tmp=min(stupp,tmp)
      stbound=tmp
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
      call pythia_get_no_emission_prob(temp, stupp,
     #     stlow, md, id, itype, iseed, min_py_sudakov)
      py_compute_sudakov=temp
      write(iunit,*) 'md=', md, ' start=', stupp,
     #           ' stop=', stlow, ' --> sud=', temp
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


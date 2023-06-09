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
      integer ipartlow,ipartupp,itypelow,itypeupp,imasslow,imassupp
      integer ifakegrid
      real*8 stlow,stupp,xmlow,xmupp,alst,q0st,alxm,q0xm,
     # qnodeval
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
      character*1 str_itype,str_ipart
      character*3 str_inxm
      character*50 fname
c
      do i=1,21
        mcmass(i)=0.d0
        grid(i)=.false.
      enddo
      include 'MCmasses_PYTHIA8.inc'
c
c      call dire_init(mcmass)
      call pythia_init(mcmass)
      open(unit=iunit1,file='sudakov.log',status='unknown')
      open(unit=iunit2,file='sudakov.err',status='unknown')

      write(*,*)'enter lower and upper bounds of st range'
      read(*,*)stlow,stupp
c      stlow=5.0
c      stupp=1000.0
      write(*,*)'enter lower and upper bounds of M range'
      read(*,*)xmlow,xmupp
c      xmlow=5.0
c      xmupp=100.0
      write(*,*)'enter Sudakov lower threshold'
      write(*,*)' Sudakov will be set to zero if below threshold'
      read(*,*)xlowthrs
c      xlowthrs=0.0001

      write(*,*)'enter -1 to use Pythia default seed'
      write(*,*)'       0 to use Pythia timestamp'
      write(*,*)'       >=1 to use random seeds'
      read(*,*)ifk88seed
c      ifk88seed=-1

      write(*,*)'enter itype and ipart and dipole mass'
      read(*,*)itypelow,ipartlow,imasslow
      itypeupp=itypelow
      ipartupp=ipartlow
      imassupp=imasslow
      if(itypelow.lt.1.or.itypeupp.gt.4.or.ipartlow.lt.1)then
         write(*,*)'wrong itype and/or ipart'
         write(*,*)itypelow,itypeupp,ipartlow
         stop
      endif

c Discard first (tends to be extremely small)
      if(ifk88seed.ge.1)rnd=fk88random(ifk88seed)

      call getalq0(nnst,xkst,stlow,stupp,base,alst,q0st)
      call getalq0(nnxm,xkxm,xmlow,xmupp,base,alxm,q0xm)
c
      do ipart=ipartlow,ipartupp
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
      do itype=itypelow,itypeupp
c        if (itype .ne. 1) cycle
        write(*,*)'===>Doing itype=',itype
        write(iunit1,*)'===>Doing itype=',itype
        write(iunit2,*)'===>Doing itype=',itype
        do ipart=ipartlow,ipartupp
c          if (ipart .ne. 7) cycle
          write(*,*)'   --->Doing ipart=',ipart
          write(iunit1,*)'   --->Doing ipart=',ipart
          write(iunit2,*)'   --->Doing ipart=',ipart
          do inxm=1,nnxm
             xm(inxm)=qnodeval(inxm,nnxm,xkxm,base,alxm,q0xm)
          enddo
c          do inxm=1,nnxm
          do inxm=imasslow,imassupp
            do inst=1,nnst
              st(inst)=qnodeval(inst,nnst,xkst,base,alst,q0st)
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
              restmp = py_compute_sudakov(
     #          st(inst),xm(inxm),ipmap(ipart),itype,
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

c write to grid files
               write(str_itype,'(i1)')itype
               write(str_ipart,'(i1)')ipart
               if (inxm.le.9) then
                  write(str_inxm,'(a,i1)') '00',inxm
               elseif (inxm.le.99) then
                  write(str_inxm,'(a,i2)') '0',inxm
               else
                  write(str_inxm,'(i3)')inxm
               endif
               fname='grid_'//trim(str_itype)//'_'//trim(str_ipart)//'_'
     $              //trim(str_inxm)//'.txt'
               open(unit=33,file=fname)
               if (inxm.eq.1) then
                  write(33,'(a)')
     #'      data stv/'
                  i1=nnst/4
                  i2=mod(nnst,4)
                  if(i2.eq.0)i1=i1-1
                  do i=1,i1
                     write(33,'(a,4(d15.8,1h,))')
     #'     #',(st(j),j=1+(i-1)*4,i*4)
                  enddo
                  write(33,'(a,4(d15.8,a))')
     #'     #',(st(j),',',j=1+i1*4,nnst-1),st(nnst),'/'
c
                  write(33,'(a)')
     #'      data xmv/'
                  i1=nnxm/4
                  i2=mod(nnxm,4)
                  if(i2.eq.0)i1=i1-1
                  do i=1,i1
                     write(33,'(a,4(d15.8,1h,))')
     #'     #',(xm(j),j=1+(i-1)*4,i*4)
                  enddo
                  write(33,'(a,4(d15.8,a))')
     #'     #',(xm(j),',',j=1+i1*4,nnxm-1),xm(nnxm),'/'
               endif
c
               write(33,'(a,i3,a,i3,a)')
     #'      data (gridv(inst,',inxm,'),inst=1,',nnst,')/'
               i1=nnst/4
               i2=mod(nnst,4)
               if(i2.eq.0)i1=i1-1
               do i=1,i1
                  write(33,'(a,4(d15.8,1h,))')
     #'     #',(grids(j,inxm,ipart,itype),j=1+(i-1)*4,i*4)
               enddo
               write(33,'(a,4(d15.8,a))')
     #'     #',(grids(j,inxm,ipart,itype),',',j=1+i1*4,nnst-1),
     #        grids(nnst,inxm,ipart,itype),'/'
               close(33)
            enddo
         enddo
      enddo
c
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
c      call dire_get_no_emission_prob(temp, stupp,
c     #     stlow, md, id, itype, iseed, min_py_sudakov)
      call pythia_get_no_emission_prob(temp, stupp,
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


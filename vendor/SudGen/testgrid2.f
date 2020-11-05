c gfortran -o testgrid2 testgrid2.f sudakov.f dfint.f kerset.f
c
c Parameters must set by hand equal to those hardwired in gridsudgen.f 
c or given in input to that code.
c If changes occur there, they must be implemented here as well
c
c This codes outputs a few files; not all of them are available in the
c three testing options, which correspond to:
c  - test a given st and xm scales
c  - test randomly in both st and xm
c  - plot the Sudakov as a function of either st or xm
c The output files are:
c check_sudakov.dat -> nodal values, must be identical to those in sudakov.f
c fort.60 -> summary of the random test, with values of the averages of ratios
c            of sudakovs computed with the grids and directly with Pythia
c fort.66 -> plots the Sudakov, both with the grid and directly with Pythia,
c            and their ratios. Vertical lines correspond to nodal values
c fort.67 -> as for fort.66, on a log scale on the x axis
      implicit real*8(a-h,o-z)
      real*8 mcmass(21)
      integer ifk88seed,iunit1,iseedpy
      parameter (iunit1=20)
      double precision st1, st2, xm1, xm2, stupp
      double precision stval(100),xmval(100)
      double precision rnrat(8),rnmin(8),rnmax(8)
      double precision avgn(8),stsv1n(8),stsv2n(8),xmsv1n(8),xmsv2n(8)
      double precision plres(0:1000,2)

      double precision res_grid, res_pythia
      double precision res_mlo_stlo_grid, res_mhi_stlo_grid
      double precision res_mlo_sthi_grid, res_mhi_sthi_grid

      double precision res_mlo_stlo_pythia, res_mhi_stlo_pythia
      double precision res_mlo_sthi_pythia, res_mhi_sthi_pythia

      double precision xlowthrs

      common/cifk88seed/ifk88seed
c
      do i=1,21
        mcmass(i)=0.d0
      enddo
      include 'MCmasses_PYTHIA8.inc'

      call dire_init(mcmass)
      open(unit=iunit1,file='testgrid.log',status='unknown')

c$$$      nnst=26
      nnst=100
      xkst=1.d0
      base=10.d0
c$$$      nnxm=20
      nnxm=50
      xkxm=1.d0

      write(*,*)'  '
      write(*,*)'This codes assumes sudakov.f to have been'
      write(*,*)' created with the following parameters:'
      write(*,*)' nnst=',nnst
      write(*,*)' xkst=',xkst
      write(*,*)' base=',base
      write(*,*)' nnxm=',nnxm
      write(*,*)' xkxm',xkxm
      write(*,*)'  '

      write(*,*)'enter lower and upper bounds of fitted st range'
      read(*,*)stlow,stupp
      write(*,*)'enter lower and upper bounds of fitted M range'
      read(*,*)xmlow,xmupp

      call getalq0(nnst,xkst,stlow,stupp,base,alst,q0st)
      call getalq0(nnxm,xkxm,xmlow,xmupp,base,alxm,q0xm)

      write(6,*)'alst, q0st= ',alst,q0st
      write(6,*)'alxm, q0xm= ',alxm,q0xm

c write onto check_sudakov.dat the nodal values, in the same
c format as in sudakov.f
      do inst=1,nnst
        stval(inst)=qnodeval(inst,nnst,xkst,base,alst,q0st)
      enddo
      do inxm=1,nnxm
        xmval(inxm)=qnodeval(inxm,nnxm,xkxm,base,alxm,q0xm)
      enddo
      open(unit=10,file='check_sudakov.dat',status='unknown')
      write(10,'(a)')
     #'      data stv/'
      i1=nnst/4
      i2=mod(nnst,4)
      if(i2.eq.0)i1=i1-1
      do i=1,i1
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(stval(j),j=1+(i-1)*4,i*4)
      enddo
      write(10,'(a,4(d15.8,a))')
     #'     #',(stval(j),',',j=1+i1*4,nnst-1),stval(nnst),'/'
c
      write(10,'(a)')
     #'      data xmv/'
      i1=nnxm/4
      i2=mod(nnxm,4)
      if(i2.eq.0)i1=i1-1
      do i=1,i1
        write(10,'(a,4(d15.8,1h,))')
     #'     #',(xmval(j),j=1+(i-1)*4,i*4)
      enddo
      write(10,'(a,4(d15.8,a))')
     #'     #',(xmval(j),',',j=1+i1*4,nnxm-1),xmval(nnxm),'/'
      close(10)

      write(6,*)'   '
      write(6,*)'enter 0 for a manual test'
      write(6,*)'      1 for a random test'
      write(6,*)'      2 to plot Sudakov'
      read(5,*)itest

      if(itest.eq.1)goto 200
      if(itest.eq.2)goto 300
      write(6,*)'enter id, itype'
      read(5,*)id,itype

c Use Pythia internal seed. Call rnd generator before each
c call to py_compute_sudakov otherwise
      iseedpy=-1
c set cutoff to zero for Pythia to always compute Sudakovs
      xlowthrs=0.d0

 1    write(6,*)'enter st,xm'
      read(5,*)st,xm

      call invqnodeval(st,26,1.d0,10.d0,alst,q0st,j1st,j2st)
      call invqnodeval(xm,20,1.d0,10.d0,alxm,q0xm,j1xm,j2xm)
      st1 = qnodeval(j1st,26,1.d0,10.d0,alst,q0st)
      st2 = qnodeval(j2st,26,1.d0,10.d0,alst,q0st)
      xm1 = qnodeval(j1xm,20,1.d0,10.d0,alxm,q0xm)
      xm2 = qnodeval(j2xm,20,1.d0,10.d0,alxm,q0xm)

      write(6,*)'nearest nodes numbers:',j1st,j2st,j1xm,j2xm
      write(6,*)'nodal values:'
      write(*,*)'st1=', st1, 'st2=', st2
      write(*,*)'xm1=', xm1, 'xm2=', xm2
      if( st.lt.st1.or.st.gt.st2 .or.
     #    xm.lt.xm1.or.xm.gt.xm2 )then
        write(6,*)'WARNINGWARNINGWARNINGWARNINGWARNING'
        write(6,*)'scales out of range'
      endif

c Results at selected scales
      res_grid=pysudakov(st,xm,id,itype,mcmass)
      res_pythia=py_compute_sudakov(st,xm,id,itype,
     #                              mcmass,stupp,
     #                              iseedpy,xlowthrs,
     #                              iunit1)
      write(6,*)'    '
      write(6,*)'Results at the selected scales'
      write(6,*)'res(grid)=  ',res_grid
      write(6,*)'res(pythia)=',res_pythia
      write(6,*)'ratio=      ',res_grid/res_pythia

c Results at nearest nodal values
      res_mlo_stlo_grid=pysudakov(st1,xm1,id,itype,mcmass)
      res_mlo_sthi_grid=pysudakov(st2,xm1,id,itype,mcmass)
      res_mhi_stlo_grid=pysudakov(st1,xm2,id,itype,mcmass)
      res_mhi_sthi_grid=pysudakov(st2,xm2,id,itype,mcmass)
      res_mlo_stlo_pythia = py_compute_sudakov(st1,xm1,id,itype,
     #                                         mcmass,stupp,
     #                                         iseedpy,xlowthrs,
     #                                         iunit1)
      res_mlo_sthi_pythia = py_compute_sudakov(st2,xm1,id,itype,
     #                                         mcmass,stupp,
     #                                         iseedpy,xlowthrs,
     #                                         iunit1)
      res_mhi_stlo_pythia = py_compute_sudakov(st1,xm2,id,itype,
     #                                         mcmass,stupp,
     #                                         iseedpy,xlowthrs,
     #                                         iunit1)
      res_mhi_sthi_pythia = py_compute_sudakov(st2,xm2,id,itype,
     #                                         mcmass,stupp,
     #                                         iseedpy,xlowthrs,
     #                                         iunit1)

      r1 = res_mlo_stlo_grid/res_mlo_stlo_pythia
      r2 = res_mlo_sthi_grid/res_mlo_sthi_pythia
      r3 = res_mhi_stlo_grid/res_mhi_stlo_pythia
      r4 = res_mhi_sthi_grid/res_mhi_sthi_pythia

      write(6,*)'    '
      write(6,*)'Ratios grid/pythia at nearest nodal values'
      write(6,*)r1
      write(6,*)r2
      write(6,*)r3
      write(6,*)r4
      write(6,*)'average of ratios:',(r1+r2+r3+r4)/4.d0

      r1 = res_grid/res_mlo_stlo_grid
      r2 = res_grid/res_mlo_sthi_grid
      r3 = res_grid/res_mhi_stlo_grid
      r4 = res_grid/res_mhi_sthi_grid

      write(6,*)'    '
      write(6,*)'Ratios grid(central)/grid(nodal)'
      write(6,*)r1
      write(6,*)r2
      write(6,*)r3
      write(6,*)r4
      write(6,*)'average of ratios:',(r1+r2+r3+r4)/4.d0

      write(6,*)'    '
      write(6,*)'    '
      goto 1
 200  continue
      write(6,*)'enter id, itype'
      write(6,*)' enter -1 to loop over either/both'
      read(5,*)idin,itypein
      if(idin.gt.0)then
        idl=idin
        idu=idin
      else
        idl=1
        idu=7
      endif
      if(itypein.gt.0)then
        itypel=itypein
        itypeu=itypein
      else
        itypel=1
        itypeu=4
      endif
      write(6,*)'enter stlow, stupp'
      read(5,*)rstlow,rstupp
      write(6,*)'enter xmlow, xmupp'
      read(5,*)rxmlow,rxmupp
      write(6,*)'enter number of points and seed'
      read(5,*)npoints,iseed
      ifk88seed=iseed
      rnd=fk88random(ifk88seed)
      do itype=itypel,itypeu
        do id=idl,idu
          id0=id
          if(id0.eq.7)id0=21
          avg=0.d0
          rmin=1.d8
          rmax=-1.d8
          do j=1,8
            avgn(j)=0.d0
            rnmin(j)=rmin
            rnmax(j)=rmax
          enddo
          do n=1,npoints
            rnd=fk88random(ifk88seed)
            st=rstlow+rnd*(rstupp-rstlow)
            rnd=fk88random(ifk88seed)
            xm=rxmlow+rnd*(rxmupp-rstlow)
            if(st.lt.stlow.or.st.gt.stupp .or.
     #         xm.lt.xmlow.or.xm.gt.xmupp)then
              write(*,*)'scales out of range',st,xm
              stop
            endif

            res_grid=pysudakov(st,xm,id0,itype,mcmass)
            res_pythia=py_compute_sudakov(st,xm,id0,itype,
     #                                    mcmass,stupp,
     #                                    iseedpy,xlowthrs,
     #                                    iunit1)
            rat=res_grid/res_pythia
            avg=avg+rat
            if(rat.lt.rmin)then
              rmin=rat
              stsv1=st
              xmsv1=xm
            endif
            if(rat.gt.rmax)then
              rmax=rat
              stsv2=st
              xmsv2=xm
            endif

            call invqnodeval(st,26,1.d0,10.d0,alst,q0st,j1st,j2st)
            call invqnodeval(xm,20,1.d0,10.d0,alxm,q0xm,j1xm,j2xm)
            st1 = qnodeval(j1st,26,1.d0,10.d0,alst,q0st)
            st2 = qnodeval(j2st,26,1.d0,10.d0,alst,q0st)
            xm1 = qnodeval(j1xm,20,1.d0,10.d0,alxm,q0xm)
            xm2 = qnodeval(j2xm,20,1.d0,10.d0,alxm,q0xm)

            res_mlo_stlo_grid=pysudakov(st1,xm1,id0,itype,mcmass)
            res_mlo_sthi_grid=pysudakov(st2,xm1,id0,itype,mcmass)
            res_mhi_stlo_grid=pysudakov(st1,xm2,id0,itype,mcmass)
            res_mhi_sthi_grid=pysudakov(st2,xm2,id0,itype,mcmass)
            res_mlo_stlo_pythia = py_compute_sudakov(st1,xm1,id0,itype,
     #                                               mcmass,stupp,
     #                                               iseedpy,xlowthrs,
     #                                               iunit1)
            res_mlo_sthi_pythia = py_compute_sudakov(st2,xm1,id0,itype,
     #                                               mcmass,stupp,
     #                                               iseedpy,xlowthrs,
     #                                               iunit1)
            res_mhi_stlo_pythia = py_compute_sudakov(st1,xm2,id0,itype,
     #                                               mcmass,stupp,
     #                                               iseedpy,xlowthrs,
     #                                               iunit1)
            res_mhi_sthi_pythia = py_compute_sudakov(st2,xm2,id0,itype,
     #                                               mcmass,stupp,
     #                                               iseedpy,xlowthrs,
     #                                               iunit1)

            rnrat(1) = res_mlo_stlo_grid/res_mlo_stlo_pythia
            rnrat(2) = res_mlo_sthi_grid/res_mlo_sthi_pythia
            rnrat(3) = res_mhi_stlo_grid/res_mhi_stlo_pythia
            rnrat(4) = res_mhi_sthi_grid/res_mhi_sthi_pythia

            rnrat(5) = res_grid/res_mlo_stlo_grid
            rnrat(6) = res_grid/res_mlo_sthi_grid
            rnrat(7) = res_grid/res_mhi_stlo_grid
            rnrat(8) = res_grid/res_mhi_sthi_grid

            do j=1,8
              avgn(j)=avgn(j)+rnrat(j)
              if(rnrat(j).lt.rnmin(j))then
                rnmin(j)=rnrat(j)
                stsv1n(j)=st
                xmsv1n(j)=xm
              endif
              if(rnrat(j).gt.rnmax(j))then
                rnmax(j)=rnrat(j)
                stsv2n(j)=st
                xmsv2n(j)=xm
              endif
            enddo

          enddo
          avg=avg/dfloat(npoints)
          do j=1,8
            avgn(j)=avgn(j)/dfloat(npoints)
          enddo
          write(60,*)'    '
          write(60,*)'------------------------'
          write(60,*)'Done:',id0,itype
          write(60,*)'    '
          write(60,*)'Central values'
          write(60,*)'<r>:',avg
          write(60,*)'rmin:',rmin
          write(60,*)'  at:',stsv1,xmsv1
          write(60,*)'rmax:',rmax
          write(60,*)'  at:',stsv2,xmsv2
          write(60,*)'    '
          write(60,*)'Nodal values (grid vs pythia)'
          do j=1,4
            write(60,*)' case j=',j
            write(60,*)'<r>:',avgn(j)
            write(60,*)'rmin:',rnmin(j)
            write(60,*)'  at:',stsv1n(j),xmsv1n(j)
            write(60,*)'rmax:',rnmax(j)
            write(60,*)'  at:',stsv2n(j),xmsv2n(j)
          enddo
          write(60,*)'    '
          write(60,*)'Nodal values (grid vs grid extrapolation)'
          do j=5,8
            write(60,*)' case j=',j
            write(60,*)'<r>:',avgn(j)
            write(60,*)'rmin:',rnmin(j)
            write(60,*)'  at:',stsv1n(j),xmsv1n(j)
            write(60,*)'rmax:',rnmax(j)
            write(60,*)'  at:',stsv2n(j),xmsv2n(j)
          enddo
        enddo
      enddo
      goto 400
 300  continue
      write(6,*)'enter id, itype'
      read(5,*)id,itype 
      write(6,*)'enter 1 to plot as a function of st'
      write(6,*)'      2 to plot as a function of xm'
      read(5,*)iplot
      iexcl=0
      write(6,*)'enter 1 to exclude Pythia from plots'
      write(6,*)'      0 otherwise'
      read(5,*)iexcl
      if(iexcl.eq.1)then
        npoints=500
      else
        npoints=100
      endif
      if(iplot.eq.1)then
        write(6,*)'enter xm'
        read(5,*)fixvar
        write(6,*)'enter stlow, stupp'
        read(5,*)pllow,plupp
      else
        write(6,*)'enter st'
        read(5,*)fixvar
        write(6,*)'enter xmlow, xmupp'
        read(5,*)pllow,plupp
      endif
      do n=0,npoints
        pl=pllow+n*(plupp-pllow)/dfloat(npoints)
        if(iplot.eq.1)then
          plres(n,1)=pysudakov(pl,fixvar,id,itype,mcmass)
          if(iexcl.eq.0)
     #      plres(n,2)=py_compute_sudakov(pl,fixvar,id,itype,
     #                                    mcmass,stupp,
     #                                    iseedpy,xlowthrs,
     #                                    iunit1)
        else
          plres(n,1)=pysudakov(fixvar,pl,id,itype,mcmass)
          if(iexcl.eq.0)
     #      plres(n,2)=py_compute_sudakov(fixvar,pl,id,itype,
     #                                    mcmass,stupp,
     #                                    iseedpy,xlowthrs,
     #                                    iunit1)
        endif
      enddo
      write(67,*)' set scale x log'
      do n=0,npoints
        pl=pllow+n*(plupp-pllow)/dfloat(npoints)
        write(66,*)pl,plres(n,1)
        write(67,*)pl,plres(n,1)
      enddo
      write(66,*)' join'
      write(67,*)' join'
      if(iexcl.eq.0)then
        do n=0,npoints
          pl=pllow+n*(plupp-pllow)/dfloat(npoints)
          write(66,*)pl,plres(n,2)
          write(67,*)pl,plres(n,2)
        enddo
      endif
      write(66,*)' set pattern .02 .09'
      write(66,*)' join patterned'
      write(67,*)' set pattern .02 .09'
      write(67,*)' join patterned'
      if(iplot.eq.1)then
        do inst=1,nnst
          write(66,*)stval(inst),-10.d0
          write(66,*)stval(inst),10.d0
          write(66,*)' set pattern .09 .09'
          write(66,*)' join patterned'
          write(67,*)stval(inst),-10.d0
          write(67,*)stval(inst),10.d0
          write(67,*)' set pattern .09 .09'
          write(67,*)' join patterned'
        enddo
      else
        do inxm=1,nnxm
          write(66,*)xmval(inxm),-10.d0
          write(66,*)xmval(inxm),10.d0
          write(66,*)' set pattern .09 .09'
          write(66,*)' join patterned'
          write(67,*)xmval(inxm),-10.d0
          write(67,*)xmval(inxm),10.d0
          write(67,*)' set pattern .09 .09'
          write(67,*)' join patterned'
        enddo
      endif
c
      write(66,*)' new plot'
      write(67,*)' new plot'
      write(66,*)' set scale y log'
      write(67,*)' set scale x log'
      write(67,*)' set scale y log'
      do n=0,npoints
        pl=pllow+n*(plupp-pllow)/dfloat(npoints)
        write(66,*)pl,plres(n,1)
        write(67,*)pl,plres(n,1)
      enddo
      write(66,*)' join'
      write(67,*)' join'
      if(iexcl.eq.0)then
        do n=0,npoints
          pl=pllow+n*(plupp-pllow)/dfloat(npoints)
          write(66,*)pl,plres(n,2)
          write(67,*)pl,plres(n,2)
        enddo
      endif
      write(66,*)' set pattern .02 .09'
      write(66,*)' join patterned'
      write(67,*)' set pattern .02 .09'
      write(67,*)' join patterned'
      if(iplot.eq.1)then
        do inst=1,nnst
          write(66,*)stval(inst),-10.d0
          write(66,*)stval(inst),10.d0
          write(66,*)' set pattern .09 .09'
          write(66,*)' join patterned'
          write(67,*)stval(inst),-10.d0
          write(67,*)stval(inst),10.d0
          write(67,*)' set pattern .09 .09'
          write(67,*)' join patterned'
        enddo
      else
        do inxm=1,nnxm
          write(66,*)xmval(inxm),-10.d0
          write(66,*)xmval(inxm),10.d0
          write(66,*)' set pattern .09 .09'
          write(66,*)' join patterned'
          write(67,*)xmval(inxm),-10.d0
          write(67,*)xmval(inxm),10.d0
          write(67,*)' set pattern .09 .09'
          write(67,*)' join patterned'
        enddo
      endif
c
      if(iexcl.eq.1)goto 400
      write(66,*)' new plot'
      write(67,*)' new plot'
      write(67,*)' set scale x log'
      do n=0,npoints
        pl=pllow+n*(plupp-pllow)/dfloat(npoints)
        write(66,*)pl,plres(n,1)/plres(n,2)
        write(67,*)pl,plres(n,1)/plres(n,2)
      enddo
      write(66,*)' join'
      write(67,*)' join'
      if(iplot.eq.1)then
        do inst=1,nnst
          write(66,*)stval(inst),-10.d0
          write(66,*)stval(inst),10.d0
          write(66,*)' set pattern .09 .09'
          write(66,*)' join patterned'
          write(67,*)stval(inst),-10.d0
          write(67,*)stval(inst),10.d0
          write(67,*)' set pattern .09 .09'
          write(67,*)' join patterned'
        enddo
      else
        do inxm=1,nnxm
          write(66,*)xmval(inxm),-10.d0
          write(66,*)xmval(inxm),10.d0
          write(66,*)' set pattern .09 .09'
          write(66,*)' join patterned'
          write(67,*)xmval(inxm),-10.d0
          write(67,*)xmval(inxm),10.d0
          write(67,*)' set pattern .09 .09'
          write(67,*)' join patterned'
        enddo
      endif
 400  continue
      close(iunit1)
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
c
      return
      end
c

c      function pysudakov(stlo,md,id,itype,mcmass)
c      implicit none
c      double precision pysudakov, stlo, sthi, md
c      integer id, itype
c      real*8 mcmass(21)
cc$$$      external SUDAKOV FUNCTION
c      external py_compute_sudakov
c      double precision py_compute_sudakov
c      double precision stupp, temp1, temp2
cc
c      stupp = 1000.0
c      sthi=stupp
c
c      temp1 = py_compute_sudakov(stlo,md,id,itype,mcmass,stupp)
c      temp2 = py_compute_sudakov(sthi,md,id,itype,mcmass,stupp)
c
c      pysudakov=temp1/temp2
c
c      return
c      end

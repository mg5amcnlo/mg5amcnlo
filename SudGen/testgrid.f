c gfortran -o testgrid testgrid.f sudakov.f dfint.f kerset.f
c
c Parameters set by hand equal to those in gridsudgen.f or given
c in input to that code.
c If changes occur there, they must be done here as well
      implicit real*8(a-h,o-z)
      real*8 mcmass(21)
      integer ifk88seed
      common/cifk88seed/ifk88seed
c
      do i=1,21
        mcmass(i)=0.d0
      enddo
      include 'MCmasses_PYTHIA8.inc'

      call dire_init(mcmass)

      call getalq0(26,1.d0,1.d0,100.d0,10.d0,alst,q0st)
      call getalq0(20,1.d0,2.d0,200.d0,10.d0,alxm,q0xm)
      write(6,*)'enter 0 for a manual test'
      write(6,*)'      1 for a random test'
      read(5,*)itest
      if(itest.eq.1)goto 200
      write(6,*)'enter id, itype'
      read(5,*)id,itype
 1    write(6,*)'enter st,xm'
      read(5,*)st,xm
      res=pysudakov(st,xm,id,itype,mcmass)
      write(6,*)'res=',res
      call invqnodeval(st,26,1.d0,10.d0,alst,q0st,j1st,j2st)
      call invqnodeval(xm,20,1.d0,10.d0,alxm,q0xm,j1xm,j2xm)
      write(6,*)'nearest nodes:',j1st,j2st,j1xm,j2xm
      write(6,*)'grid values at nodes:'
      write(6,*)ifakegrid(j1st,j1xm,id,itype)
      write(6,*)ifakegrid(j2st,j1xm,id,itype)
      write(6,*)ifakegrid(j1st,j2xm,id,itype)
      write(6,*)ifakegrid(j2st,j2xm,id,itype)
      write(6,*)'ratios to res:'
      r1=res/ifakegrid(j1st,j1xm,id,itype)
      r2=res/ifakegrid(j2st,j1xm,id,itype)
      r3=res/ifakegrid(j1st,j2xm,id,itype)
      r4=res/ifakegrid(j2st,j2xm,id,itype)
      write(6,*)r1
      write(6,*)r2
      write(6,*)r3
      write(6,*)r4
      write(6,*)'average of ratios:',(r1+r2+r3+r4)/4.d0
      goto 1
 200  continue
      write(6,*)'enter stlow'
      read(5,*)rstlow
      write(6,*)'enter xmlow, xmupp'
      read(5,*)rxmlow,rxmupp
      write(6,*)'enter number of points and seed'
      read(5,*)npoints,iseed
      ifk88seed=iseed
      rnd=fk88random(ifk88seed)
      do itype=1,4
        do id=1,7
          id0=id
          if(id0.eq.7)id0=21
          avg=0.d0
          rmin=1.d8
          rmax=-1.d8
          do n=1,npoints
            rnd=fk88random(ifk88seed)
            st=rstlow+rnd*(100.d0-rstlow)
            rnd=fk88random(ifk88seed)
            xm=rxmlow+rnd*(rxmupp-rstlow)
            res=pysudakov(st,xm,id0,itype,mcmass)
            call invqnodeval(st,26,1.d0,10.d0,alst,q0st,j1st,j2st)
            call invqnodeval(xm,20,1.d0,10.d0,alxm,q0xm,j1xm,j2xm)
            r1=res/ifakegrid(j1st,j1xm,id0,itype)
            r2=res/ifakegrid(j2st,j1xm,id0,itype)
            r3=res/ifakegrid(j1st,j2xm,id0,itype)
            r4=res/ifakegrid(j2st,j2xm,id0,itype)
            rr=(r1+r2+r3+r4)/4.d0
            avg=avg+rr
            if(rr.lt.rmin)then
              rmin=rr
              stsv1=st
              xmsv1=xm
            endif
            if(rr.gt.rmax)then
              rmax=rr
              stsv2=st
              xmsv2=xm
            endif
          enddo
          avg=avg/dfloat(npoints)
          write(60,*)'    '
          write(60,*)'done:',id0,itype
          write(60,*)'<r>:',avg
          write(60,*)'rmin:',rmin
          write(60,*)'  at:',stsv1,xmsv1
          write(60,*)'rmax:',rmax
          write(60,*)'  at:',stsv2,xmsv2
        enddo
      enddo
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

      function py_compute_sudakov(stlow,md,id,itype,mcmass,stupp)
      implicit none
      double precision py_compute_sudakov, stlow, stupp, md
      integer id, itype
      real*8 mcmass(21)
      double precision temp

      call dire_get_no_emission_prob(temp, stupp,
     #     stlow, md, id, itype)
      py_compute_sudakov=temp

      return
      end

      function pysudakov(stlo,md,id,itype,mcmass)
      implicit none
      double precision pysudakov, stlo, sthi, md
      integer id, itype
      real*8 mcmass(21)
c$$$      external SUDAKOV FUNCTION
      external py_compute_sudakov
      double precision py_compute_sudakov
      double precision stupp, temp1, temp2
c
      stupp = 1000.0
      sthi=stupp

      temp1 = py_compute_sudakov(stlo,md,id,itype,mcmass,stupp)
      temp2 = py_compute_sudakov(sthi,md,id,itype,mcmass,stupp)

      pysudakov=temp1/temp2

      return
      end

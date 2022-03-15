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
     $     ,itype_low,itype_upp,ipart_low,ipart_upp,inxm_low,inxm_upp
     $     ,ntrial,itrial,iev
      integer ifakegrid
      real*8 stlow,stupp,xmlow,xmupp,alst,q0st,alxm,q0xm,stval,xmval,
     $     stmax,stbound,qnodeval,qnodeval2,qnodeval3,aux1_qnode3
     $     ,kernelC,grids_tmp(nnst)
      
      real*8 emission_scales(10000),emission_weights(10000)
      double precision xlowthrs
      integer maxseed,iseed,iseedtopy,ifk88seed
      parameter (maxseed=100000)
      common/cifk88seed/ifk88seed
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
      call pythia_init(mcmass)

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
      read(*,*)ifk88seed
c Discard first (tends to be extremely small)
      if(ifk88seed.ge.1) rnd=fk88random(ifk88seed)

      write (*,*) 'enter lower and upper bound for itype to probe'
      write (*,*) '("1 4" loops over all)'
      read (*,*) itype_low, itype_upp
      if (itype_low.gt.itype_upp .or. itype_low.lt.1 .or.
     $     itype_upp.gt.4) then
         write (*,*) 'incorrect input'
         stop 1
      endif
      write (*,*) 'enter lower and upper bound for ipart to probe'
      write (*,*) '("1 7" loops over all)'
      read (*,*) ipart_low, ipart_upp
      if (ipart_low.gt.ipart_upp .or. ipart_low.lt.1 .or.
     $     ipart_upp.gt.4) then
         write (*,*) 'incorrect input'
         stop 1
      endif
      write (*,*) 'enter lower and upper bound for dipole mass bins '/
     $     /'to probe'
      write (*,*) '("1 50" loops over all)'
      read (*,*) inxm_low, inxm_upp
      if (inxm_low.gt.inxm_upp .or. inxm_low.lt.1 .or.
     $     inxm_upp.gt.nnxm) then
         write (*,*) 'incorrect input'
         stop 1
      endif
      write (*,*) 'enter number of trial showers'
      read(*,*) ntrial
      write (*,*) 'enter splitting kernel enhancement'
      read(*,*) kernelC
      
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

! setup grid nodes for dipole mass
      do inxm=inxm_low,inxm_upp
         if(inode.eq.1)then
            xm(inxm)=qnodeval(inxm,nnxm,xkxm,base,alxm,q0xm)
         elseif(inode.eq.2)then
            xm(inxm)=qnodeval2(inxm,nnxm,xkxm,xmlow,xmupp)
         elseif(inode.eq.3)then
            xm(inxm)=qnodeval3(inxm,nnxm,xkxm)
         endif
      enddo
      
! setup grid nodes for st
      do inst=1,nnst
         if(inode.eq.1)then
            st(inst)=qnodeval(inst,nnst,xkst,base,alst,q0st)
         elseif(inode.eq.2)then
            st(inst)=qnodeval2(inst,nnst,xkst,stlow,stupp)
         elseif(inode.eq.3)then
            st(inst)=qnodeval3(inst,nnst,xkst)
         endif
      enddo

      
      do itype=itype_low,itype_upp
         write(*,*)'===>Doing itype=',itype
         do ipart=ipart_low,ipart_upp
            write(*,*)'   --->Doing ipart=',ipart
            do inxm=inxm_low,inxm_upp
               if(inode.eq.1.or.inode.eq.2)then
                  xmval=xm(inxm)
               else
                  xmval=aux1_qnode3(xm(inxm),xmlow,xmupp)
               endif
               iseed=iseedtopy()
c     now I have a given itype, ipart and dipole mass
               do itrial=1,ntrial
                  
                  call py_compute_sudakov(emission_scales
     $                 ,emission_weights,xmval,ipmap(ipart),itype
     $                 ,stupp,stlow,kernelC,iseed)

                  do iev=1,10000
                     if (emission_scales(iev).lt.0d0) exit
                     do inst=1,nnst
                        if(inode.eq.1.or.inode.eq.2)then
                           stval=st(inst)
                        else
                           stmax=stbound(xmval,stupp
     $                          ,mcmass(ipmap(ipart)),itype
     $                          ,ipmap(ipart))
                           stval=aux1_qnode3(st(inst),stlow,stmax)
                        endif
                        if (emission_scales(iev).gt.stval) then
                           grids_tmp(inst)=
     $                          grids_tmp(inst)+emission_weights(iev)
                        endif
                     enddo
                  enddo
                  grids(1:nnst,inxm,ipart,itype)=
     $                 grids(1:nnst,inxm,ipart,itype)+grids_tmp(inst)
               enddo
               grids(1:nnst,inxm,ipart,itype)=
     $              grids(1:nnst,inxm,ipart,itype)/dble(ntrial)
c$$$               do inst=1,nnst
c$$$                  if (grids(inst,inxm,ipart,itype).lt.xlowthrs) then
c$$$                     grids(inst,inxm,ipart,itype)=0d0
c$$$                  endif
c$$$               endif

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


      subroutine py_compute_sudakov(emission_scales,emission_weights,md
     $     ,id,itype,stupp,stlow,kernelC,iseed)
      implicit none
      double precision stlow,stupp,md,emission_scales(10000)
     $     ,emission_weights(10000),kernelC
      integer id,itype,iseed
      double precision em_scales(10000),em_wgts(10000)
c
      call pythia_get_no_emission_prob(em_scales,em_wgts, stupp, stlow,
     $     md, id, itype, kernelC,iseed)
      emission_scales=em_scales
      emission_weights=em_wgts
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


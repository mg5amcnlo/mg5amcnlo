C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      call inihist
      call mbook(1,'total rate',1.0e0,0.5e0,5.5e0)
      call mbook(2,'ttb inv m',8e0,300e0,1100e0)
      call mbook(3,'ttb inv m 2',8e0,300e0,1100e0)
      call mbook(4,'g energy',12e0,-100e0,1100e0)
      call mbook(5,'sqrt(shat)',1e0,990e0,1010e0)
      call mbook(6,'log10(pt(ttb))',0.04e0,0e0,4e0)
      end


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      implicit none
      real*8 xnorm
      integer i,k,ievt
      character*14 ytit
      open(unit=99,name='PYTEETT.top',status='unknown')
      xnorm=1.d0/ievt
      do i=1,100
 	call mfinal3(i) 
        call mcopy(i,i+100)
        call mopera(i+100,'F',i+100,i+100,sngl(xnorm),0.e0)
 	call mfinal3(i+100)             
      enddo                          
      k=0
      ytit='sigma oer bin'
      call multitop(100+k+1,99,2,3,'total rate',ytit,'LOG')
      call multitop(100+k+2,99,2,3,'ttb inv m',ytit,'LOG')
      call multitop(100+k+3,99,2,3,'ttb inv m 2',ytit,'LOG')
      call multitop(100+k+4,99,2,3,'g energy',ytit,'LOG')
      call multitop(100+k+5,99,2,3,'sqrt(shat)',ytit,'LOG')
      call multitop(100+k+6,99,2,3,'log10(pt(ttb))',ytit,'LOG')

      close(99)
      end


C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit none
      integer ichsum,ichini,ihep
      integer id1,ist,iq1,iq2,it1,it2,it3,it4,j,ij,ig,iglu,
     &id,iori
      real*8 www0
      integer pychge
      external pydata
      integer n,npad,k,mstu,mstj,kchg,mdcy,mdme,kfdp,msel,mselpd,msub,
     &kfin,mstp,msti
      double precision p,v,paru,parj,pmas,parf,vckm,brat,ckin,parp,pari
      integer kf1,kf2,i,it5,it6

      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)

      logical test1,test2,test5
      double precision evweight
      common/cevweight/evweight
      double precision tiny
      parameter (tiny=3.d-5)
      double precision p1(5),p2(5),psum(5),ptq(50,5),ptb(50,5),
     &pihep(5),p12(5),psum12(5),psum34(5),pglu(5),invm,invm2,shat,
     &ptop(4,5),mtop32,mtop42,vdot,virtq,virtb,virtq2,virtb2,mtop,
     &alpha3,alpha4,beta3,beta4,var,virt3,virt4,ptttb
c incoming partons may travel in the same direction: it's a power-suppressed
c effect, so throw the event away
      if(sign(1.d0,p(3,3)).eq.sign(1.d0,p(4,3)))then
         write(*,*)'warning 111 in pyanal'
        goto 999
      endif
      www0=evweight
c initialization
      do ij=1,5
         do i=1,50
            ptq(i,ij)=0d0
            ptb(i,ij)=0d0
         enddo
         do i=1,4
            ptop(i,ij)=0d0
         enddo
         pglu(ij)=0d0
         p1(ij)=0d0
         p2(ij)=0d0
         psum(ij)=0d0
         psum12(ij)=0d0
         psum34(ij)=0d0
         p12(ij)=0d0
         pihep(ij)=0d0
      enddo

      ig=0
      iglu=0
      ihep=0
      iq1=0
      iq2=0

      do j=1,4
         p1(j)=p(1,j)
         p2(j)=p(2,j)
      enddo
      call vvsum(4,p1,p2,psum)
      call vsca(4,-1d0,psum,psum)
      ichsum=0
      kf1=k(1,2)
      kf2=k(2,2)
      ichini=pychge(kf1)+pychge(kf2)

c loop over particles in the event
      do ihep=1,n
        ist=k(ihep,1)      
        id1=k(ihep,2)
        iori=k(ihep,3)
        do j=1,5
           pihep(j)=p(ihep,j)
        enddo
        if(ist.le.10)then
           call vvsum(4,pihep,psum,psum)
           ichsum=ichsum+pychge(id1)
        endif
c test for top and antitop
        test1=id1.eq. 6
        test2=id1.eq.-6
c test for extra gluon
        test5=id1.eq.21.and.iori.eq.0.and.ist.eq.21.and.ig.eq.0
c
        if(test1)then
           iq1=iq1+1
           do j=1,5
              ptq(iq1,j)=p(ihep,j)
           enddo
        endif
        if(test2)then
           iq2=iq2+1
           do j=1,5
              ptb(iq2,j)=p(ihep,j)
           enddo
        endif
        if(test5)then
           ig=ig+1
           iglu=ihep
        endif
      enddo

c check momentum and charge conservation
      if (vdot(3,psum,psum).gt.1.e-4*p(1,4)**2) then
         write(*,*)'warning 112 in pyanal',
     &         vdot(3,psum,psum),1.e-4*p(1,4)**2
         goto 999
      endif
      if (ichsum.ne.ichini) then
         write(*,*)'error 113 in pyanal'
         stop
      endif

c fill the four-momenta
      do j=1,5
         pglu(j)=p(iglu,j)
         p12(j)=p1(j)+p2(j)
         ptop(1,j)=ptq(1,j)
         ptop(2,j)=ptb(1,j)
      enddo
      do i=1,49
         if(ptq(i+1,1).eq.0d0.and.ptq(i+1,2).eq.0d0
     & .and.ptq(i+1,3).eq.0d0.and.ptq(i+1,4).eq.0d0)then
            do j=1,5
               ptop(3,j)=ptq(i,j)
            enddo
            goto 333
         endif
      enddo
 333  continue
      do i=1,49
         if(ptb(i+1,1).eq.0d0.and.ptb(i+1,2).eq.0d0
     & .and.ptb(i+1,3).eq.0d0.and.ptb(i+1,4).eq.0d0)then
            do j=1,5
               ptop(4,j)=ptb(i,j)
            enddo
            goto 334
         endif
      enddo
 334  continue


c$$$      do i=1,10
c$$$         write(*,*)(ptq(i,j),j=1,5)
c$$$      enddo
c$$$      do i=1,10
c$$$         write(*,*)(ptb(i,j),j=1,5)
c$$$      enddo
c$$$      write(*,*)'  '
c$$$      write(*,*)(ptop(1,j),j=1,5)
c$$$      write(*,*)(ptop(2,j),j=1,5)
c$$$      write(*,*)(ptop(3,j),j=1,5)
c$$$      write(*,*)(ptop(4,j),j=1,5)
c$$$      write(*,*)'  '
c$$$      write(*,*)(pglu(j),j=1,5),' pglu'
c$$$      write(*,*)'  '

      do j=1,4
         psum12(j)=ptop(1,j)+ptop(2,j)
         psum34(j)=ptop(3,j)+ptop(4,j)
      enddo

      var=1d0
      shat=p12(4)**2-p12(1)**2-p12(2)**2-p12(3)**2
      invm=sqrt(psum12(4)**2-psum12(1)**2-psum12(2)**2-psum12(3)**2)
      invm2=sqrt(psum34(4)**2-psum34(1)**2-psum34(2)**2-psum34(3)**2)
      ptttb=sqrt(psum34(1)**2+psum34(2)**2)

c$$$      write(*,*)sqrt(shat),invm,invm2,ptttb
c$$$      write(*,*)' '


      if(abs(invm-sqrt(shat-2*sqrt(shat)*pglu(4)))/invm.ge.tiny)then
         write(*,*)'wrong invm ',invm,sqrt(shat-2*sqrt(shat)*pglu(4)),
     &             abs(invm-sqrt(shat-2*sqrt(shat)*pglu(4)))/invm
         stop
      endif

      call mfill(1,sngl(var),sngl(www0))
      call mfill(2,sngl(invm),sngl(www0))
      call mfill(3,sngl(invm2),sngl(www0))
      call mfill(4,sngl(pglu(4)),sngl(www0))
      call mfill(5,sngl(sqrt(shat)),sngl(www0))
      call mfill(6,sngl(log10(ptttb)),sngl(www0))


 999  return
      end

c-----------------------------------------------------------------------
      subroutine vvsum(n,p,q,r)
c-----------------------------------------------------------------------
c    vector sum
c-----------------------------------------------------------------------
      implicit none
      integer n,i
      double precision p(n),q(n),r(n)
      do 10 i=1,n
   10 r(i)=p(i)+q(i)
      end



c-----------------------------------------------------------------------
      subroutine vsca(n,c,p,q)
c-----------------------------------------------------------------------
c     vector times scalar
c-----------------------------------------------------------------------
      implicit none
      integer n,i
      double precision c,p(n),q(n)
      do 10 i=1,n
   10 q(i)=c*p(i)
      end



c-----------------------------------------------------------------------
      function vdot(n,p,q)
c-----------------------------------------------------------------------
c     vector dot product
c-----------------------------------------------------------------------
      implicit none
      integer n,i
      double precision vdot,pq,p(n),q(n)
      pq=0.
      do 10 i=1,n
   10 pq=pq+p(i)*q(i)
      vdot=pq
      end

C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
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
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      real*8 xnorm
      integer i,k,ievt
      character*14 ytit
      open(unit=99,name='HWTEETT.top',status='unknown')
      xnorm=1.d3/dfloat(nevhep)
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
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      integer ichsum,ichini,ihep
      integer id1,ist,iq1,iq2,it1,it2,it3,it4,j,ij,ig,iglu,
     &id
      real*8 www0
      logical test1,test2,test3,test4,test5,didsof
      double precision tiny,hwvdot,var
      parameter (tiny=3.d-5)
      double precision p1(5),p2(5),psum(5),ptop(4,5),
     &p12(5),psum12(5),psum34(5),pglu(5),invm,invm2,shat,
     &ptttb

      IF (IERROR.NE.0) RETURN
c
c incoming partons may travel in the same direction: it's a power-suppressed
c effect, so throw the event away

      if(sign(1.d0,phep(3,4)).eq.sign(1.d0,phep(3,5)))then
        call hwwarn('hwanal',111)
        goto 999
      endif
      www0=evwgt
      call hwvsum(4,phep(1,1),phep(1,2),psum)
      call hwvsca(4,-1d0,psum,psum)
      ichsum=0
      ichini=ichrg(idhw(1))+ichrg(idhw(2))
      didsof=.false.

      do ij=1,5
         do j=1,4
            ptop(j,ij)=0d0
         enddo
         pglu(ij)=0d0
         p12(ij)=0d0
         p1(ij)=0d0
         p2(ij)=0d0
         psum(ij)=0d0
         psum12(ij)=0d0
         psum34(ij)=0d0
      enddo

      ig=0
      iglu=0
      ihep=0
      it1=0
      it2=0
      it3=0
      it4=0
      iq1=0
      iq2=0

      do ihep=1,nhep
        if (idhw(ihep).eq.16) didsof=.true.
        if (isthep(ihep).eq.1) then
          call hwvsum(4,phep(1,ihep),psum,psum)
          ichsum=ichsum+ichrg(idhw(ihep))
        endif
        ist=isthep(ihep)      
        id=idhw(ihep)
        id1=idhep(ihep)

c test for hard-event top and tbar
        test1=id1.eq. 6.and.(ist.eq.123.or.ist.eq.124).and.ihep.le.11
        test2=id1.eq.-6.and.(ist.eq.123.or.ist.eq.124).and.ihep.le.11
c test for final top and tbar
        test3=id1.eq. 6.and.ist.eq.155
        test4=id1.eq.-6.and.ist.eq.155
c test for extra gluon
        test5=id1.eq.21.and.(ist.eq.123.or.ist.eq.124).and.ihep.le.11
c
        if(test1)then
           iq1=iq1+1
           it1=ihep
        endif
        if(test2)then
           iq2=iq2+1
           it2=ihep
        endif
        if(test3)then
           it3=ihep
        endif
        if(test4)then
           it4=ihep
        endif
        if(test5)then
           ig=ig+1
           iglu=ihep
        endif

      enddo

c check momentum and charge conservation
      if (hwvdot(3,psum,psum).gt.1.e-4*phep(4,1)**2) then
         call hwuepr
         call hwwarn('hwanal',112)
         goto 999
      endif
      if (ichsum.ne.ichini) then
         call hwuepr
         call hwwarn('hwanal',113)
         goto 999
      endif
c fill the four-momenta
      do j=1,5
         ptop(1,j)=phep(j,it1)
         ptop(2,j)=phep(j,it2)
         ptop(3,j)=phep(j,it3)
         ptop(4,j)=phep(j,it4)
         pglu(j)=phep(j,iglu)
         psum12(j)=ptop(1,j)+ptop(2,j)
         psum34(j)=ptop(3,j)+ptop(4,j)
         p1(j)=phep(j,1)
         p2(j)=phep(j,2)
         p12(j)=p1(j)+p2(j)
      enddo

      var=1d0
      shat=p12(4)**2-p12(1)**2-p12(2)**2-p12(3)**2
      invm=sqrt(psum12(4)**2-psum12(1)**2-psum12(2)**2-psum12(3)**2)
      invm2=sqrt(psum34(4)**2-psum34(1)**2-psum34(2)**2-psum34(3)**2)
      ptttb=sqrt(psum34(1)**2+psum34(2)**2)

c$$$      write(*,*)it3,it4
c$$$      write(*,*)(ptop(3,j),j=1,4)
c$$$      write(*,*)(ptop(4,j),j=1,4)
c$$$      write(*,*)(psum34(j),j=1,4)


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

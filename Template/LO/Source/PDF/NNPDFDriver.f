***
*
* NNPDF Fortran Driver
*
* Stefano Carrazza for the NNPDF Collaboration
* email: stefano.carrazza@mi.infn.it
*
* February 2013
*
* Usage:
*
*  NNPDFDriver("gridname.LHgrid");
*
*  NNinitPDF(0); // select replica [0,Mem]
*
*  NNevolvePDF(x,Q,pdf); // -> returns double array (-6,7)
*     
      subroutine NNPDFDriver(gridfilename)
      implicit none
      
      integer nfl,nx,nq2,mem,rep
      double precision alphas
      double precision xgrid(100),logxgrid(100)
      double precision q2grid(60),logq2grid(60)
      double precision pdfgrid(0:100,14,100,60)
      logical hasphoton
      common /nnpdf/nfl,nx,nq2,mem,rep,hasphoton,alphas,xgrid,logxgrid,
     1     q2grid,logq2grid,pdfgrid

      character*(*) gridfilename
*
      nfl = 13
      nx = 100
      nq2 = 60
      mem = 1
      rep = 0
      alphas = 0
*
*     Logo
      write(6,*) " ****************************************"
      write(6,*) ""
      write(6,*) "      NNPDFDriver version 1.0.3"
      write(6,*) "  Grid: ", gridfilename
      write(6,*) " ****************************************"

      call readPDFSet(gridfilename)
      
      end subroutine

      subroutine NNinitPDF(irep)
      implicit none
      integer irep

      integer nfl,nx,nq2,mem,rep
      double precision alphas
      double precision xgrid(100),logxgrid(100)
      double precision q2grid(60),logq2grid(60)
      double precision pdfgrid(0:100,14,100,60)
      logical hasphoton
      common /nnpdf/nfl,nx,nq2,mem,rep,hasphoton,alphas,xgrid,logxgrid,
     1     q2grid,logq2grid,pdfgrid

      if (irep.gt.mem.or.irep.lt.0d0) then
         write(6,*) "Error: replica out of range [0,",mem,"]"
      else
         rep = irep
      endif
      
      end subroutine

      subroutine readPDFSet(gridfilename)
      implicit none
      
      integer i,ix,iq,fl,imem
      character*(*) gridfilename
      character*100 line
*
      integer nfl,nx,nq2,mem,rep
      double precision alphas
      double precision xgrid(100),logxgrid(100)
      double precision q2grid(60),logq2grid(60)
      double precision pdfgrid(0:100,14,100,60)
      logical hasphoton
      common /nnpdf/nfl,nx,nq2,mem,rep,hasphoton,alphas,xgrid,logxgrid,
     1     q2grid,logq2grid,pdfgrid
      integer IU
      common/IU/IU

*

      call OpenData(gridfilename)

*     Read header
      do i=1,1000
         read(IU,*) line
         if (line(1:14).eq.'Parameterlist:') then
            read(IU,*) line, mem, line, alphas                        
            exit
         endif
      enddo

*     Select driver
      do i=1,1000
         read(IU,*) line
         if (line(1:13).eq.'NNPDF20intqed') then
            hasphoton = .true.
            nfl = nfl + 1   
            read(IU,*) line,line            
            exit
         endif
         if (line(1:13).eq.'NNPDF20int') then
            hasphoton = .false.            
            read(IU,*) line,line            
            exit
         endif
      enddo
*
      read(IU,*) nx
      do ix=1,nx
         read(IU,*) xgrid(ix)
         logxgrid(ix) = dlog(xgrid(ix))
      enddo
*
      read(IU,*) nq2
      read(IU,*) line
      do iq=1,nq2
         read(IU,*) q2grid(iq)
         logq2grid(iq) = dlog(q2grid(iq))
      enddo
*            
      read(IU,*) line      
      do imem=0,mem
         do ix=1,nx
            do iq=1,nq2               
               read(IU,*) ( pdfgrid(imem,fl,ix,iq), fl=1,nfl,1)            
            enddo
         enddo
      enddo

      close(IU)
      
      end subroutine

      subroutine NNevolvePDF(x,Q,xpdf)
      implicit none
      
      integer i,j,ix,iq2,M,N,ipdf,fmax
      integer minx,maxx,midx
      integer minq,maxq,midq
      double precision x,Q,xpdf(-6:7),Q2
      double precision xmingrid,xch,x2,x1,dy,y
      parameter (M=4, N=2)
      parameter (xmingrid=1d-7, xch=1d-1) 

      integer nmax,mmax 
      parameter(nmax=1e3,mmax=1e3)            
      integer ix1a(mmax), ix2a(nmax)
      double precision x1a(mmax), x2a(nmax)
      double precision ya(mmax,nmax)

      integer nfl,nx,nq2,mem,rep
      double precision alphas
      double precision xgrid(100),logxgrid(100)
      double precision q2grid(60),logq2grid(60)
      double precision pdfgrid(0:100,14,100,60)
      logical hasphoton
      common /nnpdf/nfl,nx,nq2,mem,rep,hasphoton,alphas,xgrid,logxgrid,
     1     q2grid,logq2grid,pdfgrid
      

      Q2 = Q*Q
*     check bounds
      if (x.lt.xmingrid.or.x.lt.xgrid(1).or.x.gt.xgrid(nx)) then
c$$$         write(6,*) "Parton interpolation: x out of range -- freezed"
         if (x.lt.xgrid(1)) x = xgrid(1)
*         if (x.lt.xmingrid) x = xmingrid
         if (x.gt.xgrid(nx))x = xgrid(nx)
      endif
      if (Q2.lt.q2grid(1).or.Q2.gt.q2grid(nq2)) then
c$$$         write(6,*) "Parton interpolation: Q2 out of range -- freezed"
c$$$         write(6,*) "Q2 = ",Q2, " GeV2", q2grid(1)
         if (Q2.lt.q2grid(1)) Q2 = q2grid(1)
         if (Q2.gt.q2grid(nq2)) Q2 = q2grid(nq2)
      endif

      minx = 1
      maxx = NX+1
 10   continue
      midx = (minx+maxx)/2
      if (x.lt.xgrid(midx)) then
         maxx=midx
      else
         minx=midx
      endif
      if ((maxx-minx).gt.1) go to 10
      ix = minx

      minq = 1
      maxq = nq2+1
 20   continue
      midq = (minq+maxq)/2
      if (Q2.lt.q2grid(midq)) then
         maxq=midq
      else
         minq=midq
      endif 
      if ((maxq-minq).gt.1) go to 20
      iq2 = minq

*     Assign grid for interpolation. M, N -> order of polyN interpolation      
      do I=1,M
         if(IX.ge.M/2.and.IX.le.(NX-M/2)) IX1A(I) = IX - M/2 + I
         if(IX.lt.M/2) IX1A(I) = I
         if(IX.gt.(NX-M/2)) IX1A(I) = (NX - M) + I
         
*     Check grids
         if(IX1A(I).le.0.or.IX1A(I).gt.NX) then
            write(6,*) "Error in grids! "
            write(6,*) "I, IXIA(I) = ",I, IX1A(I)
            call exit(-10)
         endif
      enddo

      do J=1,N
         if(IQ2.ge.N/2.and.IQ2.le.(NQ2-N/2)) IX2A(J) = IQ2 - N/2 + J
         if(IQ2.lt.N/2) IX2A(J) = J
         if(IQ2.gt.(NQ2-N/2)) IX2A(J) = (NQ2 - N) + J
*     Check grids
         if(IX2A(J).le.0.or.IX2A(J).gt.NQ2) then
            write(6,*) "Error in grids! "
            write(6,*) "J, IXIA(J) = ",J,IX2A(J)
            call exit(-10)
         endif
      enddo
            
*     Define points where to evaluate interpolation
*     Choose between linear or logarithmic (x,Q2) interpolation

      IF(X.LT.XCH)THEN
         X1=dlog(X)          
      ELSE
         X1=X
      ENDIF
      X2=dlog(Q2)

*     initialize output vector
      do i=-6,7 
         xpdf(i) = 0
      enddo
            
      fmax = 6
      if (nfl.eq.14) fmax=7

      DO IPDF = -6,fmax,1                 
*     Choose between linear or logarithmic (x,Q2) interpolation        
         DO I=1,M
            IF(X.LT.XCH)THEN
               X1A(I)= logxgrid(IX1A(I))
            ELSE
               X1A(I)= xgrid(IX1A(I))
            ENDIF
            DO J=1,N
               X2A(J) = logq2grid(IX2A(J))
               YA(I,J) = pdfgrid(REP,IPDF+7,IX1A(I),IX2A(J))               
            enddo
         enddo
         
!     2D polynomial interpolation
         call lh_polin2(x1a,x2a,ya,m,n,x1,x2,y,dy)
         XPDF(IPDF) = y
      enddo                 

      end subroutine

      subroutine lh_polin2(x1a,x2a,ya,m,n,x1,x2,y,dy) 
      implicit none 
!                                                                       
      integer m,n,nmax,mmax 
      integer j,k 
      parameter(nmax=1e3,mmax=1e3) 
                                                                        
      real*8 dy,x1,x2,y,x1a(mmax),x2a(nmax),ya(mmax,nmax) 
      real*8 ymtmp(nmax),yntmp(nmax) 
                                                                        
      do j=1,m 
         do k=1,n 
            yntmp(k)=ya(j,k) 
         enddo 
         call lh_polint(x2a,yntmp,n,x2,ymtmp(j),dy) 
      enddo 
      call lh_polint(x1a,ymtmp,m,x1,y,dy) 
!                                                                       
      return 
      END                                           

      subroutine lh_polint(xa,ya,n,x,y,dy) 
      implicit none 
!                                                                       
      integer n,NMAX 
!     Largest anticipated value of n                                    
      parameter(nmax=1e3) 
      real*8 dy,x,y,xa(nmax),ya(nmax) 
      integer i,m,ns 
      real*8 den,dif,dift,ho,hp,w,c(nmax),d(nmax) 
      ns=1 
      dif=abs(x-xa(1)) 
      do 11 i=1,n 
         dift=abs(x-xa(i)) 
         if(dift.lt.dif) then 
            ns=i 
            dif=dift 
         endif 
         c(i)=ya(i) 
         d(i)=ya(i) 
   11 enddo 
      y=ya(ns) 
      ns=ns-1 
      do m=1,n-1 
         do i=1,n-m 
            ho=xa(i)-x 
            hp=xa(i+m)-x 
            w=c(i+1)-d(i) 
            den=ho-hp 
            if(den.eq.0) then 
               write(*,*)'failure in polint' 
               stop 
            endif 
            den=w/den 
            d(i)=hp*den 
            c(i)=ho*den 
         enddo 
         if(2*ns.lt.(n-m)) then 
            dy=c(ns+1) 
         else 
            dy=d(ns) 
            ns=ns-1 
         endif 
         y=y+dy 
      enddo 
                                                                        
      return 
      END

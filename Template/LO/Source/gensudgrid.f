      program gensudgrid
      implicit none

      include 'PDF/pdf.inc'
      include 'sudgrid.inc'
      include 'run.inc'
      include 'cuts.inc'
      
c...global variables
      double precision bthres,alam5,alam4,s,x1,x2
      integer kfl
      common/sudpars/bthres,alam5,alam4,s,x1,x2,kfl

c...local variables
      double precision pt2min,pt2max,x1min,x1max,x2min,x2max,pt2,x1tmp
      double precision calcissud,reslast,result,relerr,eps
      data eps/0.001/
      data pt2min/4d0/
      data x1min,x1max/1d-6,0.99/
      data x2min,x2max/1d-6,0.99/
      integer nfnevl,ifail,i,j,ipt2,ix1,ix2,ncalls,n,iargc
      data ncalls/0/
      character*8 arg

      n=iargc()
      if(n.eq.0)then
         write(*,*) 'Usage:'
         write(*,*) 'gensudgrid id'
         write(*,*) 'where id is a parton id, from -2 to 5'
         stop
      endif
      call getarg(1,arg)
      read(arg,*) kfl
 
c...Read the run_card.dat
      call setrun

      call pdfwrap

      s=(ebeam(1)+ebeam(2))**2-(ebeam(1)-ebeam(2))**2

      pt2max=0.25*s/4

      if(xqcut**2 .lt. pt2min) pt2min = xqcut**2

      write(*,'(''#'',a)') 'Sudakov grid for Delta(ptmax,ptEmin,x1,x2)'
      write(*,'(''#'',a1,a7,a)') '''',pdlabel,''' = pdlabel'
      if (pdlabel.eq.'''lhapdf''')
     $           write(*,'(''#'',i10,a)') lhaid,' = lhaid'
      write(*,'(''#'',f6.0,a)') ebeam(1),' = ebeam1'
      write(*,'(''#'',f6.0,a)') ebeam(2),' = ebeam2'
      write(*,'(''#'',f8.5,a)') sqrt(pt2min),' = ptEmin'

c      do kfl=-2,5
      if(kfl.eq.5) pt2min=bthres
      write(*,'(''#'',i4,a)') kfl,' = kfl'
      write(*,'(''#'',a11,3a12,2a7,a3)') 'x2','x1','ptmax','sud',
     $   'ncalls','relerr','ierr'
      do ix2=1,nx2
        x2=10**(log10(x2min)+
     $     (ix2-1)*(log10(x2max)-log10(x2min))/(nx2-1))
        do ix1=1,nx1
          if(ix1.le.nx1/4) then
            x1=10**(log10(x1min)+
     $         (ix1-1)*(log10(x1max)-log10(x1min))/(nx1-1))
          elseif(ix1.le.3*nx1/5) then
            x1tmp=10**(log10(x1min)+
     $         (nx1/4-1)*(log10(x1max)-log10(x1min))/(nx1-1))
            x1=(sqrt(x1min)+(ix1-nx1/4)*
     $         (sqrt(x1max)-sqrt(x1min))/(nx1-nx1/4))**2
          else
            x1tmp=10**(log10(x1min)+
     $         (nx1/4-1)*(log10(x1max)-log10(x1min))/(nx1-1))
            x1tmp=(sqrt(x1tmp)+(3*nx1/5-nx1/4)*
     $         (sqrt(x1max)-sqrt(x1tmp))/(nx1-nx1/4))**2
            x1=x1tmp+(ix1-3*nx1/5)*(x1max-x1tmp)/(nx1-3*nx1/5)          
          endif
          ifail=0
          reslast=0d0
          do ipt2=1,npt2
            pt2=10**(log10(pt2min)+
     $         (ipt2-1)*(log10(pt2max)-log10(pt2min))/(npt2-1))
            if(kfl.eq.5.and.x1.gt.0.6)then
              result=0d0
              nfnevl=0
              relerr=0
              ifail=0
              goto 100              
            endif
            if(kfl.eq.5.and.pt2.lt.22.3109d0)then
              result=1d0
              nfnevl=0
              relerr=0
              ifail=0
              goto 100              
            endif
            if(pt2.le.pt2min)then
              result=1d0
              nfnevl=0
              relerr=0
              ifail=0
              goto 100
            endif
            if(ifail.gt.0)then
              nfnevl=0
              relerr=0
              goto 100
            endif
            result=calcissud(kfl,x1,x2,pt2min,pt2,nfnevl,relerr,ifail)
            if(abs(result-reslast)/abs(result).lt.eps)
     $         ifail=3
            reslast=result
            ncalls=ncalls+1
 100        write(*,'(4e12.5,i7,f7.4,i3)') x2,x1,sqrt(pt2),
     $         result,nfnevl,relerr,ifail
          enddo
        enddo
      enddo
c      enddo

      write(*,'(''#'',a,i6,a)') 'Made ',ncalls,' calls to calcissud'

      end


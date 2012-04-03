      subroutine qltri6(s,m2sq,m3sq,musq,Ires)
C-----calculate the IR divergent box in DIM reg
C-----Note choosing the cutoff equal to musq, we exactly get
C-----the e=0, dim reg result 
      implicit none
      include 'qlconstants.f'
      double precision s,m2sq,m3sq,musq,m2,m3,iepsd,rexs,imxs
      double complex Ires(-2:0),cxs(3),fac,xlog,cln,qlcLi2omx2 
      logical qlzero
      
      m2 = sqrt(m2sq) 
      m3 = sqrt(m3sq) 
      
C     ieps gives the sign of the imaginary part of K   
      call qlkfn(cxs,iepsd,s,m2,m3)    
      fac=dcmplx(1d0/(m2*m3))*cxs(1)/(cxs(2)*cxs(3))
      xlog = cln(cxs(1),iepsd)


      rexs=dreal(cxs(1)) 
      imxs=dimag(cxs(1))
C----deal with s=(m2-m3)^2
      if ((qlzero(rexs-1d0)) .and. (qlzero(imxs))) then
      fac=dcmplx(0.5d0/(m2*m3))
      Ires(-2)=czip
      Ires(-1)=fac
      if (qlzero(m2-m3)) then
      Ires(0)=fac*dcmplx(log(musq/(m2*m3)))
      else
      Ires(0)=fac*(dcmplx(log(musq/(m2*m3)))-ctwo
     .        -dcmplx((m3+m2)/(m3-m2)*log(m2/m3)))
      endif
      else
C----deal with s .ne. (m2-m3)^2
      Ires(-2)=  czip
      Ires(-1)= -fac*xlog
      Ires(0) = fac*(xlog*(-chalf*xlog
     . +dcmplx(log(m2*m3/musq)))
     . -qlcLi2omx2(cxs(1),cxs(1),iepsd,iepsd)
     . +chalf*dcmplx(log(m2/m3)**2)
     . +qlcLi2omx2(cxs(1),dcmplx(m2/m3),iepsd,0d0)
     . +qlcLi2omx2(cxs(1),dcmplx(m3/m2),iepsd,0d0))
      endif

      
      return
      end

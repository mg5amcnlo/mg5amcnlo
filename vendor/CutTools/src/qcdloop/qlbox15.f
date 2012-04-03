      subroutine qlbox15(Y,musq,Ires)
      implicit none
C 15:
C I_4^({D=4-2\e})(m_1^2,\pd^2,\pt^2,m_4^2;s_{12},s_{23};0,m_2^2,0,m_4^2)

c       [                                  s12                         ]
c       [   0             0              - ---              0          ]
c       [                                   2                          ]
c       [                                                              ]
c       [                             m2sq - p2sq  - s23 + m4sq + m2sq ]
c       [   0           m2sq          -----------  ------------------- ]
c       [                                  2                2          ]
c y15 = [                                                              ]
c       [   s12      m2sq - p2sq                       m4sq - p3sq     ]
c       [ - ---      ----------           0           -----------     ]
c       [    2            2                                 2          ]
c       [                                                              ]
c       [        - s23 + m4sq + m2sq  m4sq - p3sq                      ]
c       [   0    -------------------  -----------         m4sq         ]
c       [                 2                2                           ]

C----Implementation of Bennakker-Denner Eq.2.11
C----\bibitem{Beenakker:1988jr}
C----W.~Beenakker and A.~Denner,
C----%``INFRARED DIVERGENT SCALAR BOX INTEGRALS WITH APPLICATIONS IN THE
C----%ELECTROWEAK STANDARD MODEL,''
C----Nucl.\ Phys.\  B {\bf 338}, 349 (1990).

      include 'qlconstants.f'
      integer iep
      logical qlzero,qlnonzero
      double precision Y(4,4),m2sq,m4sq,musq,m2,m4,ta,
     . si,m2sqbar,m4sqbar,ieps,iepyy,iepyi,yy,yi,imxs,rexs
      double complex xs,cxs(3),qlcLi2omx2,fac,Ires(-2:0),qllnrat,xlog,
     . cln,cyy,cyi
 
      m2sq=Y(2,2)
      m4sq=Y(4,4)
      m2sqbar=2d0*Y(2,3)
      m4sqbar=2d0*Y(3,4)
C-----Assign s and t so as to agree with notation of BD
      si=2d0*Y(2,4)-Y(2,2)-Y(4,4)
      ta=2d0*Y(1,3)
      m2=sqrt(m2sq)
      m4=sqrt(m4sq)

C     iepsi gives the sign of the imaginary part of K   
      call qlkfn(cxs,ieps,-si,m2,m4)     
      xs=cxs(1)

C     Deal with non-singular special cases first 
      if (qlzero(m2sqbar) .and. qlnonzero(m4sqbar)) then
         call qlratreal(m4*m2sqbar,m2*m4sqbar,yi,iepyi)
         cyi = dcmplx(yi) 
         fac=xs/(cone-xs**2)/dcmplx((-m2*m4*ta))
         xlog=cln(xs,ieps)
         Ires(-2)=czip
         Ires(-1)=-xlog
         Ires( 0)=xlog*(-xlog-dcmplx(log(musq/m4sq))
     .   -2d0*qllnrat(m4sqbar,ta))
     .   -qlcLi2omx2(xs,xs,ieps,ieps)
     .   +qlcLi2omx2(xs,cyi,ieps,iepyi)
     .   -qlcLi2omx2(1d0/xs,cyi,-ieps,iepyi)
          goto 20
      elseif (qlzero(m4sqbar) .and. qlnonzero(m2sqbar)) then
         call qlratreal(m2*m4sqbar,m4*m2sqbar,yy,iepyy)      
         cyy = dcmplx(yy) 
         fac=xs/(cone-xs**2)/dcmplx((-m2*m4*ta))
         xlog=cln(xs,ieps)
         Ires(-2)=czip
         Ires(-1)=-xlog
         Ires( 0)=xlog*(-xlog-dcmplx(log(musq/m2sq))
     .   -2d0*qllnrat(m2sqbar,ta))
     .   -qlcLi2omx2(xs,xs,ieps,ieps)
     .   +qlcLi2omx2(xs,cyy,ieps,iepyy)
     .   -qlcLi2omx2(1d0/xs,cyy,-ieps,iepyy)
          goto 20
      elseif (qlzero(m4sqbar) .and. qlzero(m2sqbar)) then
          write(6,*) 'qlbox15:you got here in error'
          write(6,*) 'This is really qlbox14'
          write(6,*) 'qlbox15:m2sqbar,m4sqbar',m2sqbar,m4sqbar
          stop
      endif

      call qlratreal(m2*m4sqbar,m4*m2sqbar,yy,iepyy)      
      rexs=dreal(xs) 
      imxs=dimag(xs)
C----deal with s=(m2-m4)^2
      if ((qlzero(rexs-1d0)) .and. (qlzero(imxs))) then
          fac=dcmplx(-0.5d0/(m2*m4*ta))
          Ires(-2)=czip
          Ires(-1)=cone
          Ires(0)=dcmplx(log(musq/(m2*m4)))
     .    -qllnrat(m2sqbar,ta)-qllnrat(m4sqbar,ta)-dcmplx(2d0)
     .    -dcmplx((one+yy)/(one-yy))*qllnrat(m2*m4sqbar,m4*m2sqbar)
      else
C----deal with s .ne. (m2-m4)^2
          fac=xs/(cone-xs**2)/dcmplx((-m2*m4*ta))
          xlog=cln(xs,ieps)
          Ires(-2)=czip
          Ires(-1)=-xlog
          Ires( 0)=xlog*(-0.5d0*xlog-dcmplx(log(musq/(m2*m4)))
     .    -qllnrat(m2sqbar,ta)-qllnrat(m4sqbar,ta))
     .    -qlcLi2omx2(xs,xs,ieps,ieps)
     .    +chalf*qllnrat(m2*m4sqbar,m4*m2sqbar)**2
     .    +qlcLi2omx2(xs,dcmplx(yy),ieps,iepyy)
     .    +qlcLi2omx2(xs,dcmplx(1d0/yy),ieps,-iepyy)
      endif

 20   continue
      do iep=-2,0
      Ires(iep)=Ires(iep)*dcmplx(fac)
      enddo
      return
      end




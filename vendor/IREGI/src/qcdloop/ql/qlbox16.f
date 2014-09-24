      subroutine qlbox16(Y,musq,Ires)
      implicit none

C I_4^(4-2\e)(m_2^2,\pd^2,\pt^2,m_4^2;s_{12},s_{23};0,m_2^2,m_3^2,m_4^2)

c  [                              m3sq-s12                    ]
c  [0       ,      0         ,    --------    ,        0      ]
c  [                                 2                        ]
c  [                                                          ]
c  [                          -p2sq+m3sq+m2sq   -s23+m4sq+m2sq]
c  [0       ,        m2sq    ,----------------, --------------]
c  [                                 2               2        ]
c  [                                                          ]
c  [m3sq-s12  -p2sq+m3sq+m2sq                  -p3sq+m4sq+m3sq]
c  [--------, ---------------,     m3sq       ,---------------]
c  [   2                2                              2      ]
c  [                                                          ]
c  [          -s23+m4sq+m2sq   -p3sq+m4sq+m3sq                ]
c  [0       , -------------- , ---------------,      m4sq     ]
c  [                  2                    2                  ]
c  [                                                          ]

C----Implementation of Bennakker-Denner Eq.2.9
C----\bibitem{Beenakker:1988jr}
C----W.~Beenakker and A.~Denner,
C----%``INFRARED DIVERGENT SCALAR BOX INTEGRALS WITH APPLICATIONS IN THE
C----%ELECTROWEAK STANDARD MODEL,''
C----Nucl.\ Phys.\  B {\bf 338}, 349 (1990).

      include 'qlconstants.f'
      integer iep
      logical qlzero
      double precision Y(4,4),m2sq,m3sq,m4sq,musq,m2,m3,m4,
     . si,tabar,ieps,iep2,iep3,mp2sq,mp3sq,mean,rexs,imxs
      double complex fac,cxs(3),cx2(3),cx3(3),
     . xs,Ires(-2:0),qllnrat,xlog,cln,qlcLi2omx2,qlcLi2omx3
 
      m2sq=Y(2,2)
      m3sq=Y(3,3)
      m4sq=Y(4,4)
C-----Assign s and t so as to agree with notation of BD
      tabar=2d0*Y(1,3)
      si=2d0*Y(2,4)-Y(2,2)-Y(4,4)
      mp2sq=2d0*Y(2,3)-m3sq-m2sq
      mp3sq=2d0*Y(3,4)-m3sq-m4sq
      m2=sqrt(m2sq)
      m3=sqrt(m3sq)
      m4=sqrt(m4sq)
      mean=sqrt(m3sq*musq)

C     iepsi gives the sign of the imaginary part of K   
      call qlkfn(cxs,ieps,-si,m2,m4)     
      call qlkfn(cx2,iep2,-mp2sq,m2,m3)     
      call qlkfn(cx3,iep3,-mp3sq,m3,m4)     
      
      xs=cxs(1)
      rexs=dreal(xs) 
      imxs=dimag(xs)
      if ((qlzero(rexs-1d0)) .and. (qlzero(imxs))) then
      fac=dcmplx(-half/(m2*m4*tabar))
      Ires(-2)=czip
      Ires(-1)=cone

      Ires(0)=2d0*qllnrat(mean,tabar)-dcmplx(2d0)
C     special case x2=x3=1
      if ( qlzero(dreal(cx2(1)-cx3(1))) .and.
     .     qlzero(dimag(cx2(1)-cx3(1))) .and.
     .     qlzero(dreal(cx2(1))-1d0) .and.
     .     qlzero(dimag(cx2(1)))) then
      Ires(0) = Ires(0) + dcmplx(4d0) 
C     special case x2=x3 /= 1 
      elseif (qlzero(dreal(cx2(1)-cx3(1))) .and.
     .     qlzero(dimag(cx2(1)-cx3(1)))) then
      Ires(0) = Ires(0) + ctwo
     .+2d0*(cx2(1)**2+cone)*cln(cx2(1),iep2)/(cx2(1)**2-cone) 
      else
      Ires(0)=Ires(0)
     . -(cone+cx2(1)*cx3(1))/(cone-cx2(1)*cx3(1))
     . *(cln(cx2(1),iep2)+cln(cx3(1),iep3))
     . -(cone+cx2(1)/cx3(1))/(cone-cx2(1)/cx3(1))
     . *(cln(cx2(1),iep2)-cln(cx3(1),iep3))
      endif
         
      else
      fac=dcmplx(-1d0/(m2*m4*tabar))*cxs(1)/(cone-cxs(1)**2)
      xlog=cln(xs,ieps)
      Ires(-2)=czip
      Ires(-1)=-xlog
      Ires(0)=-2d0*xlog*qllnrat(mean,tabar)
     . +cln(cx2(1),iep2)**2 +cln(cx3(1),iep3)**2 
     . -qlcLi2omx2(xs,xs,ieps,ieps)
     . +qlcLi2omx3(cxs(1),cx2(1),cx3(1),ieps,iep2,iep3)
     . +qlcLi2omx3(cxs(1),cone/cx2(1),cone/cx3(1),ieps,-iep2,-iep3)
     . +qlcLi2omx3(cxs(1),cx2(1),cone/cx3(1),ieps,iep2,-iep3)
     . +qlcLi2omx3(cxs(1),cone/cx2(1),cx3(1),ieps,-iep2,iep3)
      endif

      do iep=-2,0
      Ires(iep)=Ires(iep)*fac
      enddo
      return
      end




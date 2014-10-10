      subroutine qlbox14(Y,musq,Ires)
      implicit none
C 14:
C I_4^{\{D=4-2 \epsilon\}}(m_2^2,m_2^2,m_4^2,m_4^2;s_{12},s_{23};0,m_2^2,0,m_4^2)

c
c y14 =
c
c       [                               s12                      ]
c       [   0             0           - ---           0          ]
c       [                                2                       ]
c       [                                                        ]
c       [                                    - s23 + m4sq + m2sq ]
c       [   0           m2sq            0    ------------------- ]
c       [                                             2          ]
c       [                                                        ]
c       [   s12                                                  ]
c       [ - ---           0             0             0          ]
c       [    2                                                   ]
c       [                                                        ]
c       [        - s23 + m4sq + m2sq                             ]
c       [   0    -------------------    0           m4sq         ]
c       [                 2                                      ]


      include 'qlconstants.f'
      logical qlzero
      double precision Y(4,4),m2sq,m4sq,musq,m2,m4,ta,
     . si,xs,imxs,ieps
      double complex cxs(3),Ires(-2:0),qllnrat,xlog,wlogtmu,cln,fac  
 
      m2sq=Y(2,2)
      m4sq=Y(4,4)
C-----Assign s and t (si=-s23,ta=-s12) so as to agree with notation of BD
      ta=2d0*Y(1,3)
      si=2d0*Y(2,4)-Y(2,2)-Y(4,4)
      m2=sqrt(m2sq)
      m4=sqrt(m4sq)
      wlogtmu=qllnrat(musq,ta)

C     ieps gives the sign of the imaginary part of cxs(1)   
      call qlkfn(cxs,ieps,-si,m2,m4)     
      xs=dreal(cxs(1)) 
      imxs=dimag(cxs(1))

      if ((qlzero(xs-1d0)) .and. (qlzero(imxs))) then
      fac=dcmplx(-xs/(m2*m4*ta))
      else
      xlog=cln(cxs(1),ieps)
      fac=dcmplx(2d0/(m2*m4*ta))*cxs(1)/(cxs(2)*cxs(3))*xlog
      endif

      Ires(-2)=czip
      Ires(-1)=fac
      Ires( 0)=fac*wlogtmu
      return
      end




      subroutine qlbox2(Y,musq,Ires)
C $I_4^{D=4-2 \epsilon}(0,0,0,\pq^2;s_{12},s_{23};0,0,0,0)$}
C     One mass integral as given in egz2, Eq.(A22).
c                          [                  s12    p4sq ]
c                          [   0       0    - ---  - ---- ]
c                          [                   2      2   ]
c                          [                              ]
c                          [                         s23  ]
c                          [   0       0      0    - ---  ]
c                          [                          2   ]
c                     Y2 = [                              ]
c                          [   s12                        ]
c                          [ - ---     0      0      0    ]
c                          [    2                         ]
c                          [                              ]
c                          [   p4sq    s23                ]
c                          [ - ----  - ---    0      0    ]
c                          [    2       2                 ]
      implicit none
      double precision musq,si,ta,mp4sq,Y(4,4)
      double complex Ires(-2:0),qlLsm1,qllnrat,ctwo,fac
      parameter(ctwo=(2d0,0d0))

      si=2d0*Y(1,3)
      ta=2d0*Y(2,4)
      mp4sq=2d0*Y(1,4)
      fac=dcmplx(1d0/(si*ta))      
      Ires(-2)=fac*ctwo
      Ires(-1)=fac*ctwo
     . *(qllnrat(mp4sq,musq)
     .  -qllnrat(ta,musq)
     .  -qllnrat(si,musq))
      Ires( 0)=fac*(-qllnrat(mp4sq,musq)**2
     . +qllnrat(ta,musq)**2
     . +qllnrat(si,musq)**2
     . +ctwo*(qlLsm1(ta,mp4sq,si,mp4sq)
     . -qllnrat(ta,mp4sq)*qllnrat(si,mp4sq))
     . +qllnrat(mp4sq,ta)**2+qllnrat(mp4sq,si)**2
     . -qllnrat(ta,si)**2)

      return
      end

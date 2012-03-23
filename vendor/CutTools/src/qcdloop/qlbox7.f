      subroutine qlbox7(Y,musq,Ires)
      implicit none
C   $I_4^{\{D=4-2 \e\}}(0,0,m^2,\pq^2;s_{12},s_{23};0,0,0,m^2)$}
c                    [                          s12  msq - p4sq ]
c                    [     0           0      - ---  ---------- ]
c                    [                           2       2      ]
c                    [                                          ]
c                    [                               msq - s23  ]
c                    [     0           0        0    ---------  ]
c                    [                                   2      ]
c               y7 = [                                          ]
c                    [     s12                                  ]
c                    [   - ---         0        0        0      ]
c                    [      2                                   ]
c                    [                                          ]
c                    [ msq - p4sq  msq - s23                    ]
c                    [ ----------  ---------    0       msq     ]
c                    [     2           2                        ]
      include 'qlconstants.f'
      integer iep
      double precision si,tabar,msq,p4sqbar,musq,Y(4,4)
      double complex Ires(-2:0),qllnrat,wlogt,wlogs,wlogm,wlogp,
     . qlLi2omrat

      tabar=2d0*Y(2,4)
      p4sqbar=2d0*Y(1,4)
      si=2d0*Y(1,3)
      msq=Y(4,4)
      wlogs=qllnrat(si,msq)
      wlogt=qllnrat(tabar,msq)
      wlogp=qllnrat(p4sqbar,msq)
      wlogm=qllnrat(musq,msq)
      Ires(-2)=dcmplx(1.5d0)
      Ires(-1)=1.5d0*wlogm-2d0*wlogt-wlogs+wlogp
      Ires( 0)=2d0*wlogs*wlogt-wlogp**2-dcmplx(5d0*pisq/12d0)
     . +0.75d0*wlogm**2+wlogm*(-2d0*wlogt-wlogs+wlogp)
     . -2d0*qlLi2omrat(p4sqbar,tabar)
      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(si*tabar)
      enddo
      return
      end


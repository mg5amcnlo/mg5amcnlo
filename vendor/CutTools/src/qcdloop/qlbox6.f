      subroutine qlbox6(Y,musq,Ires)
      implicit none
c    $I_4^{\{D=4-2 \e\}}(0,0,m^2,m^2;s_{12},s_{23};0,0,0,m^2)$}
c                       [                     s12            ]
c                       [   0        0      - ---      0     ]
c                       [                      2             ]
c                       [                                    ]
c                       [                          msq - s23 ]
c                       [   0        0        0    --------- ]
c                       [                              2     ]
c                  y6 = [                                    ]
c                       [   s12                              ]
c                       [ - ---      0        0        0     ]
c                       [    2                               ]
c                       [                                    ]
c                       [        msq - s23                   ]
c                       [   0    ---------    0       msq    ]
c                       [            2                       ]

      include 'qlconstants.f'
      integer iep
      double precision si,tabar,msq,musq,Y(4,4)
      double complex Ires(-2:0),qllnrat,wlogt,wlogs,wlogm
      si=2d0*Y(1,3)
      tabar=2d0*Y(2,4)
      msq=Y(4,4)
      wlogs=qllnrat(si,msq)
      wlogt=qllnrat(tabar,msq)
      wlogm=qllnrat(musq,msq)
      Ires(-2)=dcmplx(2d0)
      Ires(-1)=2d0*(wlogm-wlogt)-wlogs
      Ires( 0)=wlogm**2-wlogm*(2d0*wlogt+wlogs)
     . +2d0*wlogt*wlogs-dcmplx(0.5d0*pisq)
      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(si*tabar)
      enddo
      return
      end


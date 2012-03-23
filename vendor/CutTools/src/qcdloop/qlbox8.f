      subroutine qlbox8(Y,musq,Ires)
      implicit none
C   $I_4^{\{D=4-2 \e\}}(0,0,\pt^2,\pq^2; s_{12},s_{23};0,0,0,m^2)$}

c                 [                            s12     msq - p4sq ]
c                 [     0           0        - ---     ---------- ]
c                 [                             2          2      ]
c                 [                                               ]
c                 [                                    msq - s23  ]
c                 [     0           0          0       ---------  ]
c                 [                                        2      ]
c            y8 = [                                               ]
c                 [     s12                            msq - p3sq ]
c                 [   - ---         0          0       ---------- ]
c                 [      2                                 2      ]
c                 [                                               ]
c                 [ msq - p4sq  msq - s23  msq - p3sq             ]
c                 [ ----------  ---------  ----------     msq     ]
c                 [     2           2          2                  ]
      include 'qlconstants.f'
      integer iep
      double precision msq,si,p3sqbar,p4sqbar,tabar,musq,Y(4,4),
     .     r1,r2,ieps1,ieps2
      double complex Ires(-2:0),qllnrat,wlogs,qlLi2omrat,qlLi2omx2,
     . wlogp3,wlogp4,dilog34,dilog4,dilog3

      msq=Y(4,4)
      tabar=2d0*Y(2,4)
      si=2d0*Y(1,3)
      p3sqbar=2d0*Y(3,4)
      p4sqbar=2d0*Y(1,4)
      wlogs=qllnrat(si,musq)
      wlogp3=qllnrat(p3sqbar,tabar)
      wlogp4=qllnrat(p4sqbar,tabar)

      dilog3=qlLi2omrat(p3sqbar,tabar)
      dilog4=qlLi2omrat(p4sqbar,tabar)
      dilog34=qlLi2omx2(p3sqbar,p4sqbar,si,msq)

      call qlratreal(p3sqbar,si,r1,ieps1)
      call qlratreal(p4sqbar,msq,r2,ieps2)
       
      Ires(-2)=cone
      Ires(-1)=wlogp3+wlogp4-wlogs

      Ires( 0)=-2d0*dilog3-2d0*dilog4-dilog34
     . -dcmplx(pisq/6d0)+0.5d0*(qllnrat(si,musq)**2-qllnrat(si,msq)**2)
     . +2d0*qllnrat(si,musq)*qllnrat(tabar,msq)
     . -qllnrat(p3sqbar,musq)*qllnrat(p3sqbar,msq)
     . -qllnrat(p4sqbar,musq)*qllnrat(p4sqbar,msq)
      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(si*tabar)
      enddo
      return
      end


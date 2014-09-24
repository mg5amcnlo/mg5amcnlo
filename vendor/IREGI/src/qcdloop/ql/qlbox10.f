      subroutine qlbox10(Y,musq,Ires)
      implicit none
C  I^{\{D=4-2 \epsilon\}}_4(0,p_2^2,p_3^2,p_4^2;s_{12},s_{23};0,0,0,m^2)

c                  [                            s12     msq - p4sq ]
c                  [     0           0        - ---     ---------- ]
c                  [                             2          2      ]
c                  [                                               ]
c                  [                            p2sq    msq - s23  ]
c                  [     0           0        - ----    ---------  ]
c                  [                             2          2      ]
c            y10 = [                                               ]
c                  [     s12        p2sq                msq - p3sq ]
c                  [   - ---      - ----        0       ---------- ]
c                  [      2          2                      2      ]
c                  [                                               ]
c                  [ msq - p4sq  msq - s23  msq - p3sq             ]
c                  [ ----------  ---------  ----------     msq     ]
c                  [     2           2          2                  ]

      include 'qlconstants.f'
      integer iep
      double precision msq,musq,
     . si,tabar,mean,fac,m3sqbar,m4sqbar,mp2sq,Y(4,4)
      double complex qlLi2omrat,qlLi2omx2,dilog(5),Ires(-2:0),qllnrat,
     . wlogtmu,wlogsmu,wlog2mu,wlog4mu

      msq=Y(4,4)
      si=2d0*Y(1,3)
      tabar=2d0*Y(2,4)
      m4sqbar=2d0*Y(1,4)
      m3sqbar=2d0*Y(3,4)
      mp2sq=2d0*Y(2,3)
      mean=sqrt(musq*msq)

      fac=si*tabar-mp2sq*m4sqbar
      wlogsmu=qllnrat(si,musq)
      wlogtmu=qllnrat(tabar,musq)
      wlog2mu=qllnrat(mp2sq,musq)
      wlog4mu=qllnrat(m4sqbar,musq)

      dilog(1)=qlLi2omrat(mp2sq,si)
      dilog(2)=qlLi2omrat(tabar,m4sqbar)
      dilog(3)=qlLi2omx2(mp2sq,m4sqbar,si,tabar)
      dilog(4)=qlLi2omx2(m3sqbar,tabar,mp2sq,msq)
      dilog(5)=qlLi2omx2(m3sqbar,m4sqbar,si,msq)

      Ires(-2)=czip
      Ires(-1)=wlog2mu+wlog4mu-wlogsmu-wlogtmu
      Ires( 0)=dilog(4)-dilog(5)
     . -2d0*dilog(1)+2d0*dilog(2)+2d0*dilog(3)
     . +2d0*Ires(-1)*qllnrat(mean,tabar)
      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(fac)
      enddo
      return
      end



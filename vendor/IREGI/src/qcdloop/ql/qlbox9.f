      subroutine qlbox9(Y,musq,Ires)
      implicit none
c  $I^{\{D=4-2 \epsilon\}}_4(0,p_2^2,p_3^2,m^2;s_{12},s_{23};0,0,0,m^2)$}

c                    [                       s12                ]
c                    [   0        0        - ---         0      ]
c                    [                        2                 ]
c                    [                                          ]
c                    [                       p2sq    msq - s23  ]
c                    [   0        0        - ----    ---------  ]
c                    [                        2          2      ]
c               y9 = [                                          ]
c                    [   s12     p2sq                msq - p3sq ]
c                    [ - ---   - ----        0       ---------- ]
c                    [    2       2                      2      ]
c                    [                                          ]
c                    [        msq - s23  msq - p3sq             ]
c                    [   0    ---------  ----------     msq     ]
c                    [            2          2                  ]
      include 'qlconstants.f'
      integer iep
      double complex Ires(-2:0),qllnrat,wlogt,wlog2,
     . dilog1,dilog2,qlLi2omrat,qlLi2omx2
      double precision Y(4,4),msq,musq,tabar,si,fac,mean,m3sqbar,mp2sq

      msq=Y(4,4)
      mean=sqrt(musq*msq)
      tabar=2d0*Y(2,4)
      si=2d0*Y(1,3)
      m3sqbar=2d0*Y(3,4)
      mp2sq=2d0*Y(2,3)
      fac=si*tabar

      wlogt=qllnrat(tabar,mean)
      wlog2=qllnrat(si,mp2sq)

      dilog1=qlLi2omx2(m3sqbar,tabar,mp2sq,msq)
      dilog2=qlLi2omrat(si,mp2sq)
      Ires(-2)=dcmplx(0.5d0)
      Ires(-1)=-wlogt-wlog2
      Ires( 0)=dilog1+2d0*dilog2+dcmplx(pisq/12d0)+(wlogt+wlog2)**2
      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(fac)
      enddo
      return
      end


      subroutine qlbox11(Y,musq,Ires)
      implicit none
C  I^{\{D=4-2 \epsilon\}}_4(0,m_3^2,\pt^2,m_4^2;s_{12},s_{23};0,0,m_3^2,m_4^2)

c y11=
c [                              m3sq - s12                            ]
c [     0           0            ----------                0           ]
c [                                  2                                 ]
c [                                                                    ]
c [                                                    m4sq - s23      ]
c [     0           0                0                 ----------      ]
c [                                                        2           ]
c [                                                                    ]
c [ m3sq - s12                                    - p3sq + m4sq + m3sq ]
c [ ----------      0               m3sq          -------------------- ]
c [     2                                                  2           ]
c [                                                                    ]
c [             m4sq - s23  - p3sq + m4sq + m3sq                       ]
c [     0       ----------  --------------------          m4sq         ]
c [                 2                2                                 ]
      include 'qlconstants.f'
      integer iep
      logical qlzero
      double precision Y(4,4),p3sq,m3sq,m4sq,musq,
     . m3mu,m4mu,sibar,tabar,x43p,x43m,x43pm1,x43mm1,ieps2
      double complex wlogt,wlogs,qllnrat,Ires(-2:0),Intbit,
     . root,cln,ln43m,ln43p,ga43p,ga43pm1,ga43m,ga43mm1,rat2p,rat2m

      m3sq=Y(3,3)
      m4sq=Y(4,4)
      sibar=2d0*Y(1,3)
      tabar=2d0*Y(2,4)
      p3sq=-(2d0*Y(3,4)-Y(3,3)-Y(4,4))

      m3mu=sqrt(m3sq*musq)
      m4mu=sqrt(m4sq*musq)
      wlogt=qllnrat(tabar,m4mu)
      wlogs=qllnrat(sibar,m3mu)

C----evaluate gamma's for the case p3sq=0
      if (qlzero(p3sq)) then
      root=cone
      x43p=-one
      x43pm1=-one
      x43m=m3sq
      x43mm1=m4sq
      else
      root=dcmplx((p3sq+m3sq-m4sq)**2-4d0*m3sq*p3sq)
      root=sqrt(root)
      ga43p=  dcmplx(+p3sq+m3sq-m4sq)+root
      ga43pm1=dcmplx(-p3sq+m3sq-m4sq)+root
      ga43m=  dcmplx(+p3sq+m3sq-m4sq)-root
      ga43mm1=dcmplx(-p3sq+m3sq-m4sq)-root

      x43p=  -dreal(ga43p)
      x43pm1=-dreal(ga43pm1)
      x43m=   dreal(ga43m)
      x43mm1= dreal(ga43mm1)
      endif


C----deal with real roots
      if (qlzero(dimag(root))) then
      ln43p=qllnrat(x43p,x43pm1)
      ln43m=qllnrat(x43m,x43mm1)
      else
      call qlratgam(rat2p,rat2m,ieps2,p3sq,m4sq,m3sq)      
      ln43p=cln(rat2p,ieps2)
      ln43m=cln(rat2m,ieps2)

      endif

      if (qlzero(p3sq)) then
      Intbit=-chalf*log(m3sq/m4sq)**2
      else
      Intbit=-chalf*(ln43p**2+ln43m**2)
      endif

      Ires(-2)=cone
      Ires(-1)=-wlogt-wlogs
      Ires( 0)=Intbit
     . +ctwo*wlogt*wlogs-dcmplx(0.5d0*pisq)
     . +dcmplx(0.25d0*log(m3sq/m4sq)**2)
      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(sibar*tabar)
      enddo
      return
      end


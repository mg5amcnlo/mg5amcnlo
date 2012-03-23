      subroutine qlbox12(Y,musq,Ires)
      implicit none
C I^{\{D=4-2 \epsilon\}}_4(0,m_3^2,\pt^2,\pq^2;s_{12},s_{23};0,0,m_3^2,m_4^2) 

c[                               m3sq - s12           m4sq - p4sq      ]
c[      0           0            ----------           -----------      ]
c[                                   2                     2           ]
c[                                                                     ]
c[                                                     m4sq - s23      ]
c[      0           0                0                 ----------      ]
c[                                                         2           ]
c[                                                                     ]
c[ m3sq - s12                                     - p3sq + m4sq + m3sq ]
c[ ----------       0               m3sq          -------------------- ]
c[     2                                                   2           ]
c[                                                                     ]
c[ m4sq - p4sq  m4sq - s23  - p3sq + m4sq + m3sq                       ]
c[ -----------  ----------  --------------------          m4sq         ]
c[      2           2                2                                 ]
      include 'qlconstants.f'
      integer iep
      logical qlzero
      double precision Y(4,4),p3sq,m3sq,m4sq,musq,
     . sibar,tabar,m4sqbar,fac,mean,
     . x43p,x43pm1,x43m,x43mm1,rat1,ieps1,ieps2
      double complex Ires(-2:0),qllnrat,dilog(3),zrat1,
     . ln43m,ln43p,cln,wlogtmu,wlogsmu,wlog4mu,wlog,qlLi2omrat,
     . qlLi2omx2,rat2p,rat2m,root,ga43p,ga43pm1,ga43m,ga43mm1

      m3sq=Y(3,3)
      m4sq=Y(4,4)
      sibar=2d0*Y(1,3)
      tabar=2d0*Y(2,4)
      m4sqbar=2d0*Y(1,4)
      p3sq=-(2d0*Y(3,4)-Y(3,3)-Y(4,4))

      mean=sqrt(musq*m3sq)
      fac=sibar*tabar
      wlogsmu=qllnrat(sibar,mean)
      wlogtmu=qllnrat(tabar,mean)
      wlog4mu=qllnrat(m4sqbar,mean)
      wlog=wlogsmu+wlogtmu-wlog4mu


C---- evaluate gamma's for the case p3sq=0
      if (qlzero(p3sq)) then
      root=cone
      x43p=-1d0
      x43pm1=-1d0
      x43m=m3sq
      x43mm1=m4sq
      else
      root=dcmplx((p3sq+m3sq-m4sq)**2-4d0*m3sq*p3sq)
      root=sqrt(root)
      ga43p=  dcmplx(+p3sq+m3sq-m4sq)+root
      ga43pm1=dcmplx(-p3sq+m3sq-m4sq)+root
      ga43m=  dcmplx(+p3sq+m3sq-m4sq)-root
      ga43mm1=dcmplx(-p3sq+m3sq-m4sq)-root

      x43p=-dreal(ga43p)
      x43pm1=-dreal(ga43pm1)
      x43m=dreal(ga43m)
      x43mm1=dreal(ga43mm1)
      endif


      dilog(1)=qlLi2omrat(m4sqbar,tabar)

C----deal with real roots
      if (qlzero(dimag(root))) then
      ln43p=qllnrat(x43p,x43pm1)
      ln43m=qllnrat(x43m,x43mm1)
      dilog(2)=qlLi2omx2(m4sqbar,x43p,sibar,x43pm1)
      dilog(3)=qlLi2omx2(m4sqbar,x43m,sibar,x43mm1)
      else
      call qlratreal(m4sqbar,sibar,rat1,ieps1)      
      call qlratgam(rat2p,rat2m,ieps2,p3sq,m4sq,m3sq)      
      zrat1=dcmplx(rat1)
      ln43p=cln(rat2p,ieps2)
      ln43m=cln(rat2m,ieps2)

      call qlspencer(zrat1,rat2p,ieps1,ieps2,dilog(2))
      call qlspencer(zrat1,rat2m,ieps1,ieps2,dilog(3))
      
      endif
      Ires(-2)=dcmplx(0.5d0)
      Ires(-1)=-wlog
      Ires( 0)=-dcmplx(pisq/12d0)
     . +2d0*wlogsmu*wlogtmu-wlog4mu**2
     . +(wlog4mu-wlogsmu)*log(m4sq/m3sq)-0.5d0*(ln43p**2+ln43m**2)
     . -2d0*dilog(1)-dilog(2)-dilog(3)

       
      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(fac)
      enddo
      return
      end


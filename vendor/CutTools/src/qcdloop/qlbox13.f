      subroutine qlbox13(Y,musq,Ires)
      implicit none
c I^{\{D=4-2 \epsilon\}}_4(0,\pd^2,\pt^2,\pq^2;s_{12},s_{23};0,0,m_3^2,m_4^2)
c y13 =
c                                m3sq - s12           m4sq - p4sq      )
c      0            0            ----------           -----------      )
c                                    2                     2           )
c                                                                      )
c                               m3sq - p2sq            m4sq - s23      )
c      0            0           -----------            ----------      )
c                                    2                     2           )
c                                                                      )
c m3sq - s12   m3sq - p2sq                        - p3sq + m4sq + m3sq )
c ----------   -----------          m3sq          -------------------- )
c     2             2                                      2           )
c                                                                      )
c m4sq - p4sq  m4sq - s23   - p3sq + m4sq + m3sq                       )
c -----------  ----------   --------------------          m4sq         )
c      2           2                 2                                 )

      include 'qlconstants.f'
      integer iep
      logical qlzero
      double precision Y(4,4),p3sq,m3sq,m4sq,musq,
     . sibar,tabar,m3sqbar,m4sqbar,fac,
     . x34p,x34m,x34pm1,x34mm1,rat3t,rat4s,ieps3t,ieps4s,
     . x43p,x43m,x43pm1,x43mm1,ieps34,ieps43
      double complex Ires(-2:0),qllnrat,
     . wlogtmu,wlogsmu,wlog3mu,wlog4mu,dilog(7),
     . rat34p,rat34m,rat43p,rat43m,root,qlLi2omrat,qlLi2omx2,
     . ga34p,ga34m,ga34pm1,ga34mm1,
     . ga43p,ga43m,ga43pm1,ga43mm1,cln,ln43p,ln43m,zrat3t,zrat4s

      m3sq=Y(3,3)
      m4sq=Y(4,4)
      sibar=2d0*Y(1,3)
      tabar=2d0*Y(2,4)
      m4sqbar=2d0*Y(1,4)
      m3sqbar=2d0*Y(2,3)
      p3sq=-(2d0*Y(3,4)-Y(3,3)-Y(4,4))

      fac=sibar*tabar-m3sqbar*m4sqbar
      wlogsmu=qllnrat(sibar,musq)
      wlogtmu=qllnrat(tabar,musq)
      wlog3mu=qllnrat(m3sqbar,musq)
      wlog4mu=qllnrat(m4sqbar,musq)
      dilog(1)=qlLi2omrat(m3sqbar,sibar)
      dilog(4)=qlLi2omrat(m4sqbar,tabar)
      dilog(7)=qlLi2omx2(m3sqbar,m4sqbar,sibar,tabar)


C---  setup gammas for qlzero p3sq
      if (qlzero(p3sq)) then
      root=cone
      x34p=-1d0
      x34pm1=-1d0
      x34m=m4sq
      x34mm1=m3sq

      x43p=m3sq
      x43pm1=m4sq
      x43m=-1d0
      x43mm1=-1d0

      else
      root=dcmplx((p3sq-m3sq+m4sq)**2-4d0*m4sq*p3sq)
      root=sqrt(root)

      ga34p=  dcmplx(+p3sq+m4sq-m3sq)+root
      ga34pm1=dcmplx(-p3sq+m4sq-m3sq)+root
      ga34m=  dcmplx(+p3sq+m4sq-m3sq)-root
      ga34mm1=dcmplx(-p3sq+m4sq-m3sq)-root

      ga43p=  dcmplx(+p3sq+m3sq-m4sq)+root
      ga43pm1=dcmplx(-p3sq+m3sq-m4sq)+root
      ga43m=  dcmplx(+p3sq+m3sq-m4sq)-root
      ga43mm1=dcmplx(-p3sq+m3sq-m4sq)-root

      x34p=-dreal(ga34p)
      x34pm1=-dreal(ga34pm1)
      x34m=dreal(ga34m)
      x34mm1=dreal(ga34mm1)

      x43p=-dreal(ga43p)
      x43pm1=-dreal(ga43pm1)
      x43m=dreal(ga43m)
      x43mm1=dreal(ga43mm1)

      endif

      if (qlzero(dimag(root))) then
      ln43p=qllnrat(x43p,x43pm1)
      ln43m=qllnrat(x43m,x43mm1)
      
      dilog(2)=qlLi2omx2(m3sqbar,x34p,tabar,x34pm1)
      dilog(3)=qlLi2omx2(m3sqbar,x34m,tabar,x34mm1)
      dilog(5)=qlLi2omx2(m4sqbar,x43p,sibar,x43pm1)
      dilog(6)=qlLi2omx2(m4sqbar,x43m,sibar,x43mm1)

      else

      call qlratreal(m3sqbar,tabar,rat3t,ieps3t)      
      call qlratreal(m4sqbar,sibar,rat4s,ieps4s)      

      call qlratgam(rat34p,rat34m,ieps34,p3sq,m3sq,m4sq)      
      call qlratgam(rat43p,rat43m,ieps43,p3sq,m4sq,m3sq)      

      zrat3t=dcmplx(rat3t)
      zrat4s=dcmplx(rat4s)

      call qlspencer(zrat3t,rat34p,ieps3t,ieps34,dilog(2))
      call qlspencer(zrat3t,rat34m,ieps3t,ieps34,dilog(3))
      call qlspencer(zrat4s,rat43p,ieps4s,ieps43,dilog(5))
      call qlspencer(zrat4s,rat43m,ieps4s,ieps43,dilog(6))

      ln43p=cln(rat43p,0d0)
      ln43m=cln(rat43m,0d0)

      endif

      Ires(-2)=czip
      Ires(-1)=wlog3mu+wlog4mu-wlogsmu-wlogtmu
      Ires( 0)=
     . -2d0*dilog(1)-dilog(2)-dilog(3)
     . -2d0*dilog(4)-dilog(5)-dilog(6)
     . +2d0*dilog(7)
     . +2d0*wlogsmu*wlogtmu-wlog3mu**2-wlog4mu**2
     . +(wlog3mu-wlogtmu)*log(m3sq/musq)
     . +(wlog4mu-wlogsmu)*log(m4sq/musq)
     . -0.5d0*(ln43p**2+ln43m**2)      

      do iep=-2,0
      Ires(iep)=Ires(iep)/dcmplx(fac)
      enddo

      return
      end




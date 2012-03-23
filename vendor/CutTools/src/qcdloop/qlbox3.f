      subroutine qlbox3(Y,musq,Ires)
      implicit none
c     I_4^{D=4-2 \epsilon}(0,\pd^2,0,\pq^2;s_{12},s_{23};0,0,0,0)
C ----%\cite{Bern:1993kr}
c----\bibitem{Bern:1993kr}
c----  Z.~Bern, L.~J.~Dixon and D.~A.~Kosower,
c----  %``Dimensionally regulated pentagon integrals,''
c----  Nucl.\ Phys.\ B {\bf 412}, 751 (1994)
c----  [arXiv:hep-ph/9306240].
c----  %%CITATION = HEP-PH 9306240;%%
c----  Eqs. (I.13)
c                         [                   s12     p4sq ]
c                         [   0       0     - ---   - ---- ]
c                         [                    2       2   ]
c                         [                                ]
c                         [                   p2sq    s23  ]
c                         [   0       0     - ----  - ---  ]
c                         [                    2       2   ]
c                    Y3 = [                                ]
c                         [   s12     p2sq                 ]
c                         [ - ---   - ----    0       0    ]
c                         [    2       2                   ]
c                         [                                ]
c                         [   p4sq    s23                  ]
c                         [ - ----  - ---     0       0    ]
c                         [    2       2                   ]
      include 'qlconstants.f'
      double precision si,ta,mp2sq,mp4sq,musq,r,Y(4,4)
      double complex Ires(-2:0),qlLsm1_2me,qlL0,qlL1,qllnrat,fac
      logical landau
      si=2d0*Y(1,3)  
      ta=2d0*Y(2,4)  
      mp4sq=2d0*Y(1,4)  
      mp2sq=2d0*Y(2,3)  
      r=1d0-mp2sq*mp4sq/(si*ta)

C     Use expansion only in cases where signs (si,ta,mp2sq,mp4sq) are not
C     ++-- or --++      
      landau=((sign(1d0,si) .eq. sign(1d0,ta)) 
     . .and. (sign(1d0,mp2sq) .eq. sign(1d0,mp4sq))  
     . .and. (sign(1d0,si) .ne. sign(1d0,mp2sq)))  
      if ((abs(r) .lt. 1d-6) .and. (landau .eqv. .false.)) then         
C---expanded case
      fac=dcmplx(1d0/(si*ta))
      Ires(-2)=czip
      Ires(-1)=-dcmplx(2d0+r)*fac
      Ires( 0)=fac*(dcmplx(2d0-0.5d0*r)
     . +dcmplx(2d0+r)*(qllnrat(si,musq)+qllnrat(ta,mp4sq))
     . +dcmplx(2d0)*(qlL0(mp4sq,ta)+qlL0(mp4sq,si))
     . +dcmplx(r)*(qlL1(mp4sq,ta)+qlL1(mp4sq,si)))
      else
C---general case
      fac=dcmplx(1d0/(si*ta-mp2sq*mp4sq))
      Ires(-2)=czip
      Ires(-1)=fac*2d0*(qllnrat(mp2sq,si)
     .                 +qllnrat(mp4sq,ta))
      Ires( 0)=fac*(+qllnrat(si,musq)**2+qllnrat(ta,musq)**2
     .              -qllnrat(mp2sq,musq)**2-qllnrat(mp4sq,musq)**2
     .  +2d0*qlLsm1_2me(-si,-ta,-mp2sq,-mp4sq))
      endif

      return
      end



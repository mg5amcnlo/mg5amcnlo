      subroutine qlbox5(Y,musq,Ires)
      implicit none
C     $I_4^{D=4-2 \epsilon}(0,\pd^2,\pt^2,\pq^2;s_{12},s_{23};0,0,0,0)$}

c----%\cite{Bern:1993kr}
c----\bibitem{Bern:1993kr}
c----  Z.~Bern, L.~J.~Dixon and D.~A.~Kosower,
c----  %``Dimensionally regulated pentagon integrals,''
c----  Nucl.\ Phys.\ B {\bf 412}, 751 (1994)
c----  [arXiv:hep-ph/9306240].
c----  %%CITATION = HEP-PH 9306240;%%
c----  Eqs. (I.15)
C-----or from /hep-ph/0508308 v3 Eqn (A27) 
C-----v3 corrects previous versions.
c                         [                   s12     p4sq ]
c                         [   0       0     - ---   - ---- ]
c                         [                    2       2   ]
c                         [                                ]
c                         [                   p2sq    s23  ]
c                         [   0       0     - ----  - ---  ]
c                         [                    2       2   ]
c                    y5 = [                                ]
c                         [   s12     p2sq            p3sq ]
c                         [ - ---   - ----    0     - ---- ]
c                         [    2       2               2   ]
c                         [                                ]
c                         [   p4sq    s23     p3sq         ]
c                         [ - ----  - ---   - ----    0    ]
c                         [    2       2       2           ]
      include 'qlconstants.f'
      integer iep
      double precision r,musq,Y(4,4),si,ta,mp2sq,mp3sq,mp4sq
      double complex Ires(-2:0),qllnrat,qlL0,qlL1,fac,Li2(6),
     . qlLi2omrat,qlLi2omx2
      logical landau
      si=2d0*Y(1,3)  
      ta=2d0*Y(2,4)  
      mp2sq=2d0*Y(2,3)  
      mp3sq=2d0*Y(3,4)  
      mp4sq=2d0*Y(1,4)  

      r=1d0-mp2sq*mp4sq/(si*ta)

C     Use expansion only in cases where signs (si,ta,mp2sq,mp4sq) are not
C     ++-- or --++      
      landau=((sign(1d0,si) .eq. sign(1d0,ta)) 
     . .and. (sign(1d0,mp2sq) .eq. sign(1d0,mp4sq))  
     . .and. (sign(1d0,si) .ne. sign(1d0,mp2sq)))  
      if ((abs(r) .lt. 1d-6) .and. (landau .eqv. .false.)) then         
C---expanded case
      Ires(-2)=czip
      Ires(-1)=-dcmplx((1d0+0.5d0*r)/(si*ta))
      Ires(0)=Ires(-1)*(qllnrat(musq,si)+qllnrat(mp3sq,ta)-dcmplx(2d0)
     . -dcmplx(1d0+mp4sq/ta)*qlL0(mp4sq,ta))
     . +dcmplx(r/(si*ta))*(qlL1(mp4sq,ta)-qlL0(mp4sq,ta)-cone)
      else
C---General case
      fac=dcmplx(1d0/(si*ta-mp2sq*mp4sq))
      Ires(-2)=czip
      Ires(-1)=qllnrat(mp2sq,ta)+qllnrat(mp4sq,si)

      Li2(1)=qlLi2omrat(mp2sq,si)
      Li2(2)=qlLi2omrat(mp4sq,ta)
      Li2(3)=qlLi2omx2(mp2sq,mp4sq,si,ta)
      Ires(0)=
     . -chalf*(qllnrat(ta,mp2sq)**2+qllnrat(si,mp4sq)**2)
     .  +(qllnrat(mp3sq,ta)+qllnrat(musq,ta))*qllnrat(mp2sq,ta)
     .  +(qllnrat(mp3sq,si)+qllnrat(musq,si))*qllnrat(mp4sq,si)
     . -ctwo*(Li2(1)+Li2(2)-Li2(3))-qllnrat(si,ta)**2
      do iep=-1,0
      Ires(iep)=fac*Ires(iep)
      enddo

      endif 

      return
      end


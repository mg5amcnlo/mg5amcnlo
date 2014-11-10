      subroutine qlbox4(Y,musq,Ires)
      implicit none
C     $I_4^{D=4-2 \epsilon}(0,0,\pt^2,\pq^2;s_{12},s_{23};0,0,0,0)$}

c----%\cite{Bern:1993kr}
c----\bibitem{Bern:1993kr}
c----  Z.~Bern, L.~J.~Dixon and D.~A.~Kosower,
c----  %``Dimensionally regulated pentagon integrals,''
c----  Nucl.\ Phys.\ B {\bf 412}, 751 (1994)
c----  [arXiv:hep-ph/9306240].
c----  %%CITATION = HEP-PH 9306240;%%
c----  Eqs. (I.14)
c                         [                  s12     p4sq ]
c                         [   0       0    - ---   - ---- ]
c                         [                   2       2   ]
c                         [                               ]
c                         [                          s23  ]
c                         [   0       0      0     - ---  ]
c                         [                           2   ]
c                    Y4 = [                               ]
c                         [   s12                    p3sq ]
c                         [ - ---     0      0     - ---- ]
c                         [    2                      2   ]
c                         [                               ]
c                         [   p4sq    s23    p3sq         ]
c                         [ - ----  - ---  - ----    0    ]
c                         [    2       2      2           ]
      include 'qlconstants.f' 
      double precision si,ta,mp3sq,mp4sq,musq,Y(4,4)
      double complex Ires(-2:0),qlLsm1_2mht,qllnrat,fac

      si=2d0*Y(1,3)  
      ta=2d0*Y(2,4)  
      mp4sq=2d0*Y(1,4)  
      mp3sq=2d0*Y(3,4)  

      fac=dcmplx(1d0/(si*ta))
      Ires(-2)=fac
      Ires(-1)=-fac
     . *(qllnrat(si,mp3sq)
     .  +qllnrat(ta,mp4sq)
     .  +qllnrat(ta,musq))
   
      Ires( 0)=fac*(qllnrat(ta,musq)**2
     .       +chalf*qllnrat(si,musq)**2
     .       -chalf*qllnrat(mp3sq,musq)**2
     .       -chalf*qllnrat(mp4sq,musq)**2
     .  +ctwo*qlLsm1_2mht(-si,-ta,-mp3sq,-mp4sq))
      return
      end



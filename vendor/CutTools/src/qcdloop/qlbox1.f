      subroutine qlbox1(Y,musq,Ires)
c    $I_4^{D=4-2 \epsilon}(0,0,0,0;s_{12},s_{23};0,0,0,0)$}
c----%\cite{Bern:1993kr}
c----\bibitem{Bern:1993kr}
c----  Z.~Bern, L.~J.~Dixon and D.~A.~Kosower,
c----  %``Dimensionally regulated pentagon integrals,''
c----  Nucl.\ Phys.\ B {\bf 412}, 751 (1994)
c----  [arXiv:hep-ph/9306240].
c----  %%CITATION = HEP-PH 9306240;%%
c----  Eqn (I.11)
c----Cayley matrix
c                [                 s12        ]
c                [   0      0    - ---    0   ]
c                [                  2         ]
c                [                            ]
c                [                        s23 ]
c                [   0      0      0    - --- ]
c                [                         2  ]
c           Y1 = [                            ]
c                [   s12                      ]
c                [ - ---    0      0      0   ]
c                [    2                       ]
c                [                            ]
c                [          s23               ]
c                [   0    - ---    0      0   ]
c                [           2                ]
      implicit none
      include 'qlconstants.f'
      double precision musq,Y(4,4),si,ta
      double complex Ires(-2:0),qllnrat,fac

      si=two*Y(1,3)
      ta=two*Y(2,4)
      fac=dcmplx(1d0/(si*ta))     
      Ires(-2)=fac*ctwo*ctwo
      Ires(-1)=fac*ctwo*(
     . -qllnrat(ta,musq)-qllnrat(si,musq))
      Ires( 0)=fac*(qllnrat(ta,musq)**2+qllnrat(si,musq)**2
     . -qllnrat(ta,si)**2-dcmplx(pi**2))
      return
      end


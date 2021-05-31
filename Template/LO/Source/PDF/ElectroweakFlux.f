c     /* ********************************************************* *
c                 Effective Vector Boson Approximation
c     /* ********************************************************* *      
c     File: ElectroweakFlux.f
c     R. Ruiz (2021 February)
c     For details, see companion paper by ..., et al [arXiv:]
c     /* ********************************************************* *
c     /* ********************************************************* *      
c     function eva_fX_to_vV(gg2,gL2,gR2,fLpol,mv2,x,mu2,ievo):
c     call electroweak PDF for for vector boson (hel=V=0,+,-) from fermion
c     with fractional (0<pol<1) LH (RH) polarization of fLpol (1-fLpol)
c     - fLpol = 0.5d0 = parent fermion is spin-averaged
c     /* ********************************************************* *
c     /* ********************************************************* *      
c     function eva_fF_to_vV(gg2,gF2,mv2,x,mu2,ievo):
c     electroweak PDF for vector boson (hel=V=0,+,-) from fermion (hel=F=L/R)
c     - gg2 = (coupling)^2
c     - gF2 = (L/R chiral coupling)^2
c     - mv2 = (mass of boson)^2
c     - x = momentum fraction = (E_V / E_f)
c     - mu2 = (max of evolution scale)^2
c     - ievo = for evolution by virtuality (or !=0 for pT)      
c     /* ********************************************************* *
c     /* ********************************************************* *
c     /* ********************************************************* *            
c     /* ********************************************************* *
      double precision function eva_fX_to_vp(gg2,gL2,gR2,fLpol,mv2,x,mu2,ievo)
      implicit none
      integer ievo
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2
      double precision eva_fL_to_vp,eva_fR_to_vp

      eva_fX_to_vp =       fLpol*eva_fL_to_vp(gg2,gL2,mv2,x,mu2,ievo)
     &             + (1d0-fLpol)*eva_fR_to_vp(gg2,gR2,mv2,x,mu2,ievo)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_vm(gg2,gL2,gR2,fLpol,mv2,x,mu2,ievo)
      implicit none
      integer ievo
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2
      double precision eva_fL_to_vm,eva_fR_to_vm

      eva_fX_to_vm =       fLpol*eva_fL_to_vm(gg2,gL2,mv2,x,mu2,ievo)
     &             + (1d0-fLpol)*eva_fR_to_vm(gg2,gR2,mv2,x,mu2,ievo)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_v0(gg2,gL2,gR2,fLpol,mv2,x,mu2,ievo)
      implicit none
      integer ievo
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2
      double precision eva_fL_to_v0,eva_fR_to_v0

      eva_fX_to_v0 =       fLpol*eva_fL_to_v0(gg2,gL2,mv2,x,mu2,ievo)
     &             + (1d0-fLpol)*eva_fR_to_v0(gg2,gR2,mv2,x,mu2,ievo)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_fL(gg2,gL2,gR2,fLpol,mv2,x,mu2,ievo)
      implicit none
      integer ievo
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2
      double precision eva_fL_to_fL,eva_fR_to_fL

      eva_fX_to_fL =       fLpol*eva_fL_to_fL(gg2,gL2,mv2,x,mu2,ievo)
     &             + (1d0-fLpol)*eva_fR_to_fL(gg2,gR2,mv2,x,mu2,ievo)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_fR(gg2,gL2,gR2,fLpol,mv2,x,mu2,ievo)
      implicit none
      integer ievo
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2
      double precision eva_fL_to_fR,eva_fR_to_fR

      eva_fX_to_fR =       fLpol*eva_fL_to_fR(gg2,gL2,mv2,x,mu2,ievo)
     &             + (1d0-fLpol)*eva_fR_to_fR(gg2,gR2,mv2,x,mu2,ievo)
      return
      end      
c     /* ********************************************************* *      
c     EVA (1/6) for f_L > v_+
      double precision function eva_fL_to_vp(gg2,gL2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gL2,mv2,x,mu2
      double precision coup2,split,xxlog,fourPiSq
      data fourPiSq/39.47841760435743d0/ ! = 4pi**2

c      print*,'gg2,gL2,mv2,x,mu2,ievo',gg2 !3,gL2,mv2,x,mu2,ievo
      coup2 = gg2*gL2/fourPiSq
      split = (1.d0-x)**2 / 2.d0 / x
      if(ievo.eq.0) then
         xxlog = dlog(mu2/mv2)
      else
         xxlog = dlog(mu2/mv2/(1.d0-x))
      endif
      
      eva_fL_to_vp = coup2*split*xxlog
      return
      end
c     /* ********************************************************* *
c     EVA (2/6) for f_L > v_-
      double precision function eva_fL_to_vm(gg2,gL2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gL2,mv2,x,mu2
      double precision coup2,split,xxlog,fourPiSq
      data fourPiSq/39.47841760435743d0/ ! = 4pi**2
      
      coup2 = gg2*gL2/fourPiSq
      split = 1.d0 / 2.d0 / x
      if(ievo.eq.0) then
         xxlog = dlog(mu2/mv2)
      else
         xxlog = dlog(mu2/mv2/(1.d0-x))
      endif

      eva_fL_to_vm = coup2*split*xxlog
      return
      end
c     /* ********************************************************* *
c     EVA (3/6) for f_L > v_0
      double precision function eva_fL_to_v0(gg2,gL2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gL2,mv2,x,mu2
      double precision coup2,split,xxlog,fourPiSq
      data fourPiSq/39.47841760435743d0/ ! = 4pi**2
c
      coup2 = gg2*gL2/fourPiSq
      split = (1.d0-x) / x
      xxlog = 1.d0
      
      eva_fL_to_v0 = coup2*split*xxlog
      return
      end
c     /* ********************************************************* *
c     EVA (4/6) for f_R > v_+
      double precision function eva_fR_to_vp(gg2,gR2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gR2,mv2,x,mu2
      double precision eva_fL_to_vm

      eva_fR_to_vp = eva_fL_to_vm(gg2,gR2,mv2,x,mu2,ievo)
      return
      end
c     /* ********************************************************* *
c     EVA (5/6) for f_R > v_-
      double precision function eva_fR_to_vm(gg2,gR2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gR2,mv2,x,mu2
      double precision eva_fL_to_vp

      eva_fR_to_vm = eva_fL_to_vp(gg2,gR2,mv2,x,mu2,ievo)
      return
      end      
c     /* ********************************************************* *
c     EVA (6/6) for f_R > v_0
      double precision function eva_fR_to_v0(gg2,gR2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gR2,mv2,x,mu2
      double precision eva_fL_to_v0

      eva_fR_to_v0 = eva_fL_to_v0(gg2,gR2,mv2,x,mu2,ievo)
      return
      end
c     /* ********************************************************* *  
c     EVA () for f_L > f_L
c     fL_to_fL(z) = fL_to_vp(1-z) + fL_to_vm(1-z) 
      double precision function eva_fL_to_fL(gg2,gL2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gL2,mv2,x,mu2
      double precision tmpVp,tmpVm,z
      double precision eva_fL_to_vp,eva_fL_to_vm

      z = 1.d0 - x
      tmpVp = eva_fL_to_vp(gg2,gL2,mv2,z,mu2,ievo)
      tmpVm = eva_fL_to_vm(gg2,gL2,mv2,z,mu2,ievo)

      eva_fL_to_fL = tmpVp + tmpVm
      return
      end
c     /* ********************************************************* *  
c     EVA () for f_R > f_R
c     fR_to_fR(z) = fR_to_vp(1-z) + fR_to_vm(1-z) 
      double precision function eva_fR_to_fR(gg2,gR2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gR2,mv2,x,mu2
      double precision tmpVp,tmpVm,z
      double precision eva_fR_to_vp,eva_fR_to_vm

      z = 1.d0 - x
      tmpVp = eva_fR_to_vp(gg2,gR2,mv2,z,mu2,ievo)
      tmpVm = eva_fR_to_vm(gg2,gR2,mv2,z,mu2,ievo)

      eva_fR_to_fR = tmpVp + tmpVm
      return
      end      
c     /* ********************************************************* *  
c     EVA () for f_L > f_R
      double precision function eva_fL_to_fR(gg2,gL2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gL2,mv2,x,mu2

      eva_fL_to_fR = 0d0
      return
      end
c     /* ********************************************************* *       
c     EVA () for f_R > f_L
      double precision function eva_fR_to_fL(gg2,gR2,mv2,x,mu2,ievo)
      implicit none
      integer ievo              ! evolution by q2 or pT2
      double precision gg2,gR2,mv2,x,mu2

      eva_fR_to_fL = 0d0
      return
      end
c     /* ********************************************************* *       
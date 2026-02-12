c     /* ********************************************************* *
c                 Effective Vector Boson Approximation
c     /* ********************************************************* *      
c     File: ElectroweakFlux.f
c     R. Ruiz (2021 February)
c     R. Ruiz (2024 June -- update)
c     For details, see companion papers by:
c     -- EVA@LLA in MG5aMC: Ruiz, Costantini, et al [arXiv:2111.02442]
c     -- EVA@LP/NLP: Bigaran & Ruiz [arXiv:2502.07878]
c     /* ********************************************************* *
c     /* ********************************************************* *      
c     function eva_fX_to_vHEL(gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam,ievo,evaorder):
c     call electroweak PDF for for vector boson (HEL=0,+,-) from fermion
c     with fractional (0<pol<1) LH (RH) polarization of fLpol (1-fLpol)
c     - fLpol = 0.5d0 = parent fermion is spin-averaged
c     /* ********************************************************* *
c     /* ********************************************************* *      
c     function eva_fhel_to_vHEL(gg2,gF2,mv2,x,mu2,ebeam,ievo,evaorder):
c     electroweak PDF for vector boson (HEL=0,+,-) from fermion (hel=L/R)
c     - gg2 = (coupling)^2
c     - gF2 = (L/R chiral coupling)^2
c     - mv2 = (mass of boson)^2
c     - x   = momentum fraction = (E_V / E_f)
c     - mu2 = (max of evolution scale)^2
c     - ebeam = energy of beam [GeV]
c     - ievo: for evolution by virtuality (or !=0 for pT)      
c     - evaorder (EVA order): 0=EVA@lla, 1=EVA@lp, 2=iEVA@nlp
c     /* ********************************************************* *
c     /* ********************************************************* *
c     /* ********************************************************* *            
c     /* ********************************************************* *
      double precision function eva_fX_to_vp(gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam
      double precision eva_fL_to_vp,eva_fR_to_vp

      eva_fX_to_vp =       fLpol*eva_fL_to_vp(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
     &             + (1d0-fLpol)*eva_fR_to_vp(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_vm(gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam
      double precision eva_fL_to_vm,eva_fR_to_vm

      eva_fX_to_vm =       fLpol*eva_fL_to_vm(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
     &             + (1d0-fLpol)*eva_fR_to_vm(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_v0(gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam
      double precision eva_fL_to_v0,eva_fR_to_v0

      eva_fX_to_v0 =       fLpol*eva_fL_to_v0(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
     &             + (1d0-fLpol)*eva_fR_to_v0(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_fL(gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam
      double precision eva_fL_to_fL,eva_fR_to_fL

      eva_fX_to_fL =       fLpol*eva_fL_to_fL(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
     &             + (1d0-fLpol)*eva_fR_to_fL(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end
c     /* ********************************************************* *
      double precision function eva_fX_to_fR(gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gL2,gR2,fLpol,mv2,x,mu2,ebeam
      double precision eva_fL_to_fR,eva_fR_to_fR

      eva_fX_to_fR =       fLpol*eva_fL_to_fR(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
     &             + (1d0-fLpol)*eva_fR_to_fR(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end      
c     /* ********************************************************* *      
c     EVA (1/6) for f_L > v_+
      double precision function eva_fL_to_vp(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo      ! evolution by q2 or pT2
      integer evaorder  ! 0=EVA@lla, 1=EVA@lp, 2=iEVA@nlp
      double precision gg2,gL2,mv2,x,mu2,ebeam
      double precision coup2,split,xxlog,fourPiSq
      data fourPiSq/39.47841760435743d0/ ! = 4pi**2      
      double precision ev2,mvOev,muO2ev,mvOmu,muOmumv,mumvOmv
      double precision tmpXLP1,tmpXLP2,tmpNLP1,tmpNLP2,tmp
c      
      coup2 = gg2*gL2/fourPiSq
      split = (1.d0-x)**1 / 2.d0 / x ! note: here is the first (1-x)
c     needed for full LP / NLP
      ev2     = (x*ebeam)**2
      mvOev   = mv2 / ev2         ! -> 0 as ev->inf
      muO2ev  = mu2 / ev2 / 2.d0  ! -> 0 as ev->inf
      mvOmu   = mv2 / mu2         ! -> 0 as muf->inf, O(1) otherwise
      muOmumv = 1.0d0 / (1.0d0 + mvOmu) ! = mu2/(mu2+mv2) ! -> 1 as muf->inf 
      mumvOmv = (mu2/mv2) + 1.0d0       ! = (mu2+mv2)/mv2 ! -> inf as muf->inf
      xxlog   = dlog(mumvOmv) 
c     full LP / NLP terms
      tmpXLP1 = (1.d0-x)             ! note: here is the second (1-x)
      tmpXLP2 = tmpXLP1
      tmpNLP1 = (2.d0-x)*mvOev
      tmpNLP2 = tmpNLP1 + (2.d0-x)*muO2ev
c     set PDF according to order
      select case (evaorder)
      case (2) ! NLP
            tmp = xxlog*(tmpXLP1+tmpNLP1) - muOmumv*(tmpXLP2+tmpNLP2)
      case (1) ! full LP 
            tmp = xxlog*tmpXLP1 - muOmumv*tmpXLP2
      case default ! LLA
            if(ievo.eq.0) then
                  xxlog = dlog(mu2/mv2) ! update
            else
                  xxlog = dlog(mu2/mv2) - dlog(1.d0-x) ! update
            endif
            tmp = tmpXLP1*xxlog
      end select

      eva_fL_to_vp = coup2*split*tmp
      return
      end
c     /* ********************************************************* *
c     EVA (2/6) for f_L > v_-
      double precision function eva_fL_to_vm(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo  ! evolution by q2 or pT2
      integer evaorder  ! 0=EVA@lla, 1=EVA@lp, 2=iEVA@nlp
      double precision gg2,gL2,mv2,x,mu2,ebeam
      double precision coup2,split,xxlog,fourPiSq
      data fourPiSq/39.47841760435743d0/ ! = 4pi**2      
      double precision ev2,mvOev,muO2ev,mvOmu,muOmumv,mumvOmv
      double precision tmpXLP1,tmpXLP2,tmpNLP1,tmpNLP2,tmp
c
      coup2 = gg2*gL2/fourPiSq
      split = 1.d0 / 2.d0 / x
c     needed for full LP / NLP
      ev2     = (x*ebeam)**2
      mvOev   = mv2 / ev2         ! -> 0 as ev->inf
      muO2ev  = mu2 / ev2 / 2.d0  ! -> 0 as ev->inf
      mvOmu   = mv2 / mu2         ! -> 0 as muf->inf, O(1) otherwise
      muOmumv = 1.0d0 / (1.0d0 + mvOmu) ! = mu2/(mu2+mv2) ! -> 1 as muf->inf 
      mumvOmv = (mu2/mv2) + 1.0d0       ! = (mu2+mv2)/mv2 ! -> inf as muf->inf
      xxlog   = dlog(mumvOmv) 
c     full LP / NLP terms
      tmpXLP1 = 1.d0
      tmpXLP2 = tmpXLP1
      tmpNLP1 = (2.d0-x)*mvOev
      tmpNLP2 = tmpNLP1 + (2.d0-x)*muO2ev
c     set PDF according to order
      select case (evaorder)
      case (2) ! NLP
            tmp = xxlog*(tmpXLP1+tmpNLP1) - muOmumv*(tmpXLP2+tmpNLP2)
      case (1) ! full LP 
            tmp = xxlog*tmpXLP1 - muOmumv*tmpXLP2
      case default ! LLA
            if(ievo.eq.0) then
                  xxlog = dlog(mu2/mv2) ! update
            else
                  xxlog = dlog(mu2/mv2) - dlog(1.d0-x) ! update
            endif
            tmp = tmpXLP1*xxlog
      end select

      eva_fL_to_vm = coup2*split*tmp
      return
      end
c     /* ********************************************************* *
c     EVA (3/6) for f_L > v_0
      double precision function eva_fL_to_v0(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo      ! evolution by q2 or pT2
      integer evaorder  ! 0=EVA@lla, 1=EVA@lp, 2=iEVA@nlp
      double precision gg2,gL2,mv2,x,mu2,ebeam
      double precision coup2,split,xxlog,fourPiSq
      data fourPiSq/39.47841760435743d0/ ! = 4pi**2
      double precision ev2,mvOev,mvOmu,muOmumv,hflip(2),tmp(2)
      double precision eva_fL_to_vm,eva_fL_to_vp
c
      coup2 = gg2*gL2/fourPiSq
      split = (1.d0-x) / x
      xxlog = 1.d0
c     needed for full LP / NLP
      ev2     = (x*ebeam)**2
      mvOev   = mv2 / ev2     ! -> 0 as ev->inf
      mvOmu   = mv2 / mu2     ! -> 0 as muf->inf, O(1) otherwise
      muOmumv = 1.0d0 / (1.0d0 + mvOmu) ! = mu2/(mu2+mv2) ! -> 1 as muf->inf 
c     set PDF according to order
      select case (evaorder)
      case (2) ! NLP
            hflip(1) = eva_fL_to_vm(gg2,gL2,mv2,x,mu2,ebeam,ievo,1)
            hflip(2) = eva_fL_to_vp(gg2,gL2,mv2,x,mu2,ebeam,ievo,1)
            tmp(1) = muOmumv
            tmp(2) = 0.5d0 * mvOev * (hflip(1)+hflip(2))
      case (1) ! full LP 
            tmp(1) = muOmumv
            tmp(2) = 0.0d0
      case default ! LLA
            tmp(1) = 1.0d0
            tmp(2) = 0.0d0
      end select

      eva_fL_to_v0 = coup2*split*tmp(1) - tmp(2)
      return
      end
c     /* ********************************************************* *
c     EVA (4/6) for f_R > v_+
      double precision function eva_fR_to_vp(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gR2,mv2,x,mu2,ebeam
      double precision eva_fL_to_vm

      eva_fR_to_vp = eva_fL_to_vm(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end
c     /* ********************************************************* *
c     EVA (5/6) for f_R > v_-
      double precision function eva_fR_to_vm(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gR2,mv2,x,mu2,ebeam
      double precision eva_fL_to_vp

      eva_fR_to_vm = eva_fL_to_vp(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end      
c     /* ********************************************************* *
c     EVA (6/6) for f_R > v_0
      double precision function eva_fR_to_v0(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gR2,mv2,x,mu2,ebeam
      double precision eva_fL_to_v0

      eva_fR_to_v0 = eva_fL_to_v0(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      return
      end
c     /* ********************************************************* *  
c     EVA () for f_L > f_L
c     fL_to_fL(z) = fL_to_vp(1-z) + fL_to_vm(1-z) 
      double precision function eva_fL_to_fL(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gL2,mv2,x,mu2,ebeam
      double precision tmpVp,tmpVm,z
      double precision eva_fL_to_vp,eva_fL_to_vm

      z = 1.d0 - x
      tmpVp = eva_fL_to_vp(gg2,gL2,mv2,z,mu2,ebeam,ievo,evaorder)
      tmpVm = eva_fL_to_vm(gg2,gL2,mv2,z,mu2,ebeam,ievo,evaorder)

      eva_fL_to_fL = tmpVp + tmpVm
      return
      end
c     /* ********************************************************* *  
c     EVA () for f_R > f_R
c     fR_to_fR(z) = fR_to_vp(1-z) + fR_to_vm(1-z) 
      double precision function eva_fR_to_fR(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gR2,mv2,x,mu2,ebeam
      double precision tmpVp,tmpVm,z
      double precision eva_fR_to_vp,eva_fR_to_vm

      z = 1.d0 - x
      tmpVp = eva_fR_to_vp(gg2,gR2,mv2,z,mu2,ebeam,ievo,evaorder)
      tmpVm = eva_fR_to_vm(gg2,gR2,mv2,z,mu2,ebeam,ievo,evaorder)

      eva_fR_to_fR = tmpVp + tmpVm
      return
      end      
c     /* ********************************************************* *  
c     EVA () for f_L > f_R
      double precision function eva_fL_to_fR(gg2,gL2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gL2,mv2,x,mu2,ebeam

      eva_fL_to_fR = 0d0
      return
      end
c     /* ********************************************************* *       
c     EVA () for f_R > f_L
      double precision function eva_fR_to_fL(gg2,gR2,mv2,x,mu2,ebeam,ievo,evaorder)
      implicit none
      integer ievo, evaorder
      double precision gg2,gR2,mv2,x,mu2,ebeam

      eva_fR_to_fL = 0d0
      return
      end
c     /* ********************************************************* *       

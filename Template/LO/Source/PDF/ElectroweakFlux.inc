c     /* ********************************************************* *
c                 Effective Vector Boson Approximation
c     /* ********************************************************* *
c     File: ElectroweakFlux.inc
c     R. Ruiz (2021 February)
c     For details, see companion paper by Costantini, et al [arXiv:]
c     /* ********************************************************* *
	double precision eva_one,eva_zero,eva_half,eva_pi
	double precision eva_mz, eva_mw, eva_mh, eva_mx
	double precision eva_mz2,eva_mw2,eva_mh2,eva_mx2
	double precision eva_aEM,eva_ee2
	double precision eva_cw2,eva_sw2,eva_gw2,eva_gz2
	double precision eva_qeu, eva_qed, eva_qev, eva_qee
	double precision eva_qeu2,eva_qed2,eva_qev2,eva_qee2
	double precision eva_tLu,eva_tLd,eva_tLv,eva_tLe
	double precision eva_zRu,eva_zRd,eva_zRv,eva_zRe
	double precision eva_zLu,eva_zLd,eva_zLv,eva_zLe
	double precision eva_mu, eva_md, eva_mc, eva_ms, eva_mt, eva_mb
	double precision eva_mu2,eva_md2,eva_mc2,eva_ms2,eva_mt2,eva_mb2	
	double precision eva_me, eva_mm, eva_ml
	double precision eva_me2,eva_mm2,eva_ml2	
c
c	math
	parameter (eva_one = 1.d0)
	parameter (eva_zero = 0.d0)
	parameter (eva_half = 0.5d0)
	parameter (eva_pi   = 3.141592653589793d0)
c	boson mass inputs 2020 PDG
	parameter (eva_mw = 80.379d0)
	parameter (eva_mz = 91.1876d0)
	parameter (eva_mh = 125.10d0)
	parameter (eva_mx =  10.0d0)
	parameter (eva_mz2 = eva_mz**2)
	parameter (eva_mw2 = eva_mw**2)
	parameter (eva_mh2 = eva_mh**2)
	parameter (eva_mx2 = eva_mx**2)
c	fermion mass inputs 2020 PDG
	parameter (eva_mu = 2.16d-3)	
	parameter (eva_md = 4.67d-3)
	parameter (eva_mc = 1.27d0)
	parameter (eva_ms = 93.0d-3)
	parameter (eva_mt = 172.76d0)
	parameter (eva_mb = 4.18d0)
	parameter (eva_me = 0.5109989461d-3)
	parameter (eva_mm = 105.6583745d-3)
	parameter (eva_ml = 1.77686d0)
	parameter (eva_mu2 = eva_mu**2)		
	parameter (eva_md2 = eva_md**2)
	parameter (eva_mc2 = eva_mc**2)
	parameter (eva_ms2 = eva_ms**2)
	parameter (eva_mb2 = eva_mt**2)
	parameter (eva_mt2 = eva_mb**2)	
	parameter (eva_me2 = eva_me**2)
	parameter (eva_mm2 = eva_mm**2)
	parameter (eva_ml2 = eva_ml**2)	
c	coupling inputs
	parameter (eva_aEM = 1.d0/1.279000d2)
	parameter (eva_cw2 = eva_mw2/eva_mz2)
	parameter (eva_sw2 = 1.d0 - eva_cw2)
	parameter (eva_ee2 = 4.d0*eva_pi*eva_aEM)
	parameter (eva_gw2 = eva_ee2/eva_sw2)
	parameter (eva_gz2 = eva_gW2/eva_cw2)
c	gauge charges
	parameter (eva_qeu = +2d0/3d0)
	parameter (eva_qed = -1d0/3d0)
	parameter (eva_qev =  0d0)
	parameter (eva_qee = -1d0)
	parameter (eva_qeu2 = eva_qeu**2)
	parameter (eva_qed2 = eva_qed**2)
	parameter (eva_qev2 = eva_qev**2)
	parameter (eva_qee2 = eva_qee**2)
	parameter (eva_tLu = +0.5d0)
	parameter (eva_tLd = -0.5d0)	
	parameter (eva_tLv = +0.5d0)
	parameter (eva_tLe = -0.5d0)
	parameter (eva_zRu = -eva_qeu*eva_sw2)
	parameter (eva_zRd = -eva_qed*eva_sw2)
	parameter (eva_zRv = -eva_qev*eva_sw2)
	parameter (eva_zRe = -eva_qee*eva_sw2)
	parameter (eva_zLu = eva_tLu-eva_qeu*eva_sw2)
	parameter (eva_zLd = eva_tLd-eva_qed*eva_sw2)
	parameter (eva_zLv = eva_tLv-eva_qev*eva_sw2)
	parameter (eva_zLe = eva_tLe-eva_qee*eva_sw2)
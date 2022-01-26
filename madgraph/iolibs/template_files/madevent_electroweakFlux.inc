

c     /* ********************************************************* *
c                 Effective Vector Boson Approximation
c     /* ********************************************************* *
c     File: ElectroweakFlux.inc
c     R. Ruiz (2021 February)
c     For details, see companion paper by Costantini, et al [arXiv:]
c     /* ********************************************************* *
	double precision eva_one,eva_zero,eva_half,eva_pi,eva_sqr2
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

        save eva_mz, eva_mw, eva_mh, eva_mx
        save eva_mz2,eva_mw2,eva_mh2,eva_mx2
        save eva_aEM,eva_ee2
        save eva_cw2,eva_sw2,eva_gw2,eva_gz2
        save  eva_qeu, eva_qed, eva_qev, eva_qee
        save eva_qeu2,eva_qed2,eva_qev2,eva_qee2
        save eva_tLu,eva_tLd,eva_tLv,eva_tLe
        save eva_zRu,eva_zRd,eva_zRv,eva_zRe
        save eva_zLu,eva_zLd,eva_zLv,eva_zLe
        save eva_mu, eva_md, eva_mc, eva_ms, eva_mt, eva_mb
        save eva_mu2,eva_md2,eva_mc2,eva_ms2,eva_mt2,eva_mb2
        save eva_me, eva_mm, eva_ml
        save eva_me2,eva_mm2,eva_ml2

	parameter (eva_one  = 1.d0)
	parameter (eva_zero = 0.d0)
	parameter (eva_half = 0.5d0)
	parameter (eva_pi   = 3.141592653589793d0)
	parameter (eva_sqr2 = 1.414213562373095d0)

	include '../MODEL/coupl.inc'

	logical first
	data first /.true./
	save first

	if (first) then

c	boson mass inputs 2020 PDG
	eva_mw = %(MW)s
	eva_mz = %(MZ)s
	eva_mh = 125.10d0
	eva_mx =  10.0d0
	eva_mz2 = eva_mz**2
	eva_mw2 = eva_mw**2
	eva_mh2 = eva_mh**2
	eva_mx2 = eva_mx**2
c	fermion mass inputs 2020 PDG
	eva_mu = 2.16d-3	
	eva_md = 4.67d-3
	eva_mc = 1.27d0
	eva_ms = 93.0d-3
	eva_mt = 172.76d0
	eva_mb = 4.18d0
	eva_me = 0.5109989461d-3
	eva_mm = 105.6583745d-3
	eva_ml = 1.77686d0
	eva_mu2 = eva_mu**2		
	eva_md2 = eva_md**2
	eva_mc2 = eva_mc**2
	eva_ms2 = eva_ms**2
	eva_mb2 = eva_mt**2
	eva_mt2 = eva_mb**2	
	eva_me2 = eva_me**2
	eva_mm2 = eva_mm**2
	eva_ml2 = eva_ml**2	
c	coupling inputs
	eva_aEM = gal(1)*gal(1)*0.07957747154594767 ! 1/4/pi
        eva_ee2 = gal(1)*gal(1)
	eva_cw2 = eva_mw2/eva_mz2
	eva_sw2 = 1.d0 - eva_cw2
	eva_gw2 = eva_ee2/eva_sw2 		! aEM,MW,MZ scheme
	eva_gz2 = eva_gW2/eva_cw2
c	gauge charges
	eva_qeu = +2d0/3d0
	eva_qed = -1d0/3d0
	eva_qev =  0d0
	eva_qee = -1d0
	eva_qeu2 = eva_qeu**2
	eva_qed2 = eva_qed**2
	eva_qev2 = eva_qev**2
	eva_qee2 = eva_qee**2
	eva_tLu = +0.5d0
	eva_tLd = -0.5d0	
	eva_tLv = +0.5d0
	eva_tLe = -0.5d0
	eva_zRu = -eva_qeu*eva_sw2
	eva_zRd = -eva_qed*eva_sw2
	eva_zRv = -eva_qev*eva_sw2
	eva_zRe = -eva_qee*eva_sw2
	eva_zLu = eva_tLu-eva_qeu*eva_sw2
	eva_zLd = eva_tLd-eva_qed*eva_sw2
	eva_zLv = eva_tLv-eva_qev*eva_sw2
	eva_zLe = eva_tLe-eva_qee*eva_sw2

	first = .false.
	endif	   


c
c	math


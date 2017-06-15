      module extra_weights
      
         integer,parameter :: iwgtinfo=-5,maxscales=9,maxPDFs=200
     $     ,maxPDFsets=25,maxdynscales=10
         integer :: max_mom_str=1,max_mext=1,max_n_ctr=1
         logical :: doreweight,lscalevar(maxdynscales)
     $        ,lpdfvar(maxPDFsets)
         integer :: iwgtnumpartn,jwgtinfo,mexternal
     $        ,lhaPDFid(0:maxPDFsets),nmemPDF(maxPDFsets)
     $        ,dyn_scale(0:maxdynscales),n_ctr_found,n_mom_conf
         double precision :: wgtdegrem_xi,wgtdegrem_lxi,wgtdegrem_muF
     $        ,wgtnstmp,wgtwnstmpmuf,wgtwnstmpmur,wgtnstmp_avgvirt
     $        ,wgtref,scalevarR(0:maxscales),scalevarF(0:maxscales)
     $        ,wgtxsecmu(maxscales,maxscales,maxdynscales)
     $        ,wgtxsecPDF(0:maxPDFs,maxPDFsets),wgtbpower,wgtcpower
     $        ,veto_multiplier,H1_factor_virt,veto_compensating_factor
     $        ,born_wgt_veto
         double precision,allocatable :: momenta_str(:,:,:)
         character(len= 100) :: LHAPDFsetname(maxPDFsets) 
         character(len=1024),allocatable :: n_ctr_str(:)
        
c input of cpower (checked against calculated value)
         double precision,parameter :: cpowerinput=0d0
c switch for running muR-dependent factor runfac=1(running)/0(fixed)
         integer,parameter :: runfac=0
c WARNING: If you set runfac=1 to include a muR-dependent factor
c          make sure you modified the function rwgt_muR_dep_fac in
c	   reweight_xsec.f and compute_cpower in fks_singular.f
c          appropiately to include all muR dependent overall factors
c          (except for alpha_s) in the calculation. This procedure 
c          will be incorrect, if you miss one of the muR dependent
c          factors or if there is a not factorizing muR dependent term.
c	   You also have to set ren_group_coeff_in and cpowerinput
c	   to the proper values.

c first order coefficient of renormalization group equation of the
c muR-dependent factor,
c e.g. for masses:      ren_group_coeff = gamma0 = 3/2*C_F,
c i.e. also for Yukawa: ren_group_coeff = gamma0
         integer,parameter :: ren_group_coeff_in=0d0

         save
      end module extra_weights

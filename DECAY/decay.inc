c*****************************************************
c     PARAMETERS
c*****************************************************     	
c
c     VEGAS MAX DIMENSIONS
c
      integer 	 mxdim  	!to be the same as in vegas.f 
      parameter (mxdim=10)      !rebin.f
c                               
      integer    MaxParticles	 ! max number of particles in 
      parameter (MaxParticles=25)! the event
      integer    MaxDecPart	 ! max particles in the decay
      parameter (MaxDecPart=5)   ! 1->2,..,MaxDecPart  
      integer    MaxiEvent	 ! max nr of processes
      parameter (MaxiEvent=100)  ! in the proc_card
c
      integer MaxPoints  	! maxpoints used in the integration
      parameter(MaxPoints=500000)
c
c     Numbers
c
      double precision  Zero, One, Two, Three, Four, Half,Rt2
      parameter( Zero = 0.0d0, One = 1.0d0, Two = 2.0d0 )
      parameter( Three = 3.0d0, Four = 4.0d0, Half = 0.5d0 )
      parameter( Rt2   = 1.414213562d0 )
      double precision  Pi, Fourpi
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( Fourpi = Four * Pi )
c
c     Units	
c
      integer luni,lunr,lunw,lunp,lunrnd,lunout
      parameter (luni=15,lunr=16,lunw=17,lunp=18,lunrnd=19,lunout=20) 	
c
c*****************************************************
c     GLOBAL
c*****************************************************     	
c
c     IP    = PDG CODE FOR THE PARTICLE TO DECAY
c     IMODE = DECAY MODE
c     ND    = NUMBER OF DECAY PRODUCTS
c
      INTEGER       IP,IMODE,ND	
      COMMON/DEC_ID/IP,IMODE,ND
c
c     masses and couplings for the decay routines
c
      REAL*8          M1,M2,M3,M4,M5,MV,GV
      COMMON/m_decays/M1,M2,M3,M4,M5,MV,GV
      DOUBLE COMPLEX       GX,GXX(2),GXX1(2)  
      COMMON/WEAK_COUPL/   GX,GXX   ,GXX1
c
c    MC width, analytic width and best branching ratio
c     	
      real*8          MC_width,calc_width,bratio,calc_br
      common/to_width/MC_width,calc_width,bratio,calc_br
c
c    Particle's ID
c     	
      character*3 id(-100:100)
      real*8 pdgmass(-100:100)
      common/to_id/pdgmass,id
c
c    Color label
c	
      integer cindex 
      common/c_ind/cindex
c
c    Vegas 	
c
      real*4          x(MXDIM)
      COMMON/TO_VEGAS/x
      integer                                 n_wei,n_ove
      real*8          mxwgt,s_wei,s_unw,s_ove
      common/to_event/mxwgt,s_wei,s_unw,s_ove,n_wei,n_ove
c
c     HDECAY branching ratios
c
      integer           ihiggs, nnlo, ipole
      common /flag/     ihiggs, nnlo, ipole  
      integer           ionsh, ionwz, iofsusy
      common /onshell/  ionsh, ionwz, iofsusy 
      double precision  gf, alph, almass, ammuon, amz, amw
      common /param/    gf, alph, almass, ammuon, amz, amw 
      double precision  ams, amc, amb, amt
      common /masses/   ams, amc, amb, amt 
      double precision  gamc0, gamt0, gamt1, gamw, gamz
      common /wzwdth/   gamc0, gamt0, gamt1, gamw, gamz 
      double precision  SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,
     &                  SMBRG,SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
      common /widthsm/  SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,
     &                  SMBRG,SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
c
      double precision  hdals, hdmhbeg, hdmhend, tgbet
      common /hdparms/  hdals, hdmhbeg, hdmhend, tgbet
c
c     PDG branching ratios
c
      double precision br_ta_lv,br_ta_pi,br_ta_ro
      double precision br_w_lv,br_w_jj
      double precision br_z_ll,br_z_vv,br_z_cc,br_z_bb,br_z_jj 
      common/brs/      br_ta_lv,br_ta_pi,br_ta_ro,
     &                 br_w_lv,br_w_jj,
     &                 br_z_ll,br_z_vv,br_z_cc,br_z_bb,br_z_jj 


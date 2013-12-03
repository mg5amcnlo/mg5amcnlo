
      subroutine HEPNAM_old(ID,CHAU)

C...Purpose: to give the particle/parton name as a character string.

C       ID   = particle ID
C       CHAU = particle name

      character CHAU*20
      character CHSP(1605)*15

      integer hepchg_old,hepcmp_old
      external hepchg_old,hepcmp_old

      data (CHSP(I), I=1,119) /'d','u','s','c','b','t','l','h',2*' ',
     1'e','nu_e','mu','nu_mu','tau','nu_tau','chi','nu_chi','reggeon',
     2'pomeron','g','gamma','Z','W','H_1',' ','eta_tech','LQ','R','Z_2',
     3'W_2','H_2','H_2','H_2','Z_3','H_3','H_3','H_3','H_4','H_5',
     4'L susy d','L susy u','L susy s','L susy c',
     5'L susy b','L susy t','gluino','photino',2*' ',
     6'L susy e','L susy nu_e','L susy mu','L susy nu_mu',
     7'L susy tau','L susy nu_tau','susy HL','susy HH','susy HA',
     8' ','R susy d','R susy u','R susy s','R susy c',
     9'R susy b','R susy t','zino','susy Z2','susy Z3',
     *'susy Z4','R susy e','R susy nu_e','R susy mu',
     1'R susy nu_mu','R susy tau','R susy nu_tau','wino',
     2'susy W2',2*' ',19*'gen. code',' ','KS','KL','diquark',
     3'l-baryon','h-baryon',' ','rho_diffr','pi_diffr','omega_diffr',
     4'phi_diffr','psi_diffr','n_diffr','p_diffr',
     5'remnant gamma','remnant nucleon',4*' '/
      data (CHSP(I), I=120,219) /'pi','a_0(980)','pi(1300)',7*' ',
     1'rho(770)','b_1(1235)','a_1(1260)','rho(1450)','rho(1700)',5*' ',
     2'a_2(1320)','pi_2(1670)',8*' ','rho_3(1690)',19*' ','eta',
     3'f_0(975)','eta(1295)','f_0(1400)','eta(1440)','f_0(1590)',
     4'f_0(1710)',3*' ','omega(783)','h_1(1170)','f_1(1285)',
     5'omega(1390)','f_1(1420)','f_1(1510)','omega(1600)',3*' ',
     6'f_2(1270)',' ','f_2(2010)','f_2(2300)','f_2(2340)',5*' ',
     7'omega_3(1670)',9*' ','f_4(2050)',9*' '/
      data (CHSP(I), I=220,369) /'eta''(958)','f''0',8*' ','phi(1020)',
     1 'h_1(1380)','f_1(1420)','phi(1680)','f''_1',5*' ','f''_2(1525)',
     2 9*' ','phi_3(1850)',19*' ','eta_c(1S)','chi_c0(1P)','eta_c(2S)',
     3 7*' ','J/PSI','hc(1P)','chi_c1(1P)','PSI(2S)','PSI(3770)',
     4 'PSI(4040)','PSI(4160)','PSI(4415)',2*' ','chi_c2(1P)',29*' ',
     5 'eta_b(1S)','chi_b0(1P)','eta_b(2S)','chi_b0(2P)','eta_b(3S)',
     6 5*' ','upsilon(1S)','h_b(1P)','chi_b1(1P)','upsilon(2S)',
     7 'h_b(2P)','chi_b1(2P)','upsilon(3S)','upsilon(4S)',
     8 'upsilon(10860)','upsilon(11020)','chi_b2(1P)','chi_b2(2P)',
     9 28*' '/
      data (CHSP(I), I=370,419) /'eta_t','chi_t0',8*' ','theta',
     1'h_1t','chi_t1',7*' ','chi_t2',29*' '/
      data (CHSP(I), I=420,449) /'eta_l','chi_l0',' ',
     2'theta_l','h_1l','chi_l1','chi_l2',8*' ','eta_h',
     3'chi_h0',' ','theta_h','h_1h','chi_h1','chi_h2',8*' '/
      data (CHSP(I), I=450,499) /'pi','a_0(980)','pi(1300)',7*' ',
     1'rho(770)','b_1(1235)','a_1(1260)','rho(1450)','rho(1700)',5*' ',
     2'a_2(1320)','pi_2(1670)',8*' ','rho_3(1690)',19*' '/
      data (CHSP(I), I=500,599) /'K','K*_0(1430)',8*' ','K*(892)',
     1'K_1(1270)','K_1(1400)','K*(1410)','K*(1680)',5*' ','K*_2(1430)',
     2'K_2(1770)','K_2(1820)',7*' ','K*_3(1780)',9*' ','K*_4(2045)',
     3 9*' ','K','K*_0(1430)',8*' ','K*(892)','K_1(1270)','K_1(1400)',
     4'K*(1410)','K*(1680)',5*' ','K*_2(1430)','K_2(1770)','K_2(1820)',
     5 7*' ','K*_3(1780)',9*' ','K*_4(2045)',9*' '/
      data (CHSP(I), I=600,749) /'D','D*_0',8*' ','D*(2010)',
     1'D_1(2420)','D*_1',7*' ','D*_2(2460)',29*' ','D','D*_0',8*' ',
     2'D*(2010)','D_1(2420)','D*_1',7*' ','D*_2(2460)',29*' ','D_s',
     3'D*_s0',8*' ','D*_s','D_s1(2536)','D*_s1',7*' ','D*_s2',29*' '/
      data (CHSP(I), I=750,949) /'B','B*_0',8*' ','B*',
     1'B_1','B*_1',7*' ','B*_2',29*' ','B','B*_0',8*' ',
     2'B*','B_1','B*_1',7*' ','B*_2',29*' ','B_s',
     3'B*_s0',8*' ','B*_s','B_s1','B*_s1',7*' ','B*_s2',29*' ','B_c',
     3'B*_c0',8*' ','B*_c','B_c1','B*_c1',7*' ','B*_c2',29*' '/
      data (CHSP(I), I=950,1199) /'T','T*_0',8*' ','T*',
     1'T_1','T*_1',7*' ','T*_2',29*' ','T','T*_0',8*' ',
     2'T*','T_1','T*_1',7*' ','T*_2',29*' ','T_s',
     3'T*_s0',8*' ','T*_s','T_s1','T*_s1',7*' ','T*_s2',29*' ','T_c',
     3'T*_c0',8*' ','T*_c','T_c1','T*_c1',7*' ','T*_c2',29*' ','T_b',
     3'T*_b0',8*' ','T*_b','T_b1','T*_b1',7*' ','T*_b2',29*' '/
      data (CHSP(I), I=1200,1289) /'L','L*_0',' ','L*','L_1','L*_1',
     1'L*_2',8*' ','L','L*_0',' ','L*','L_1','L*_1','L*_2',8*' ',
     2'L_s','L*_s0',' ','L*_s','L_s1','L*_s1','L*_s2',8*' ',
     3'L_c','L*_c0',' ','L*_c','L_c1','L*_c1','L*_c2',8*' ',
     4'L_b','L*_b0',' ','L*_b','L_b1','L*_b1','L*_b2',8*' ',
     5'L_t','L*_t0',' ','L*_t','L_t1','L*_t1','L*_t2',8*' '/
      data (CHSP(I), I=1290,1394) /'H','H*_0',' ','H*','H_1','H*_1',
     1'H*_2',8*' ','H','H*_0',' ','H*','H_1','H*_1','H*_2',8*' ',
     2'H_s','H*_s0',' ','H*_s','H_s1','H*_s1','H*_s2',8*' ',
     3'H_c','H*_c0',' ','H*_c','H_c1','H*_c1','H*_c2',8*' ',
     4'H_b','H*_b0',' ','H*_b','H_b1','H*_b1','H*_b2',8*' ',
     5'H_t','H*_t0',' ','H*_t','H_t1','H*_t1','H*_t2',8*' ',
     6'H_l','H*_l0',' ','H*_l','H_l1','H*_l1','H*_l2',8*' '/
      data (CHSP(I), I=1395,1415) /' ','Lambda','Lambda_c',
     1'Xi_c','Xi_c','Lambda_b','Xi_b','Xi_b','Xi_bc',
     2'Xi_bc','Omega_bc','Lambda_t','Xi_t','Xi_t',
     3'Xi_tc','Xi_tc','Omega_tc','Xi_tb','Xi_tb',
     4'Omega_tb','Omega_tbc'/
      data (CHSP(I), I=1416,1475) /' ','n','p',' ','Sigma','Sigma',
     1'Sigma','Xi','Xi',' ','Sigma_c','Sigma_c','Sigma_c','Xi_c''',
     2'Xi_c''','Omega_c','Xi_cc','Xi_cc','Omega_cc',' ','Sigma_b',
     3'Sigma_b','Sigma_b','Xi_b''','Xi_b''','Omega_b','Xi_bc''',
     4'Xi_bc''','Omega_bc''','Omega_bcc','Xi_bb','Xi_bb','Omega_bb',
     5'Omega_bbc',' ','Sigma_t','Sigma_t','Sigma_t','Xi_t''','Xi_t''',
     6'Omega_t','Xi_tc''','Xi_tc''','Omega_tc''','Omega_tcc','Xi_tb''',
     7'Xi_tb''','Omega_tb''','Omega_tbc''','Omega_tbb','Xi_tt','Xi_tt',
     8'Omega_tt','Omega_ttc','Omega_ttb',5*' '/
      data (CHSP(I), I=1476,1535) /'Delta','Delta','Delta','Delta',
     1'Sigma*','Sigma*','Sigma*','Xi*','Xi*','Omega',
     2'Sigma*_c','Sigma*_c','Sigma*_c','Xi*_c','Xi*_c','Omega*_c',
     3'Xi*_cc','Xi*_cc','Omega*_cc','Omega*_ccc',
     4'Sigma*_b','Sigma*_b','Sigma*_b','Xi*_b','Xi*_b','Omega*_b',
     5'Xi*_bc','Xi*_bc','Omega*_bc','Omega*_bcc',
     6'Xi*_bb','Xi*_bb','Omega*_bb','Omega*_bbc','Omega*_bbb',
     7'Sigma*_t','Sigma*_t','Sigma*_t','Xi*_t','Xi*_t','Omega*_t',
     8'Xi*_tc','Xi*_tc','Omega*_tc','Omega*_tcc','Xi*_tb','Xi*_tb',
     9'Omega*_tb','Omega*_tbc','Omega*_tbb','Xi*_tt','Xi*_tt',
     *'Omega*_tt','Omega*_ttc','Omega*_ttb','Omega*_ttt',4*' '/
      data (CHSP(I), I=1536,1600) /65*' '/
      data (CHSP(I), I=1601,1605) /'Hydrogen','Deuteron','Tritium',
     1'He3','Alpha'/

      save CHSP


C...Initial values. Charge (in units of 1/3). Subdivide code.
      CHAU=' '
      IDA=IABS(ID)
      if(IDA.EQ.0) return

      IDCM=hepcmp_old(ID)
      if(IDCM.LE.0) return

      LEN=0
      KQ=hepchg_old(ID)
      KQN=MOD(IDA/1000000000,10)
      IDQ3=MOD(IDA/1000,10)
      IDQ2=MOD(IDA/100,10)
      IDQ1=MOD(IDA/10,10)
      IDJS=MOD(IDA,10)
      IDEX=MOD(IDA/10000,10)

C...Read out root name and spin for simple particles and special cases
      if(IDA.LE.100) then
        CHAU=CHSP(IDCM)
        LEN=0
        do 100 LEM=1,20
  100     if(CHAU(LEM:LEM).NE.' ') LEN=LEM
      elseif(KQN.eq.1) then
        CHAU=CHSP(IDCM)
        LEN=0
        do 110 LEM=1,20
  110     if(CHAU(LEM:LEM).NE.' ') LEN=LEM
      elseif(IDJS.EQ.0) then
        CHAU=CHSP(IDCM)
        LEN=0
        do 150 LEM=1,20
  150     if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Construct root name for diquark. Add on spin.
      elseif(IDQ1.EQ.0) then
        CHAU(1:2)=CHSP(IDQ3)(1:1)//CHSP(IDQ2)(1:1)
        if(IDJS.EQ.1) CHAU(3:4)='_0'
        if(IDJS.EQ.3) CHAU(3:4)='_1'
        LEN=4

C...Construct root name for meson.
      elseif(IDQ3.EQ.0) then
        CHAU=CHSP(IDCM)
        LEN=0
        do 200 LEM=1,20
  200     if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Construct root name for Herwig remnant particles
      elseif(IDQ1.eq.9 .and. IDQ2.eq.9 .and. IDQ3.eq.9) then
        CHAU=CHSP(IDCM)
        LEN=0
        do 250 LEM=1,20
  250     if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Construct root name and spin for baryon.
      else
        if(IDQ3.LE.6) then
          CHAU=CHSP(IDCM)
          LEN=0
          do 300 LEM=1,20
  300       if(CHAU(LEM:LEM).NE.' ') LEN=LEM
        else
C...Construct root name and spin for heavy baryon.
          LEN=0
          if(IDQ2.LE.2 .AND. IDQ1.LE.2)then
            CHAU='Sigma'
            if(IDJS.EQ.4) CHAU='Sigma*'
            if(IDQ1.GT.IDQ2) CHAU='Lambda'
          elseif(IDQ1.LE.2) then
            CHAU='Xi'''
            if(IDJS.EQ.4) CHAU='Xi*'
          elseif(IDQ2.LE.2) then
            CHAU='Xi'
          elseif(IDQ1.GT.IDQ2) then
            CHAU='Omega'
          else
            CHAU='Omega'''
            if(IDJS.EQ.4) CHAU='Omega*'
          endif
          do 320 LEM=1,20
  320       if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Add on heavy flavour content for heavy baryon.
          CHAU(LEN+1:LEN+2)='_'//CHSP(IDQ3)(1:1)
          LEN=LEN+2
          if(IDQ2.GE.IDQ1.AND.IDQ1.GE.4) then
            CHAU(LEN+1:LEN+2)=CHSP(IDQ2)(1:1)//CHSP(IDQ1)(1:1)
            LEN=LEN+2
          elseif(IDQ2.GE.IDQ1.AND.IDQ2.GE.4) then
            CHAU(LEN+1:LEN+1)=CHSP(IDQ2)(1:1)
            LEN=LEN+1
          elseif(IDQ1.GT.IDQ2.AND.IDQ2.GE.4) then
            CHAU(LEN+1:LEN+2)=CHSP(IDQ1)(1:1)//CHSP(IDQ2)(1:1)
            LEN=LEN+2
          elseif(IDQ1.GT.IDQ2.AND.IDQ1.GE.4) then
            CHAU(LEN+1:LEN+1)=CHSP(IDQ1)(1:1)
            LEN=LEN+1
          endif
        endif
      endif

C...Add on bar sign for antiparticle (where necessary).
      if(ID.GT.0.OR.LEN.EQ.0) then
      elseif(IDA.GT.10.AND.IDA.LE.40.AND.KQ.NE.0) then
      elseif(IDA.GT.46.AND.IDA.LE.60.AND.KQ.NE.0) then
      elseif(IDA.GT.66.AND.IDA.LE.80.AND.KQ.NE.0) then
      elseif(IDA.GE.81.AND.IDA.LE.99) then
      elseif(IDA.GT.100.AND.IDQ3.EQ.0.AND.KQ.NE.0) then
      else
        CHAU(LEN+1:LEN+1)='~'
        LEN=LEN+1
      endif

C...Add on charge where applicable (conventional cases skipped).
      if(IDA.GE.81.AND.IDA.LE.100) then
C...generator specific codes
      elseif(IDCM.eq.114 .or. IDCM.eq.115) then
C...Herwig remnant particles
      elseif(KQN.eq.1) then
C...ions
      elseif(LEN.NE.0)then
C...everything else
        if(KQ.EQ.6) CHAU(LEN+1:LEN+2)='++'
        if(KQ.EQ.-6) CHAU(LEN+1:LEN+2)='--'
        if(KQ.EQ.3) CHAU(LEN+1:LEN+1)='+'
        if(KQ.EQ.-3) CHAU(LEN+1:LEN+1)='-'

        if(KQ.EQ.0.AND.IDA.LE.22) then
C...quarks and leptons
        elseif(KQ.EQ.0.AND.(IDA.GE.41.AND.IDA.LE.56)) then
C...left squarks, sleptons, etc.
        elseif(KQ.EQ.0.AND.(IDA.GE.61.AND.IDA.LE.67)) then
C...right squarks
        elseif(KQ.EQ.0.AND.(IDA.GE.71.AND.IDA.LE.76)) then
C...right sleptons
        elseif(IDA.GT.100.AND.IDQ3.EQ.0.AND.IDQ2.EQ.IDQ1.AND.
     &      IDQ2.NE.1) then
C... eta, psi, upsilon, etc.
        elseif(KQ.EQ.0) then
          CHAU(LEN+1:LEN+1)='0'
        endif
      endif

      return
      end

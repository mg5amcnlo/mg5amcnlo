
      subroutine hepnam_2004(id,chaup)

C...Purpose: to give the particle/parton name as a character string.

C       ID   = particle ID
C       chau = particle name

      integer id
      character chaup*(*)
      character chau*20
      character chsp(2210)*16

      integer hepchg_2004,hepcmp_2004
      external hepchg_2004,hepcmp_2004

      data (chsp(i), i=1,100) /
     1'd','u','s','c','b','t','b''','t''',' ',' ',
     2'e','nu_e','mu','nu_mu','tau','nu_tau','tau''','nu_tau''',' ',' ',
     4'g','gamma','Z','W','H_1',' ',' ',' ',' ',' ',
     5' ','Z_2','Z_3','W_2','H_2','H_3','H',' ','G',' ',
     6'R','LQc',' ',' ',' ',' ',' ',' ',' ',' ',
     7'H_L','H_1','H_2','H_2','H_4',' ',' ',' ',' ',' ',
     8 20*' ',20*'gen. code'/
      data (chsp(i), i=101,110) /
     9 'KS','KL','diquark','l-baryon','h-baryon',
     1' ','reggeon','pomeron','odderon',' '/
      data (chsp(i), i=111,190) /
     1'L susy d','L susy u','L susy s','L susy c',
     2'L susy b','L susy t',4*' ',
     3'L susy e','L susy nu_e','L susy mu','L susy nu_mu',
     4'L susy tau','L susy nu_tau',4*' ',
     5'gluino','susy chi_1','susy chi_2','susy chi_1','susy chi_3',
     6 9*' ',
     7'susy chi_4',' ','susy chi_2',
     8 ' ','gravitino',16*' ',5*' ',
     9'R susy d','R susy u','R susy s','R susy c',
     1'R susy b','R susy t',4*' ',
     1'R susy e','R susy nu_e','R susy mu','R susy nu_mu',
     2'R susy tau','R susy nu_tau',4*' '/
      data (chsp(i), i=191,210) /
     1'pi_tech','pi_tech','pi''_tech','rho_tech','rho_tech','eta_tech',
     1'omega_tech',3*' ',
     2'V8_tech','pi_tech_22_1','pi_tech_22_8',
     2'rho_tech_11','rho_tech_12','rho_tech_21','rho_tech_22',3*' '/
      data (chsp(i), i=211,230) /
     4 'd*','u*',8*' ','e*','nu*_e',7*' ','G*'/
      data (chsp(i), i=231,240) /
     4 'H_L','H_R','W_R','nu_Re','nu_Rmu','nu_Rtau','Z_R',
     4 'Theta','Phi',' '/
      data (chsp(i), i=241,259) /
     5'rho_diffr','pi_diffr','omega_diffr','phi_diffr',
     6'psi_diffr','n_diffr','p_diffr',
     6'ccbar[3S1(8)]','ccbar[1S0(8)]','ccbar[3P0(8)]',
     7'remnant photon','remnant nucleon','gamma_virt','Cerenkov',
     8'bbbar[3S1(8)]','bbbar[1S0(8)]','bbbar[3P0(8)]',2*' '/
      data (chsp(i), i=260,559) /'pi','rho(770)','a_2(1320)',
     1'rho_3(1690)','a_4(2040)','a_0(980)','pi_1(1400)',
     2'pi_2(2100)','rho_3(1900)',' ','a_0(1450)','b_1(1235)',
     3'pi_2(1670)',2*' ','pi(1800)','pi_1(1600)',' ','rho_3(2250)',
     4 2*' ','a_1(1260)',4*' ','rho_0(1900)',4*' ',
     5'rho(1700)',4*' ','rho_0(2150)',9*' ',' ',3*' '
     6,'pi(1300)','rho(1450)',' ',37*' ',11*' ',
     7 39*' ',10*' ','pi','rho(770)','a_2(1320)',
     1'rho_3(1690)','a_4(2040)','a_0(980)','pi_1(1400)',
     2'pi_2(2100)','rho_3(1900)',' ','a_0(1450)','b_1(1235)',
     3'pi_2(1670)',2*' ','pi(1800)','pi_1(1600)',' ','rho_3(2250)',
     4 2*' ','a_1(1260)',4*' ','rho_0(1900)',4*' ',
     5'rho(1700)',4*' ','rho_0(2150)',9*' ',' ',3*' ',
     6'pi(1300)','rho(1450)',' ',37*' ',11*' ',
     7 39*' ',10*' '/

      data (chsp(i), i=560,859) /'eta','omega(782)','f_2(1270)',
     1'omega_3(1670)','f_4(2050)','f_0(600)','f_1(1510)',
     2'f_2(1430)',' ','f_J(2220)','f_0(1370)','h_1(1170)','eta_2(1645)',
     2 2*' ',
     3'f_0(980)',' ','f_2(1565)',' ','f_4(2300)',' ','f_1(1285)',3*' ',
     4'eta(1405)',' ','f_2(1640)',3*' ','omega(1650)',3*' ','f_0(1500)',
     5' ','f_2(1810)',7*' ','f_0(2020)',' ','f_2(1910)',7*' ',
     6'f_0(2100)',' ','f_2(1950)',7*' ','f_0(2200)',' ',
     6 'f_2(2010)',7*' ','f_0(2330)',' ','f_2(2150)',9*' ','f_2(2300)',
     7 9*' ','f_2(2340)',2*' ','eta(1295)','omega(1420)',' ',77*' ',
     8 120*' '/

      data (chsp(i), i=860,919) /'eta''(958)','phi(1020)','f''_2(1525)',
     1'phi_3(1850)',5*' ',' ','f_0(1710)','h_1(1380)',
     2 'eta_2(1870)',8*' ',
     2'f_1(1420)',8*' ','eta(1475)','phi(1680)',' ',27*' '/
      data (chsp(i), i=920,1159) /'K','K*(892)','K*_2(1430)',
     1'K*_3(1780)','K*_4(2045)','K*_0(800)',' ','K_2(1580)',
     2' ','K_4(2500)','K*_0(1430)','K_1(1270)','K_2(1770)',2*' ',
     3'K*_0(1950)',' ','K_2(2250)','K_3(2320)',2*' ','K_1(1400)',
     4'K_2(1820)',8*' ','K*(1680)',8*' ','K(1460)','K*(1410)',
     5' ',37*' ',' ',39*' ','K','K*(892)','K*_2(1430)',
     1'K*_3(1780)','K*_4(2045)','K*_0(800)',' ','K_2(1580)',
     2' ','K_4(2500)','K*_0(1430)','K_1(1270)','K_2(1770)',2*' ',
     3'K*_0(1950)',' ','K_2(2250)','K_3(2320)',2*' ','K_1(1400)',
     4'K_2(1820)',8*' ','K*(1680)',8*' ','K(1460)','K*(1410)',
     5'K_2(1980)',37*' ',' ',39*' '/
      data (chsp(i), i=1160,1249) /'D','D*(2010)','D*_2(2460)',7*' ',
     1'D*_0','D_1(2420)',9*' ','D_1(H)',8*' ',
     2'D','D*(2010)','D*_2(2460)',7*' ',
     3'D*_0','D_1(2420)',9*' ','D_1(H)',8*' ',
     4'D_s','D*_s','D*_s2',7*' ','D*_s0','D_s1(2536)',9*' ',
     5'D_s1(H)',8*' '/
      data (chsp(i), i=1250,1339) /'D(2S)','D*(2S)',28*' ',
     1'D(2S)','D*(2S)',58*' '/
      data (chsp(i), i=1340,1429) /90*' '/
      data (chsp(i), i=1430,1509) /'eta_c(1S)','J/psi(1S)','chi_c2(1P)',
     1 3*' ','psi(4040)','psi(3836)',2*' ',
     2 'chi_c0(1P)','hc(1P)',4*' ','psi(4160)',
     3 4*' ','chi_c1(1P)',4*' ','psi(4415)',4*' ',
     4'psi(3770)',8*' ','eta_c(2S)','psi(2S)',38*' '/
      data (chsp(i), i=1510,1629) /'B','B*','B*_2',7*' ',
     1'B*_0','B_1(L)',9*' ','B_1(H)',8*' ','B','B*','B*_2',7*' ',
     2'B*_0','B_1(L)',9*' ','B_1(H)',8*' ','B_s','B*_s','B*_s2',
     3 7*' ','B*_s0','B_s1(L)',' ',8*' ','B_s1(H)',8*' ',
     4'B_c','B*_c','B*_c2',7*' ','B*_c0','B_c1(L)',9*' ','B_c1(H)',
     5 8*' '/
      data (chsp(i), i=1630,1789) /'eta_b(1S)','upsilon(1S)',
     1'chi_b2(1P)','upsilon_3(1D)',2*' ','upsilon(10860)',3*' ',
     2'chi_b0(1P)','h_b(1P)','eta_b2(1D)',3*' ','upsilon(11020)',4*' ',
     3'chi_b1(1P)','upsilon_2(1D)',3*' ','upsilon(7S)',
     3 4*' ','upsilon_1(1D)',8*' ',
     4'eta_b(2S)','upsilon(2S)','chi_b2(2P)','upsilon_3(2D)',6*' ',
     5'chi_b0(2P)','h_b(2P)','eta_b2(2D)',8*' ','chi_b1(2P)',
     6'upsilon_2(2D)',8*' ','upsilon_1(2D)',8*' ','eta_b(3S)',
     7'upsilon(3S)','chi_b2(3P)',7*' ','chi_b0(3P)','h_b(3P)',9*' ',
     8'chi_b1(3P)',19*' ','upsilon(4S)',38*' '/
      data (chsp(i), i=1790,1849) /'T','T*',8*' ','T','T*',8*' ',
     1'T_s','T*_s',8*' ','T_c','T*_c',8*' ','T_b','T*_b',8*' ',
     2'eta_t','theta',8*' '/
      data (chsp(i), i=1850,1919) /'L','L*',8*' ','L','L*',8*' ',
     1'L_s','L*_s',8*' ','L_c','L*_c',8*' ','L_b','L*_b',8*' ',
     2'L_t','L*_t',8*' ','eta_l','theta_l',8*' '/
      data (chsp(i), i=1920,1999) /'H','H*',8*' ','H','H*',8*' ',
     1'H_s','H*_s',8*' ','H_c','H*_c',8*' ','H_b','H*_b',8*' ',
     2'H_t','H*_t',8*' ','H_l','H*_l',8*' ','eta_h','theta_H',8*' '/

      data (chsp(i), i=2000,2020) /' ','Lambda','Lambda_c',
     1'Xi_c','Xi_c','Lambda_b','Xi_b','Xi_b','Xi_bc',
     2'Xi_bc','Omega_bc','Lambda_t','Xi_t','Xi_t',
     3'Xi_tc','Xi_tc','Omega_tc','Xi_tb','Xi_tb',
     4'Omega_tb','Omega_tbc'/
      data (chsp(i), i=2021,2080) /' ','n','p',' ','Sigma','Sigma',
     1'Sigma','Xi','Xi',' ','Sigma_c','Sigma_c','Sigma_c','Xi''_c',
     2'Xi''_c','Omega_c','Xi_cc','Xi_cc','Omega_cc',' ','Sigma_b',
     3'Sigma_b','Sigma_b','Xi''_b','Xi''_b','Omega_b','Xi''_bc',
     4'Xi''_bc','Omega''_bc','Omega_bcc','Xi_bb','Xi_bb','Omega_bb',
     5'Omega_bbc',' ','Sigma_t','Sigma_t','Sigma_t','Xi''_t','Xi''_t',
     6'Omega_t','Xi''_tc','Xi''_tc','Omega''_tc','Omega_tcc','Xi''_tb',
     7'Xi''_tb','Omega''_tb','Omega''_tbc','Omega_tbb','Xi_tt','Xi_tt',
     8'Omega_tt','Omega_ttc','Omega_ttb',5*' '/
      data (chsp(i), i=2081,2140) /'Delta','Delta','Delta','Delta',
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
      data (chsp(i), i=2141,2200) /60*' '/
      data (chsp(i), i=2201,2210) /'Hydrogen','Deuteron','Tritium',
     1'He3','Alpha',5*' '/

      save chsp


C...Initial values. Charge (in units of 1/3). Subdivide code.
      chau=' '
      chaup = chau
      ida=iabs(id)
      if(ida.eq.0) return

      idcm=hepcmp_2004(id)
      if(idcm.le.0) return

      len=0
      kq=hepchg_2004(id)
      kqn=mod(ida/1000000000,10)
      kqx=mod(ida/1000000,10)
      kqr=mod(ida/100000,10)
      kql=mod(ida/10000,10)
      idq3=mod(ida/1000,10)
      idq2=mod(ida/100,10)
      idq1=mod(ida/10,10)
      idjs=mod(ida,10)
      irt = mod(ida,10000)
      k99=mod(ida/100000,100)

C...Read out root name and spin for simple particles and special cases
      if(ida.le.100) then
        chau=chsp(idcm)
        len=0
        do 100 lem=1,20
  100     if(chau(lem:lem).ne.' ') len=lem
      elseif(kqn.eq.1) then
        chau=chsp(idcm)
        len=0
        do 110 lem=1,20
  110     if(chau(lem:lem).ne.' ') len=lem
      elseif(idjs.eq.0) then
        chau=chsp(idcm)
        len=0
        do 150 lem=1,20
  150     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name for SUSY, technicolor, and excited particles
      elseif(kqx.ge.1 .and. kqx.le.5) then
        chau=chsp(idcm)
        len=0
        do 250 lem=1,20
  250     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name for miscellaneous particles
      elseif(k99.eq.99) then
        chau=chsp(idcm)
        len=0
        do 260 lem=1,20
  260     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name for diquark. Add on spin.
      elseif(idq1.eq.0) then
        if(idq3.lt.7) then
          chau(1:2)=chsp(idq3)(1:1)//chsp(idq2)(1:1)
          if(idjs.eq.1) chau(3:4)='_0'
          if(idjs.eq.3) chau(3:4)='_1'
          len=4
	elseif(idq3.ge.7 .and. idq2.lt.7) then
          chau(1:3)=chsp(idq3)(1:2)//chsp(idq2)(1:1)
          if(idjs.eq.1) chau(4:5)='_0'
          if(idjs.eq.3) chau(4:5)='_1'
          len=5
	elseif(idq3.ge.7 .and. idq2.ge.7) then
          chau(1:4)=chsp(idq3)(1:2)//chsp(idq2)(1:2)
          if(idjs.eq.1) chau(5:6)='_0'
          if(idjs.eq.3) chau(5:6)='_1'
          len=6
	endif

C...Construct root name for meson.
      elseif(idq3.eq.0) then
        chau=chsp(idcm)
        len=0
        do 200 lem=1,20
  200     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name and spin for baryon.
      else
        if(idq3.le.6) then
          chau=chsp(idcm)
          len=0
          do 300 lem=1,20
  300       if(chau(lem:lem).ne.' ') len=lem
        else
C...Construct root name and spin for heavy baryon.
          len=0
          if(idq2.le.2 .and. idq1.le.2)then
            chau='Sigma'
            if(idjs.eq.4) chau='Sigma*'
            if(idq1.gt.idq2) chau='Lambda'
          elseif(idq1.le.2) then
            chau='Xi'''
            if(idjs.eq.4) chau='Xi*'
          elseif(idq2.le.2) then
            chau='Xi'
          elseif(idq1.gt.idq2) then
            chau='Omega'
          else
            chau='Omega'''
            if(idjs.eq.4) chau='Omega*'
          endif
          do 320 lem=1,20
  320       if(chau(lem:lem).ne.' ') len=lem

C...Add on heavy flavour content for heavy baryon.
          chau(len+1:len+3)='_'//chsp(idq3)(1:2)
          len=len+3
          if(idq2.ge.idq1.and.idq1.ge.4) then
	    if(idq2.lt.7 .and. idq1.lt.7) then
              chau(len+1:len+2)=chsp(idq2)(1:1)//chsp(idq1)(1:1)
              len=len+2
	    elseif(idq2.ge.7 .and. idq1.lt.7) then
              chau(len+1:len+3)=chsp(idq2)(1:2)//chsp(idq1)(1:1)
              len=len+3
	    else 
              chau(len+1:len+4)=chsp(idq2)(1:2)//chsp(idq1)(1:2)
              len=len+4
	    endif
          elseif(idq2.ge.idq1.and.idq2.ge.4) then
	    if(idq2.lt.7) then
              chau(len+1:len+1)=chsp(idq2)(1:1)
              len=len+1
	    else
              chau(len+1:len+2)=chsp(idq2)(1:2)
              len=len+2
	    endif
          elseif(idq1.gt.idq2.and.idq2.ge.4) then
	    if(idq2.lt.7 .and. idq1.lt.7) then
              chau(len+1:len+2)=chsp(idq1)(1:1)//chsp(idq2)(1:1)
              len=len+2
	    elseif(idq2.ge.7 .and. idq1.lt.7) then
              chau(len+1:len+3)=chsp(idq1)(1:1)//chsp(idq2)(1:2)
              len=len+3
	    else 
              chau(len+1:len+4)=chsp(idq1)(1:2)//chsp(idq2)(1:2)
              len=len+4
	    endif
          elseif(idq1.gt.idq2.and.idq1.ge.4) then
	    if(idq1.lt.7) then
              chau(len+1:len+1)=chsp(idq1)(1:1)
              len=len+1
	    else
              chau(len+1:len+2)=chsp(idq1)(1:2)
              len=len+2
	    endif
          endif
        endif
      endif

C...Add on bar sign for antiparticle (where necessary).
      if(id.gt.0.or.len.eq.0) then
      elseif(ida.gt.10.and.ida.le.60.and.kq.ne.0) then
      elseif(ida.ge.81.and.ida.le.100) then
      elseif(ida.gt.100.and.idq3.eq.0.and.kq.ne.0.and.kqx.eq.0) then
      elseif(ida.gt.100.and.idq3.eq.0.and.kq.ne.0.and.kqx.eq.9) then
      elseif(irt.gt.100.and.idq3.eq.0.and.kq.ne.0.and.kqx.eq.3) then
      elseif(irt.gt.10.and.kq.ne.0.and.kqx.eq.1) then
      elseif(irt.gt.10.and.kq.ne.0.and.kqx.eq.2) then
      elseif(irt.gt.10.and.kq.ne.0.and.kqx.eq.4) then
      elseif(idcm.ge.201 .and. idcm.le.203) then
      else
        chau(len+1:len+1)='~'
        len=len+1
      endif

C...Add on charge where applicable (conventional cases skipped).
      if(ida.ge.81.and.ida.le.100) then
      elseif(idcm.ge.201 .and. idcm.le.209) then
      elseif(idcm.ge.234 .and. idcm.le.236) then
C...generator specific codes
      elseif(idcm.eq.251 .or. idcm.eq.252) then
C...Herwig remnant particles
      elseif(kqn.eq.1) then
C...ions
      elseif(len.ne.0)then
C...everything else
        if(kq.eq.6) chau(len+1:len+2)='++'
        if(kq.eq.-6) chau(len+1:len+2)='--'
        if(kq.eq.3) chau(len+1:len+1)='+'
        if(kq.eq.-3) chau(len+1:len+1)='-'

        if(kq.eq.0) then
C...quarks and leptons
          if(ida.le.22) then
          elseif(ida.eq.39 .or. ida.eq.42) then
C...left squarks, sleptons, etc.
          elseif(kqx.eq.1.and.irt.le.21) then
          elseif(kqx.eq.1.and.irt.eq.39) then
C...right squarks, sleptons, etc.
          elseif(kqx.eq.2.and.irt.le.21) then
C...excited particles
          elseif(kqx.eq.4) then
C...reggeon, pomeron, odderon
          elseif(idcm.ge.107 .and. idcm.le.109) then
C... eta, psi, upsilon, etc.
          elseif(irt.gt.100.and.idq3.eq.0.and.idq2.eq.idq1.and.
     &      idq2.ne.1) then
          else
            chau(len+1:len+1)='0'
          endif
        endif
      endif
      chaup = chau

      return
      end

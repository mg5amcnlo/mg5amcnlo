
      subroutine hepnam(id,chaup)

C...Purpose: to give the particle/parton name as a character string.

C       ID   = particle ID
C       chau = particle name

      integer id
      character chaup*(*)
      character chau*20
      character chqk(9)*2

      data chqk(   1) /'d'/
      data chqk(   2) /'u'/
      data chqk(   3) /'s'/
      data chqk(   4) /'c'/
      data chqk(   5) /'b'/
      data chqk(   6) /'t'/
      data chqk(   7) /'b'''/
      data chqk(   8) /'t'''/
      data chqk(   9) /'g'/

      integer hepchg,hepcmp
      external hepchg,hepcmp

C...Initial values. Charge (in units of 1/3). Subdivide code.
      chau=' '
      chaup = chau
      ida=iabs(id)
      if(ida.eq.0) return

      idcm=hepcmp(id)
      if(idcm.le.0) return

      len=0
      kq=hepchg(id)
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
        call chsplook(idcm,chau)
        len=0
        do 100 lem=1,20
  100     if(chau(lem:lem).ne.' ') len=lem
      elseif(kqn.eq.1) then
        call chsplook(idcm,chau)
        len=0
        do 110 lem=1,20
  110     if(chau(lem:lem).ne.' ') len=lem
      elseif(idjs.eq.0) then
        call chsplook(idcm,chau)
        len=0
        do 150 lem=1,20
  150     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name for SUSY, technicolor, and excited particles
      elseif(kqx.ge.1 .and. kqx.le.5) then
        call chsplook(idcm,chau)
        len=0
        do 250 lem=1,20
  250     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name for miscellaneous particles
      elseif(k99.eq.99) then
        call chsplook(idcm,chau)
        len=0
        do 260 lem=1,20
  260     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name for diquark. Add on spin.
      elseif(idq1.eq.0) then
        if(idq3.lt.7) then
          chau(1:2)=chqk(idq3)(1:1)//chqk(idq2)(1:1)
          if(idjs.eq.1) chau(3:4)='_0'
          if(idjs.eq.3) chau(3:4)='_1'
          len=4
	elseif(idq3.ge.7 .and. idq2.lt.7) then
          chau(1:3)=chqk(idq3)(1:2)//chqk(idq2)(1:1)
          if(idjs.eq.1) chau(4:5)='_0'
          if(idjs.eq.3) chau(4:5)='_1'
          len=5
	elseif(idq3.ge.7 .and. idq2.ge.7) then
          chau(1:4)=chqk(idq3)(1:2)//chqk(idq2)(1:2)
          if(idjs.eq.1) chau(5:6)='_0'
          if(idjs.eq.3) chau(5:6)='_1'
          len=6
	endif

C...Construct root name for meson.
      elseif(idq3.eq.0) then
        call chsplook(idcm,chau)
        len=0
        do 200 lem=1,20
  200     if(chau(lem:lem).ne.' ') len=lem

C...Construct root name and spin for baryon.
      else
        if(idq3.le.6) then
          call chsplook(idcm,chau)
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
          chau(len+1:len+3)='_'//chqk(idq3)(1:2)
          len=len+3
          if(idq2.ge.idq1.and.idq1.ge.4) then
	    if(idq2.lt.7 .and. idq1.lt.7) then
              chau(len+1:len+2)=chqk(idq2)(1:1)//chqk(idq1)(1:1)
              len=len+2
	    elseif(idq2.ge.7 .and. idq1.lt.7) then
              chau(len+1:len+3)=chqk(idq2)(1:2)//chqk(idq1)(1:1)
              len=len+3
	    else 
              chau(len+1:len+4)=chqk(idq2)(1:2)//chqk(idq1)(1:2)
              len=len+4
	    endif
          elseif(idq2.ge.idq1.and.idq2.ge.4) then
	    if(idq2.lt.7) then
              chau(len+1:len+1)=chqk(idq2)(1:1)
              len=len+1
	    else
              chau(len+1:len+2)=chqk(idq2)(1:2)
              len=len+2
	    endif
          elseif(idq1.gt.idq2.and.idq2.ge.4) then
	    if(idq2.lt.7 .and. idq1.lt.7) then
              chau(len+1:len+2)=chqk(idq1)(1:1)//chqk(idq2)(1:1)
              len=len+2
	    elseif(idq2.ge.7 .and. idq1.lt.7) then
              chau(len+1:len+3)=chqk(idq1)(1:1)//chqk(idq2)(1:2)
              len=len+3
	    else 
              chau(len+1:len+4)=chqk(idq1)(1:2)//chqk(idq2)(1:2)
              len=len+4
	    endif
          elseif(idq1.gt.idq2.and.idq1.ge.4) then
	    if(idq1.lt.7) then
              chau(len+1:len+1)=chqk(idq1)(1:1)
              len=len+1
	    else
              chau(len+1:len+2)=chqk(idq1)(1:2)
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

C... deal with R-hadrons
      if( kqx.eq.1 .and. idcm.gt.2210 ) then
C...Construct root name
        chau='R~'
	len = 2
C...Add quarks as we find them
	if(kql.gt.0)then
	  len = len+1
	  chau(len:len)=chqk(kql)
	endif
	if(idq3.gt.0)then
	  if(idq3.eq.7.or.idq3.eq.8)then
            chau(len+1:len+2)=chqk(idq3)(1:2)
            len=len+2
	  else
            chau(len+1:len+1)=chqk(idq3)(1:1)
  	    len = len+1
	  endif
	endif
	if(idq2.gt.0)then
	  if(idq2.eq.7.or.idq2.eq.8)then
            chau(len+1:len+2)=chqk(idq2)(1:2)
            len=len+2
	  else
	    len = len+1
	    chau(len:len)=chqk(idq2)
	  endif
	endif
	if(idq1.gt.0)then
	  if(idq1.eq.7.or.idq1.eq.8)then
            chau(len+1:len+2)=chqk(idq1)(1:2)
            len=len+2
	  else
	    len = len+1
	    chau(len:len)=chqk(idq1)
	  endif
	endif
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
        if(kq.eq.6)then
	  chau(len+1:len+2)='++'
        elseif(kq.eq.-6)then
	  chau(len+1:len+2)='--'
        elseif(kq.eq.3)then
	  chau(len+1:len+1)='+'
        elseif(kq.eq.-3)then
	  chau(len+1:len+1)='-'
        elseif(kq.eq.0) then
C...quarks and leptons
          if(ida.le.22) then
          elseif(ida.eq.39 .or. ida.eq.42) then
C...left squarks, sleptons, etc.
          elseif(kqx.eq.1.and.irt.le.21) then
          elseif(kqx.eq.1.and.irt.eq.39) then
C...R-hadrons
          elseif(kqx.eq.1 .and. irt.gt.100) then
            chau(len+1:len+1)='0'
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


      subroutine chsplook(ic,chaup)

C...Purpose: lookup id in chsp and return a character string.

C       ic   = compressed particle id
C       chaup = particle name

      integer id
      character chaup*(*)
      character chau*20
      character chsp(2210)*16

      data chsp(   1) /'d'/
      data chsp(   2) /'u'/
      data chsp(   3) /'s'/
      data chsp(   4) /'c'/
      data chsp(   5) /'b'/
      data chsp(   6) /'t'/
      data chsp(   7) /'b'''/
      data chsp(   8) /'t'''/
      data chsp(   9) /' '/
      data chsp(  10) /' '/
      data chsp(  11) /'e'/
      data chsp(  12) /'nu_e'/
      data chsp(  13) /'mu'/
      data chsp(  14) /'nu_mu'/
      data chsp(  15) /'tau'/
      data chsp(  16) /'nu_tau'/
      data chsp(  17) /'tau'''/
      data chsp(  18) /'nu_tau'''/
      data chsp(  19) /' '/
      data chsp(  20) /' '/
      data chsp(  21) /'g'/
      data chsp(  22) /'gamma'/
      data chsp(  23) /'Z'/
      data chsp(  24) /'W'/
      data chsp(  25) /'H_1'/
      data chsp(  26) /' '/
      data chsp(  27) /' '/
      data chsp(  28) /' '/
      data chsp(  29) /' '/
      data chsp(  30) /' '/
      data chsp(  31) /' '/
      data chsp(  32) /'Z_2'/
      data chsp(  33) /'Z_3'/
      data chsp(  34) /'W_2'/
      data chsp(  35) /'H_2'/
      data chsp(  36) /'H_3'/
      data chsp(  37) /'H'/
      data chsp(  38) /' '/
      data chsp(  39) /'G'/
      data chsp(  40) /' '/
      data chsp(  41) /'R'/
      data chsp(  42) /'LQc'/
      data chsp(  43) /' '/
      data chsp(  44) /' '/
      data chsp(  45) /' '/
      data chsp(  46) /' '/
      data chsp(  47) /' '/
      data chsp(  48) /' '/
      data chsp(  49) /' '/
      data chsp(  50) /' '/
      data chsp(  51) /'H_L'/
      data chsp(  52) /'H_1'/
      data chsp(  53) /'H_2'/
      data chsp(  54) /'H_2'/
      data chsp(  55) /'H_4'/
      data chsp(  56) /' '/
      data chsp(  57) /' '/
      data chsp(  58) /' '/
      data chsp(  59) /' '/
      data chsp(  60) /' '/
      data chsp(  61) /' '/
      data chsp(  62) /' '/
      data chsp(  63) /' '/
      data chsp(  64) /' '/
      data chsp(  65) /' '/
      data chsp(  66) /' '/
      data chsp(  67) /' '/
      data chsp(  68) /' '/
      data chsp(  69) /' '/
      data chsp(  70) /' '/
      data chsp(  71) /' '/
      data chsp(  72) /' '/
      data chsp(  73) /' '/
      data chsp(  74) /' '/
      data chsp(  75) /' '/
      data chsp(  76) /' '/
      data chsp(  77) /' '/
      data chsp(  78) /' '/
      data chsp(  79) /' '/
      data chsp(  80) /' '/
      data chsp(  81) /'gen. code'/
      data chsp(  82) /'gen. code'/
      data chsp(  83) /'gen. code'/
      data chsp(  84) /'gen. code'/
      data chsp(  85) /'gen. code'/
      data chsp(  86) /'gen. code'/
      data chsp(  87) /'gen. code'/
      data chsp(  88) /'gen. code'/
      data chsp(  89) /'gen. code'/
      data chsp(  90) /'gen. code'/
      data chsp(  91) /'gen. code'/
      data chsp(  92) /'gen. code'/
      data chsp(  93) /'gen. code'/
      data chsp(  94) /'gen. code'/
      data chsp(  95) /'gen. code'/
      data chsp(  96) /'gen. code'/
      data chsp(  97) /'gen. code'/
      data chsp(  98) /'gen. code'/
      data chsp(  99) /'gen. code'/
      data chsp( 100) /'gen. code'/
      data chsp( 101) /'KS'/
      data chsp( 102) /'KL'/
      data chsp( 103) /'diquark'/
      data chsp( 104) /'l-baryon'/
      data chsp( 105) /'h-baryon'/
      data chsp( 106) /' '/
      data chsp( 107) /'reggeon'/
      data chsp( 108) /'pomeron'/
      data chsp( 109) /'odderon'/
      data chsp( 110) /' '/
      data chsp( 111) /'L susy d'/
      data chsp( 112) /'L susy u'/
      data chsp( 113) /'L susy s'/
      data chsp( 114) /'L susy c'/
      data chsp( 115) /'L susy b'/
      data chsp( 116) /'L susy t'/
      data chsp( 117) /' '/
      data chsp( 118) /' '/
      data chsp( 119) /' '/
      data chsp( 120) /' '/
      data chsp( 121) /'L susy e'/
      data chsp( 122) /'L susy nu_e'/
      data chsp( 123) /'L susy mu'/
      data chsp( 124) /'L susy nu_mu'/
      data chsp( 125) /'L susy tau'/
      data chsp( 126) /'L susy nu_tau'/
      data chsp( 127) /' '/
      data chsp( 128) /' '/
      data chsp( 129) /' '/
      data chsp( 130) /' '/
      data chsp( 131) /'gluino'/
      data chsp( 132) /'susy chi_1'/
      data chsp( 133) /'susy chi_2'/
      data chsp( 134) /'susy chi_1'/
      data chsp( 135) /'susy chi_3'/
      data chsp( 136) /' '/
      data chsp( 137) /' '/
      data chsp( 138) /' '/
      data chsp( 139) /' '/
      data chsp( 140) /' '/
      data chsp( 141) /' '/
      data chsp( 142) /' '/
      data chsp( 143) /' '/
      data chsp( 144) /' '/
      data chsp( 145) /'susy chi_4'/
      data chsp( 146) /' '/
      data chsp( 147) /'susy chi_2'/
      data chsp( 148) /' '/
      data chsp( 149) /'gravitino'/
      data chsp( 150) /' '/
      data chsp( 151) /' '/
      data chsp( 152) /' '/
      data chsp( 153) /' '/
      data chsp( 154) /' '/
      data chsp( 155) /' '/
      data chsp( 156) /' '/
      data chsp( 157) /' '/
      data chsp( 158) /' '/
      data chsp( 159) /' '/
      data chsp( 160) /' '/
      data chsp( 161) /' '/
      data chsp( 162) /' '/
      data chsp( 163) /' '/
      data chsp( 164) /' '/
      data chsp( 165) /' '/
      data chsp( 166) /' '/
      data chsp( 167) /' '/
      data chsp( 168) /' '/
      data chsp( 169) /' '/
      data chsp( 170) /' '/
      data chsp( 171) /'R susy d'/
      data chsp( 172) /'R susy u'/
      data chsp( 173) /'R susy s'/
      data chsp( 174) /'R susy c'/
      data chsp( 175) /'R susy b'/
      data chsp( 176) /'R susy t'/
      data chsp( 177) /' '/
      data chsp( 178) /' '/
      data chsp( 179) /' '/
      data chsp( 180) /' '/
      data chsp( 181) /'R susy e'/
      data chsp( 182) /'R susy nu_e'/
      data chsp( 183) /'R susy mu'/
      data chsp( 184) /'R susy nu_mu'/
      data chsp( 185) /'R susy tau'/
      data chsp( 186) /'R susy nu_tau'/
      data chsp( 187) /' '/
      data chsp( 188) /' '/
      data chsp( 189) /' '/
      data chsp( 190) /' '/
      data chsp( 191) /'pi_tech'/
      data chsp( 192) /'pi_tech'/
      data chsp( 193) /'pi''_tech'/
      data chsp( 194) /'rho_tech'/
      data chsp( 195) /'rho_tech'/
      data chsp( 196) /'eta_tech'/
      data chsp( 197) /'omega_tech'/
      data chsp( 198) /' '/
      data chsp( 199) /' '/
      data chsp( 200) /' '/
      data chsp( 201) /'V8_tech'/
      data chsp( 202) /'pi_tech_22_1'/
      data chsp( 203) /'pi_tech_22_8'/
      data chsp( 204) /'rho_tech_11'/
      data chsp( 205) /'rho_tech_12'/
      data chsp( 206) /'rho_tech_21'/
      data chsp( 207) /'rho_tech_22'/
      data chsp( 208) /' '/
      data chsp( 209) /' '/
      data chsp( 210) /' '/
      data chsp( 211) /'d*'/
      data chsp( 212) /'u*'/
      data chsp( 213) /' '/
      data chsp( 214) /' '/
      data chsp( 215) /' '/
      data chsp( 216) /' '/
      data chsp( 217) /' '/
      data chsp( 218) /' '/
      data chsp( 219) /' '/
      data chsp( 220) /' '/
      data chsp( 221) /'e*'/
      data chsp( 222) /'nu*_e'/
      data chsp( 223) /' '/
      data chsp( 224) /' '/
      data chsp( 225) /' '/
      data chsp( 226) /' '/
      data chsp( 227) /' '/
      data chsp( 228) /' '/
      data chsp( 229) /' '/
      data chsp( 230) /'G*'/
      data chsp( 231) /'H_L'/
      data chsp( 232) /'H_R'/
      data chsp( 233) /'W_R'/
      data chsp( 234) /'nu_Re'/
      data chsp( 235) /'nu_Rmu'/
      data chsp( 236) /'nu_Rtau'/
      data chsp( 237) /'Z_R'/
      data chsp( 238) /'Theta'/
      data chsp( 239) /'Phi'/
      data chsp( 240) /' '/
      data chsp( 241) /'rho_diffr'/
      data chsp( 242) /'pi_diffr'/
      data chsp( 243) /'omega_diffr'/
      data chsp( 244) /'phi_diffr'/
      data chsp( 245) /'psi_diffr'/
      data chsp( 246) /'n_diffr'/
      data chsp( 247) /'p_diffr'/
      data chsp( 248) /'ccbar[3S1(8)]'/
      data chsp( 249) /'ccbar[1S0(8)]'/
      data chsp( 250) /'ccbar[3P0(8)]'/
      data chsp( 251) /'remnant photon'/
      data chsp( 252) /'remnant nucleon'/
      data chsp( 253) /'gamma_virt'/
      data chsp( 254) /'Cerenkov'/
      data chsp( 255) /'bbbar[3S1(8)]'/
      data chsp( 256) /'bbbar[1S0(8)]'/
      data chsp( 257) /'bbbar[3P0(8)]'/
      data chsp( 258) /' '/
      data chsp( 259) /' '/
      data chsp( 260) /'pi'/
      data chsp( 261) /'rho(770)'/
      data chsp( 262) /'a_2(1320)'/
      data chsp( 263) /'rho_3(1690)'/
      data chsp( 264) /'a_4(2040)'/
      data chsp( 265) /'a_0(980)'/
      data chsp( 266) /'pi_1(1400)'/
      data chsp( 267) /'a_2(1700)'/
      data chsp( 268) /'rho_3(1900)'/
      data chsp( 269) /' '/
      data chsp( 270) /'a_0(1450)'/
      data chsp( 271) /'b_1(1235)'/
      data chsp( 272) /'pi_2(1670)'/
      data chsp( 273) /' '/
      data chsp( 274) /' '/
      data chsp( 275) /'pi(1800)'/
      data chsp( 276) /'pi_1(1600)'/
      data chsp( 277) /'pi_2(2100)'/
      data chsp( 278) /'rho_3(2250)'/
      data chsp( 279) /' '/
      data chsp( 280) /' '/
      data chsp( 281) /'a_1(1260)'/
      data chsp( 282) /' '/
      data chsp( 283) /' '/
      data chsp( 284) /' '/
      data chsp( 285) /' '/
      data chsp( 286) /'a_1(1640)'/
      data chsp( 287) /' '/
      data chsp( 288) /' '/
      data chsp( 289) /' '/
      data chsp( 290) /' '/
      data chsp( 291) /'rho(1700)'/
      data chsp( 292) /' '/
      data chsp( 293) /' '/
      data chsp( 294) /' '/
      data chsp( 295) /' '/
      data chsp( 296) /'rho_0(1900)'/
      data chsp( 297) /' '/
      data chsp( 298) /' '/
      data chsp( 299) /' '/
      data chsp( 300) /' '/
      data chsp( 301) /' '/
      data chsp( 302) /' '/
      data chsp( 303) /' '/
      data chsp( 304) /' '/
      data chsp( 305) /' '/
      data chsp( 306) /'rho_0(2150)'/
      data chsp( 307) /' '/
      data chsp( 308) /' '/
      data chsp( 309) /' '/
      data chsp( 310) /'pi(1300)'/
      data chsp( 311) /'rho(1450)'/
      data chsp( 312) /' '/
      data chsp( 313) /' '/
      data chsp( 314) /' '/
      data chsp( 315) /' '/
      data chsp( 316) /' '/
      data chsp( 317) /' '/
      data chsp( 318) /' '/
      data chsp( 319) /' '/
      data chsp( 320) /' '/
      data chsp( 321) /' '/
      data chsp( 322) /' '/
      data chsp( 323) /' '/
      data chsp( 324) /' '/
      data chsp( 325) /' '/
      data chsp( 326) /' '/
      data chsp( 327) /' '/
      data chsp( 328) /' '/
      data chsp( 329) /' '/
      data chsp( 330) /' '/
      data chsp( 331) /' '/
      data chsp( 332) /' '/
      data chsp( 333) /' '/
      data chsp( 334) /' '/
      data chsp( 335) /' '/
      data chsp( 336) /' '/
      data chsp( 337) /' '/
      data chsp( 338) /' '/
      data chsp( 339) /' '/
      data chsp( 340) /' '/
      data chsp( 341) /' '/
      data chsp( 342) /' '/
      data chsp( 343) /' '/
      data chsp( 344) /' '/
      data chsp( 345) /' '/
      data chsp( 346) /' '/
      data chsp( 347) /' '/
      data chsp( 348) /' '/
      data chsp( 349) /' '/
      data chsp( 350) /' '/
      data chsp( 351) /' '/
      data chsp( 352) /' '/
      data chsp( 353) /' '/
      data chsp( 354) /' '/
      data chsp( 355) /' '/
      data chsp( 356) /' '/
      data chsp( 357) /' '/
      data chsp( 358) /' '/
      data chsp( 359) /' '/
      data chsp( 360) /' '/
      data chsp( 361) /' '/
      data chsp( 362) /' '/
      data chsp( 363) /' '/
      data chsp( 364) /' '/
      data chsp( 365) /' '/
      data chsp( 366) /' '/
      data chsp( 367) /' '/
      data chsp( 368) /' '/
      data chsp( 369) /' '/
      data chsp( 370) /' '/
      data chsp( 371) /' '/
      data chsp( 372) /' '/
      data chsp( 373) /' '/
      data chsp( 374) /' '/
      data chsp( 375) /' '/
      data chsp( 376) /' '/
      data chsp( 377) /' '/
      data chsp( 378) /' '/
      data chsp( 379) /' '/
      data chsp( 380) /' '/
      data chsp( 381) /' '/
      data chsp( 382) /' '/
      data chsp( 383) /' '/
      data chsp( 384) /' '/
      data chsp( 385) /' '/
      data chsp( 386) /' '/
      data chsp( 387) /' '/
      data chsp( 388) /' '/
      data chsp( 389) /' '/
      data chsp( 390) /' '/
      data chsp( 391) /' '/
      data chsp( 392) /' '/
      data chsp( 393) /' '/
      data chsp( 394) /' '/
      data chsp( 395) /' '/
      data chsp( 396) /' '/
      data chsp( 397) /' '/
      data chsp( 398) /' '/
      data chsp( 399) /' '/
      data chsp( 400) /' '/
      data chsp( 401) /' '/
      data chsp( 402) /' '/
      data chsp( 403) /' '/
      data chsp( 404) /' '/
      data chsp( 405) /' '/
      data chsp( 406) /' '/
      data chsp( 407) /' '/
      data chsp( 408) /' '/
      data chsp( 409) /' '/
      data chsp( 410) /'pi'/
      data chsp( 411) /'rho(770)'/
      data chsp( 412) /'a_2(1320)'/
      data chsp( 413) /'rho_3(1690)'/
      data chsp( 414) /'a_4(2040)'/
      data chsp( 415) /'a_0(980)'/
      data chsp( 416) /'pi_1(1400)'/
      data chsp( 417) /'a_2(1700)'/
      data chsp( 418) /'rho_3(1900)'/
      data chsp( 419) /' '/
      data chsp( 420) /'a_0(1450)'/
      data chsp( 421) /'b_1(1235)'/
      data chsp( 422) /'pi_2(1670)'/
      data chsp( 423) /' '/
      data chsp( 424) /' '/
      data chsp( 425) /'pi(1800)'/
      data chsp( 426) /'pi_1(1600)'/
      data chsp( 427) /'pi_2(2100)'/
      data chsp( 428) /'rho_3(2250)'/
      data chsp( 429) /' '/
      data chsp( 430) /' '/
      data chsp( 431) /'a_1(1260)'/
      data chsp( 432) /' '/
      data chsp( 433) /' '/
      data chsp( 434) /' '/
      data chsp( 435) /' '/
      data chsp( 436) /'a_1(1640)'/
      data chsp( 437) /' '/
      data chsp( 438) /' '/
      data chsp( 439) /' '/
      data chsp( 440) /' '/
      data chsp( 441) /'rho(1700)'/
      data chsp( 442) /' '/
      data chsp( 443) /' '/
      data chsp( 444) /' '/
      data chsp( 445) /' '/
      data chsp( 446) /'rho_0(1900)'/
      data chsp( 447) /' '/
      data chsp( 448) /' '/
      data chsp( 449) /' '/
      data chsp( 450) /' '/
      data chsp( 451) /' '/
      data chsp( 452) /' '/
      data chsp( 453) /' '/
      data chsp( 454) /' '/
      data chsp( 455) /' '/
      data chsp( 456) /'rho_0(2150)'/
      data chsp( 457) /' '/
      data chsp( 458) /' '/
      data chsp( 459) /' '/
      data chsp( 460) /'pi(1300)'/
      data chsp( 461) /'rho(1450)'/
      data chsp( 462) /' '/
      data chsp( 463) /' '/
      data chsp( 464) /' '/
      data chsp( 465) /' '/
      data chsp( 466) /' '/
      data chsp( 467) /' '/
      data chsp( 468) /' '/
      data chsp( 469) /' '/
      data chsp( 470) /' '/
      data chsp( 471) /' '/
      data chsp( 472) /' '/
      data chsp( 473) /' '/
      data chsp( 474) /' '/
      data chsp( 475) /' '/
      data chsp( 476) /' '/
      data chsp( 477) /' '/
      data chsp( 478) /' '/
      data chsp( 479) /' '/
      data chsp( 480) /' '/
      data chsp( 481) /' '/
      data chsp( 482) /' '/
      data chsp( 483) /' '/
      data chsp( 484) /' '/
      data chsp( 485) /' '/
      data chsp( 486) /' '/
      data chsp( 487) /' '/
      data chsp( 488) /' '/
      data chsp( 489) /' '/
      data chsp( 490) /' '/
      data chsp( 491) /' '/
      data chsp( 492) /' '/
      data chsp( 493) /' '/
      data chsp( 494) /' '/
      data chsp( 495) /' '/
      data chsp( 496) /' '/
      data chsp( 497) /' '/
      data chsp( 498) /' '/
      data chsp( 499) /' '/
      data chsp( 500) /' '/
      data chsp( 501) /' '/
      data chsp( 502) /' '/
      data chsp( 503) /' '/
      data chsp( 504) /' '/
      data chsp( 505) /' '/
      data chsp( 506) /' '/
      data chsp( 507) /' '/
      data chsp( 508) /' '/
      data chsp( 509) /' '/
      data chsp( 510) /' '/
      data chsp( 511) /' '/
      data chsp( 512) /' '/
      data chsp( 513) /' '/
      data chsp( 514) /' '/
      data chsp( 515) /' '/
      data chsp( 516) /' '/
      data chsp( 517) /' '/
      data chsp( 518) /' '/
      data chsp( 519) /' '/
      data chsp( 520) /' '/
      data chsp( 521) /' '/
      data chsp( 522) /' '/
      data chsp( 523) /' '/
      data chsp( 524) /' '/
      data chsp( 525) /' '/
      data chsp( 526) /' '/
      data chsp( 527) /' '/
      data chsp( 528) /' '/
      data chsp( 529) /' '/
      data chsp( 530) /' '/
      data chsp( 531) /' '/
      data chsp( 532) /' '/
      data chsp( 533) /' '/
      data chsp( 534) /' '/
      data chsp( 535) /' '/
      data chsp( 536) /' '/
      data chsp( 537) /' '/
      data chsp( 538) /' '/
      data chsp( 539) /' '/
      data chsp( 540) /' '/
      data chsp( 541) /' '/
      data chsp( 542) /' '/
      data chsp( 543) /' '/
      data chsp( 544) /' '/
      data chsp( 545) /' '/
      data chsp( 546) /' '/
      data chsp( 547) /' '/
      data chsp( 548) /' '/
      data chsp( 549) /' '/
      data chsp( 550) /' '/
      data chsp( 551) /' '/
      data chsp( 552) /' '/
      data chsp( 553) /' '/
      data chsp( 554) /' '/
      data chsp( 555) /' '/
      data chsp( 556) /' '/
      data chsp( 557) /' '/
      data chsp( 558) /' '/
      data chsp( 559) /' '/
      data chsp( 560) /'eta'/
      data chsp( 561) /'omega(782)'/
      data chsp( 562) /'f_2(1270)'/
      data chsp( 563) /'omega_3(1670)'/
      data chsp( 564) /'f_4(2050)'/
      data chsp( 565) /'f_0(600)'/
      data chsp( 566) /'f_1(1510)'/
      data chsp( 567) /'f_2(1430)'/
      data chsp( 568) /' '/
      data chsp( 569) /'f_J(2220)'/
      data chsp( 570) /'f_0(1370)'/
      data chsp( 571) /'h_1(1170)'/
      data chsp( 572) /'eta_2(1645)'/
      data chsp( 573) /' '/
      data chsp( 574) /' '/
      data chsp( 575) /'f_0(980)'/
      data chsp( 576) /'h_1(1595)'/
      data chsp( 577) /'f_2(1565)'/
      data chsp( 578) /' '/
      data chsp( 579) /'f_4(2300)'/
      data chsp( 580) /' '/
      data chsp( 581) /'f_1(1285)'/
      data chsp( 582) /' '/
      data chsp( 583) /' '/
      data chsp( 584) /' '/
      data chsp( 585) /'eta(1405)'/
      data chsp( 586) /' '/
      data chsp( 587) /'f_2(1640)'/
      data chsp( 588) /' '/
      data chsp( 589) /' '/
      data chsp( 590) /' '/
      data chsp( 591) /'omega(1650)'/
      data chsp( 592) /' '/
      data chsp( 593) /' '/
      data chsp( 594) /' '/
      data chsp( 595) /'f_0(1500)'/
      data chsp( 596) /' '/
      data chsp( 597) /'f_2(1810)'/
      data chsp( 598) /' '/
      data chsp( 599) /' '/
      data chsp( 600) /' '/
      data chsp( 601) /' '/
      data chsp( 602) /' '/
      data chsp( 603) /' '/
      data chsp( 604) /' '/
      data chsp( 605) /'eta(1760)'/
      data chsp( 606) /' '/
      data chsp( 607) /'f_2(1910)'/
      data chsp( 608) /' '/
      data chsp( 609) /' '/
      data chsp( 610) /' '/
      data chsp( 611) /' '/
      data chsp( 612) /' '/
      data chsp( 613) /' '/
      data chsp( 614) /' '/
      data chsp( 615) /'f_0(2020)'/
      data chsp( 616) /' '/
      data chsp( 617) /'f_2(1950)'/
      data chsp( 618) /' '/
      data chsp( 619) /' '/
      data chsp( 620) /' '/
      data chsp( 621) /' '/
      data chsp( 622) /' '/
      data chsp( 623) /' '/
      data chsp( 624) /' '/
      data chsp( 625) /'f_0(2100)'/
      data chsp( 626) /' '/
      data chsp( 627) /'f_2(2010)'/
      data chsp( 628) /' '/
      data chsp( 629) /' '/
      data chsp( 630) /' '/
      data chsp( 631) /' '/
      data chsp( 632) /' '/
      data chsp( 633) /' '/
      data chsp( 634) /' '/
      data chsp( 635) /'f_0(2200)'/
      data chsp( 636) /' '/
      data chsp( 637) /'f_2(2150)'/
      data chsp( 638) /' '/
      data chsp( 639) /' '/
      data chsp( 640) /' '/
      data chsp( 641) /' '/
      data chsp( 642) /' '/
      data chsp( 643) /' '/
      data chsp( 644) /' '/
      data chsp( 645) /'eta(2225)'/
      data chsp( 646) /' '/
      data chsp( 647) /'f_2(2300)'/
      data chsp( 648) /' '/
      data chsp( 649) /' '/
      data chsp( 650) /' '/
      data chsp( 651) /' '/
      data chsp( 652) /' '/
      data chsp( 653) /' '/
      data chsp( 654) /' '/
      data chsp( 655) /' '/
      data chsp( 656) /' '/
      data chsp( 657) /'f_2(2340)'/
      data chsp( 658) /' '/
      data chsp( 659) /' '/
      data chsp( 660) /'eta(1295)'/
      data chsp( 661) /'omega(1420)'/
      data chsp( 662) /' '/
      data chsp( 663) /' '/
      data chsp( 664) /' '/
      data chsp( 665) /' '/
      data chsp( 666) /' '/
      data chsp( 667) /' '/
      data chsp( 668) /' '/
      data chsp( 669) /' '/
      data chsp( 670) /' '/
      data chsp( 671) /' '/
      data chsp( 672) /' '/
      data chsp( 673) /' '/
      data chsp( 674) /' '/
      data chsp( 675) /' '/
      data chsp( 676) /' '/
      data chsp( 677) /' '/
      data chsp( 678) /' '/
      data chsp( 679) /' '/
      data chsp( 680) /' '/
      data chsp( 681) /' '/
      data chsp( 682) /' '/
      data chsp( 683) /' '/
      data chsp( 684) /' '/
      data chsp( 685) /' '/
      data chsp( 686) /' '/
      data chsp( 687) /' '/
      data chsp( 688) /' '/
      data chsp( 689) /' '/
      data chsp( 690) /' '/
      data chsp( 691) /' '/
      data chsp( 692) /' '/
      data chsp( 693) /' '/
      data chsp( 694) /' '/
      data chsp( 695) /' '/
      data chsp( 696) /' '/
      data chsp( 697) /' '/
      data chsp( 698) /' '/
      data chsp( 699) /' '/
      data chsp( 700) /' '/
      data chsp( 701) /' '/
      data chsp( 702) /' '/
      data chsp( 703) /' '/
      data chsp( 704) /' '/
      data chsp( 705) /' '/
      data chsp( 706) /' '/
      data chsp( 707) /' '/
      data chsp( 708) /' '/
      data chsp( 709) /' '/
      data chsp( 710) /' '/
      data chsp( 711) /' '/
      data chsp( 712) /' '/
      data chsp( 713) /' '/
      data chsp( 714) /' '/
      data chsp( 715) /' '/
      data chsp( 716) /' '/
      data chsp( 717) /' '/
      data chsp( 718) /' '/
      data chsp( 719) /' '/
      data chsp( 720) /' '/
      data chsp( 721) /' '/
      data chsp( 722) /' '/
      data chsp( 723) /' '/
      data chsp( 724) /' '/
      data chsp( 725) /' '/
      data chsp( 726) /' '/
      data chsp( 727) /' '/
      data chsp( 728) /' '/
      data chsp( 729) /' '/
      data chsp( 730) /' '/
      data chsp( 731) /' '/
      data chsp( 732) /' '/
      data chsp( 733) /' '/
      data chsp( 734) /' '/
      data chsp( 735) /' '/
      data chsp( 736) /' '/
      data chsp( 737) /' '/
      data chsp( 738) /' '/
      data chsp( 739) /' '/
      data chsp( 740) /' '/
      data chsp( 741) /' '/
      data chsp( 742) /' '/
      data chsp( 743) /' '/
      data chsp( 744) /' '/
      data chsp( 745) /' '/
      data chsp( 746) /' '/
      data chsp( 747) /' '/
      data chsp( 748) /' '/
      data chsp( 749) /' '/
      data chsp( 750) /' '/
      data chsp( 751) /' '/
      data chsp( 752) /' '/
      data chsp( 753) /' '/
      data chsp( 754) /' '/
      data chsp( 755) /' '/
      data chsp( 756) /' '/
      data chsp( 757) /' '/
      data chsp( 758) /' '/
      data chsp( 759) /' '/
      data chsp( 760) /' '/
      data chsp( 761) /' '/
      data chsp( 762) /' '/
      data chsp( 763) /' '/
      data chsp( 764) /' '/
      data chsp( 765) /' '/
      data chsp( 766) /' '/
      data chsp( 767) /' '/
      data chsp( 768) /' '/
      data chsp( 769) /' '/
      data chsp( 770) /' '/
      data chsp( 771) /' '/
      data chsp( 772) /' '/
      data chsp( 773) /' '/
      data chsp( 774) /' '/
      data chsp( 775) /' '/
      data chsp( 776) /' '/
      data chsp( 777) /' '/
      data chsp( 778) /' '/
      data chsp( 779) /' '/
      data chsp( 780) /' '/
      data chsp( 781) /' '/
      data chsp( 782) /' '/
      data chsp( 783) /' '/
      data chsp( 784) /' '/
      data chsp( 785) /' '/
      data chsp( 786) /' '/
      data chsp( 787) /' '/
      data chsp( 788) /' '/
      data chsp( 789) /' '/
      data chsp( 790) /' '/
      data chsp( 791) /' '/
      data chsp( 792) /' '/
      data chsp( 793) /' '/
      data chsp( 794) /' '/
      data chsp( 795) /' '/
      data chsp( 796) /' '/
      data chsp( 797) /' '/
      data chsp( 798) /' '/
      data chsp( 799) /' '/
      data chsp( 800) /' '/
      data chsp( 801) /' '/
      data chsp( 802) /' '/
      data chsp( 803) /' '/
      data chsp( 804) /' '/
      data chsp( 805) /' '/
      data chsp( 806) /' '/
      data chsp( 807) /' '/
      data chsp( 808) /' '/
      data chsp( 809) /' '/
      data chsp( 810) /' '/
      data chsp( 811) /' '/
      data chsp( 812) /' '/
      data chsp( 813) /' '/
      data chsp( 814) /' '/
      data chsp( 815) /' '/
      data chsp( 816) /' '/
      data chsp( 817) /' '/
      data chsp( 818) /' '/
      data chsp( 819) /' '/
      data chsp( 820) /' '/
      data chsp( 821) /' '/
      data chsp( 822) /' '/
      data chsp( 823) /' '/
      data chsp( 824) /' '/
      data chsp( 825) /' '/
      data chsp( 826) /' '/
      data chsp( 827) /' '/
      data chsp( 828) /' '/
      data chsp( 829) /' '/
      data chsp( 830) /' '/
      data chsp( 831) /' '/
      data chsp( 832) /' '/
      data chsp( 833) /' '/
      data chsp( 834) /' '/
      data chsp( 835) /' '/
      data chsp( 836) /' '/
      data chsp( 837) /' '/
      data chsp( 838) /' '/
      data chsp( 839) /' '/
      data chsp( 840) /' '/
      data chsp( 841) /' '/
      data chsp( 842) /' '/
      data chsp( 843) /' '/
      data chsp( 844) /' '/
      data chsp( 845) /' '/
      data chsp( 846) /' '/
      data chsp( 847) /' '/
      data chsp( 848) /' '/
      data chsp( 849) /' '/
      data chsp( 850) /' '/
      data chsp( 851) /' '/
      data chsp( 852) /' '/
      data chsp( 853) /' '/
      data chsp( 854) /' '/
      data chsp( 855) /' '/
      data chsp( 856) /' '/
      data chsp( 857) /' '/
      data chsp( 858) /' '/
      data chsp( 859) /' '/
      data chsp( 860) /'eta''(958)'/
      data chsp( 861) /'phi(1020)'/
      data chsp( 862) /'f''_2(1525)'/
      data chsp( 863) /'phi_3(1850)'/
      data chsp( 864) /' '/
      data chsp( 865) /' '/
      data chsp( 866) /' '/
      data chsp( 867) /' '/
      data chsp( 868) /' '/
      data chsp( 869) /' '/
      data chsp( 870) /'f_0(1710)'/
      data chsp( 871) /'h_1(1380)'/
      data chsp( 872) /'eta_2(1870)'/
      data chsp( 873) /' '/
      data chsp( 874) /' '/
      data chsp( 875) /' '/
      data chsp( 876) /' '/
      data chsp( 877) /' '/
      data chsp( 878) /' '/
      data chsp( 879) /' '/
      data chsp( 880) /' '/
      data chsp( 881) /'f_1(1420)'/
      data chsp( 882) /' '/
      data chsp( 883) /' '/
      data chsp( 884) /' '/
      data chsp( 885) /' '/
      data chsp( 886) /' '/
      data chsp( 887) /' '/
      data chsp( 888) /' '/
      data chsp( 889) /' '/
      data chsp( 890) /'eta(1475)'/
      data chsp( 891) /'phi(1680)'/
      data chsp( 892) /' '/
      data chsp( 893) /' '/
      data chsp( 894) /' '/
      data chsp( 895) /' '/
      data chsp( 896) /' '/
      data chsp( 897) /' '/
      data chsp( 898) /' '/
      data chsp( 899) /' '/
      data chsp( 900) /' '/
      data chsp( 901) /' '/
      data chsp( 902) /' '/
      data chsp( 903) /' '/
      data chsp( 904) /' '/
      data chsp( 905) /' '/
      data chsp( 906) /' '/
      data chsp( 907) /' '/
      data chsp( 908) /' '/
      data chsp( 909) /' '/
      data chsp( 910) /' '/
      data chsp( 911) /' '/
      data chsp( 912) /' '/
      data chsp( 913) /' '/
      data chsp( 914) /' '/
      data chsp( 915) /' '/
      data chsp( 916) /' '/
      data chsp( 917) /' '/
      data chsp( 918) /' '/
      data chsp( 919) /' '/
      data chsp( 920) /'K'/
      data chsp( 921) /'K*(892)'/
      data chsp( 922) /'K*_2(1430)'/
      data chsp( 923) /'K*_3(1780)'/
      data chsp( 924) /'K*_4(2045)'/
      data chsp( 925) /'K*_0(800)'/
      data chsp( 926) /'K_1(1650)'/
      data chsp( 927) /'K_2(1580)'/
      data chsp( 928) /' '/
      data chsp( 929) /'K_4(2500)'/
      data chsp( 930) /'K*_0(1430)'/
      data chsp( 931) /'K_1(1270)'/
      data chsp( 932) /'K_2(1770)'/
      data chsp( 933) /' '/
      data chsp( 934) /' '/
      data chsp( 935) /'K(1830)'/
      data chsp( 936) /' '/
      data chsp( 937) /'K*_2(1980)'/
      data chsp( 938) /'K_3(2320)'/
      data chsp( 939) /' '/
      data chsp( 940) /' '/
      data chsp( 941) /'K_1(1400)'/
      data chsp( 942) /'K_2(1820)'/
      data chsp( 943) /' '/
      data chsp( 944) /' '/
      data chsp( 945) /'K*_0(1950)'/
      data chsp( 946) /' '/
      data chsp( 947) /'K_2(2250)'/
      data chsp( 948) /' '/
      data chsp( 949) /' '/
      data chsp( 950) /' '/
      data chsp( 951) /'K*(1680)'/
      data chsp( 952) /' '/
      data chsp( 953) /' '/
      data chsp( 954) /' '/
      data chsp( 955) /' '/
      data chsp( 956) /' '/
      data chsp( 957) /' '/
      data chsp( 958) /' '/
      data chsp( 959) /' '/
      data chsp( 960) /'K(1460)'/
      data chsp( 961) /'K*(1410)'/
      data chsp( 962) /' '/
      data chsp( 963) /' '/
      data chsp( 964) /' '/
      data chsp( 965) /' '/
      data chsp( 966) /' '/
      data chsp( 967) /' '/
      data chsp( 968) /' '/
      data chsp( 969) /' '/
      data chsp( 970) /' '/
      data chsp( 971) /' '/
      data chsp( 972) /' '/
      data chsp( 973) /' '/
      data chsp( 974) /' '/
      data chsp( 975) /' '/
      data chsp( 976) /' '/
      data chsp( 977) /' '/
      data chsp( 978) /' '/
      data chsp( 979) /' '/
      data chsp( 980) /' '/
      data chsp( 981) /' '/
      data chsp( 982) /' '/
      data chsp( 983) /' '/
      data chsp( 984) /' '/
      data chsp( 985) /' '/
      data chsp( 986) /' '/
      data chsp( 987) /' '/
      data chsp( 988) /' '/
      data chsp( 989) /' '/
      data chsp( 990) /' '/
      data chsp( 991) /' '/
      data chsp( 992) /' '/
      data chsp( 993) /' '/
      data chsp( 994) /' '/
      data chsp( 995) /' '/
      data chsp( 996) /' '/
      data chsp( 997) /' '/
      data chsp( 998) /' '/
      data chsp( 999) /' '/
      data chsp(1000) /' '/
      data chsp(1001) /' '/
      data chsp(1002) /' '/
      data chsp(1003) /' '/
      data chsp(1004) /' '/
      data chsp(1005) /' '/
      data chsp(1006) /' '/
      data chsp(1007) /' '/
      data chsp(1008) /' '/
      data chsp(1009) /' '/
      data chsp(1010) /' '/
      data chsp(1011) /' '/
      data chsp(1012) /' '/
      data chsp(1013) /' '/
      data chsp(1014) /' '/
      data chsp(1015) /' '/
      data chsp(1016) /' '/
      data chsp(1017) /' '/
      data chsp(1018) /' '/
      data chsp(1019) /' '/
      data chsp(1020) /' '/
      data chsp(1021) /' '/
      data chsp(1022) /' '/
      data chsp(1023) /' '/
      data chsp(1024) /' '/
      data chsp(1025) /' '/
      data chsp(1026) /' '/
      data chsp(1027) /' '/
      data chsp(1028) /' '/
      data chsp(1029) /' '/
      data chsp(1030) /' '/
      data chsp(1031) /' '/
      data chsp(1032) /' '/
      data chsp(1033) /' '/
      data chsp(1034) /' '/
      data chsp(1035) /' '/
      data chsp(1036) /' '/
      data chsp(1037) /' '/
      data chsp(1038) /' '/
      data chsp(1039) /' '/
      data chsp(1040) /'K'/
      data chsp(1041) /'K*(892)'/
      data chsp(1042) /'K*_2(1430)'/
      data chsp(1043) /'K*_3(1780)'/
      data chsp(1044) /'K*_4(2045)'/
      data chsp(1045) /'K*_0(800)'/
      data chsp(1046) /'K_1(1650)'/
      data chsp(1047) /'K_2(1580)'/
      data chsp(1048) /' '/
      data chsp(1049) /'K_4(2500)'/
      data chsp(1050) /'K*_0(1430)'/
      data chsp(1051) /'K_1(1270)'/
      data chsp(1052) /'K_2(1770)'/
      data chsp(1053) /' '/
      data chsp(1054) /' '/
      data chsp(1055) /'K(1830)'/
      data chsp(1056) /' '/
      data chsp(1057) /'K*_2(1980)'/
      data chsp(1058) /'K_3(2320)'/
      data chsp(1059) /' '/
      data chsp(1060) /' '/
      data chsp(1061) /'K_1(1400)'/
      data chsp(1062) /'K_2(1820)'/
      data chsp(1063) /' '/
      data chsp(1064) /' '/
      data chsp(1065) /'K*_0(1950)'/
      data chsp(1066) /' '/
      data chsp(1067) /'K_2(2250)'/
      data chsp(1068) /' '/
      data chsp(1069) /' '/
      data chsp(1070) /' '/
      data chsp(1071) /'K*(1680)'/
      data chsp(1072) /' '/
      data chsp(1073) /' '/
      data chsp(1074) /' '/
      data chsp(1075) /' '/
      data chsp(1076) /' '/
      data chsp(1077) /' '/
      data chsp(1078) /' '/
      data chsp(1079) /' '/
      data chsp(1080) /'K(1460)'/
      data chsp(1081) /'K*(1410)'/
      data chsp(1082) /'K_2(1980)'/
      data chsp(1083) /' '/
      data chsp(1084) /' '/
      data chsp(1085) /' '/
      data chsp(1086) /' '/
      data chsp(1087) /' '/
      data chsp(1088) /' '/
      data chsp(1089) /' '/
      data chsp(1090) /' '/
      data chsp(1091) /' '/
      data chsp(1092) /' '/
      data chsp(1093) /' '/
      data chsp(1094) /' '/
      data chsp(1095) /' '/
      data chsp(1096) /' '/
      data chsp(1097) /' '/
      data chsp(1098) /' '/
      data chsp(1099) /' '/
      data chsp(1100) /' '/
      data chsp(1101) /' '/
      data chsp(1102) /' '/
      data chsp(1103) /' '/
      data chsp(1104) /' '/
      data chsp(1105) /' '/
      data chsp(1106) /' '/
      data chsp(1107) /' '/
      data chsp(1108) /' '/
      data chsp(1109) /' '/
      data chsp(1110) /' '/
      data chsp(1111) /' '/
      data chsp(1112) /' '/
      data chsp(1113) /' '/
      data chsp(1114) /' '/
      data chsp(1115) /' '/
      data chsp(1116) /' '/
      data chsp(1117) /' '/
      data chsp(1118) /' '/
      data chsp(1119) /' '/
      data chsp(1120) /' '/
      data chsp(1121) /' '/
      data chsp(1122) /' '/
      data chsp(1123) /' '/
      data chsp(1124) /' '/
      data chsp(1125) /' '/
      data chsp(1126) /' '/
      data chsp(1127) /' '/
      data chsp(1128) /' '/
      data chsp(1129) /' '/
      data chsp(1130) /' '/
      data chsp(1131) /' '/
      data chsp(1132) /' '/
      data chsp(1133) /' '/
      data chsp(1134) /' '/
      data chsp(1135) /' '/
      data chsp(1136) /' '/
      data chsp(1137) /' '/
      data chsp(1138) /' '/
      data chsp(1139) /' '/
      data chsp(1140) /' '/
      data chsp(1141) /' '/
      data chsp(1142) /' '/
      data chsp(1143) /' '/
      data chsp(1144) /' '/
      data chsp(1145) /' '/
      data chsp(1146) /' '/
      data chsp(1147) /' '/
      data chsp(1148) /' '/
      data chsp(1149) /' '/
      data chsp(1150) /' '/
      data chsp(1151) /' '/
      data chsp(1152) /' '/
      data chsp(1153) /' '/
      data chsp(1154) /' '/
      data chsp(1155) /' '/
      data chsp(1156) /' '/
      data chsp(1157) /' '/
      data chsp(1158) /' '/
      data chsp(1159) /' '/
      data chsp(1160) /'D'/
      data chsp(1161) /'D*(2010)'/
      data chsp(1162) /'D*_2(2460)'/
      data chsp(1163) /' '/
      data chsp(1164) /' '/
      data chsp(1165) /' '/
      data chsp(1166) /' '/
      data chsp(1167) /' '/
      data chsp(1168) /' '/
      data chsp(1169) /' '/
      data chsp(1170) /'D*_0'/
      data chsp(1171) /'D_1(2420)'/
      data chsp(1172) /' '/
      data chsp(1173) /' '/
      data chsp(1174) /' '/
      data chsp(1175) /' '/
      data chsp(1176) /' '/
      data chsp(1177) /' '/
      data chsp(1178) /' '/
      data chsp(1179) /' '/
      data chsp(1180) /' '/
      data chsp(1181) /'D_1(H)'/
      data chsp(1182) /' '/
      data chsp(1183) /' '/
      data chsp(1184) /' '/
      data chsp(1185) /' '/
      data chsp(1186) /' '/
      data chsp(1187) /' '/
      data chsp(1188) /' '/
      data chsp(1189) /' '/
      data chsp(1190) /'D'/
      data chsp(1191) /'D*(2010)'/
      data chsp(1192) /'D*_2(2460)'/
      data chsp(1193) /' '/
      data chsp(1194) /' '/
      data chsp(1195) /' '/
      data chsp(1196) /' '/
      data chsp(1197) /' '/
      data chsp(1198) /' '/
      data chsp(1199) /' '/
      data chsp(1200) /'D*_0'/
      data chsp(1201) /'D_1(2420)'/
      data chsp(1202) /' '/
      data chsp(1203) /' '/
      data chsp(1204) /' '/
      data chsp(1205) /' '/
      data chsp(1206) /' '/
      data chsp(1207) /' '/
      data chsp(1208) /' '/
      data chsp(1209) /' '/
      data chsp(1210) /' '/
      data chsp(1211) /'D_1(H)'/
      data chsp(1212) /' '/
      data chsp(1213) /' '/
      data chsp(1214) /' '/
      data chsp(1215) /' '/
      data chsp(1216) /' '/
      data chsp(1217) /' '/
      data chsp(1218) /' '/
      data chsp(1219) /' '/
      data chsp(1220) /'D_s'/
      data chsp(1221) /'D*_s'/
      data chsp(1222) /'D*_s2'/
      data chsp(1223) /' '/
      data chsp(1224) /' '/
      data chsp(1225) /' '/
      data chsp(1226) /' '/
      data chsp(1227) /' '/
      data chsp(1228) /' '/
      data chsp(1229) /' '/
      data chsp(1230) /'D*_s0'/
      data chsp(1231) /'D_s1(2536)'/
      data chsp(1232) /' '/
      data chsp(1233) /' '/
      data chsp(1234) /' '/
      data chsp(1235) /' '/
      data chsp(1236) /' '/
      data chsp(1237) /' '/
      data chsp(1238) /' '/
      data chsp(1239) /' '/
      data chsp(1240) /' '/
      data chsp(1241) /'D_s1(H)'/
      data chsp(1242) /' '/
      data chsp(1243) /' '/
      data chsp(1244) /' '/
      data chsp(1245) /' '/
      data chsp(1246) /' '/
      data chsp(1247) /' '/
      data chsp(1248) /' '/
      data chsp(1249) /' '/
      data chsp(1250) /'D(2S)'/
      data chsp(1251) /'D*(2S)'/
      data chsp(1252) /' '/
      data chsp(1253) /' '/
      data chsp(1254) /' '/
      data chsp(1255) /' '/
      data chsp(1256) /' '/
      data chsp(1257) /' '/
      data chsp(1258) /' '/
      data chsp(1259) /' '/
      data chsp(1260) /' '/
      data chsp(1261) /' '/
      data chsp(1262) /' '/
      data chsp(1263) /' '/
      data chsp(1264) /' '/
      data chsp(1265) /' '/
      data chsp(1266) /' '/
      data chsp(1267) /' '/
      data chsp(1268) /' '/
      data chsp(1269) /' '/
      data chsp(1270) /' '/
      data chsp(1271) /' '/
      data chsp(1272) /' '/
      data chsp(1273) /' '/
      data chsp(1274) /' '/
      data chsp(1275) /' '/
      data chsp(1276) /' '/
      data chsp(1277) /' '/
      data chsp(1278) /' '/
      data chsp(1279) /' '/
      data chsp(1280) /'D(2S)'/
      data chsp(1281) /'D*(2S)'/
      data chsp(1282) /' '/
      data chsp(1283) /' '/
      data chsp(1284) /' '/
      data chsp(1285) /' '/
      data chsp(1286) /' '/
      data chsp(1287) /' '/
      data chsp(1288) /' '/
      data chsp(1289) /' '/
      data chsp(1290) /' '/
      data chsp(1291) /' '/
      data chsp(1292) /' '/
      data chsp(1293) /' '/
      data chsp(1294) /' '/
      data chsp(1295) /' '/
      data chsp(1296) /' '/
      data chsp(1297) /' '/
      data chsp(1298) /' '/
      data chsp(1299) /' '/
      data chsp(1300) /' '/
      data chsp(1301) /' '/
      data chsp(1302) /' '/
      data chsp(1303) /' '/
      data chsp(1304) /' '/
      data chsp(1305) /' '/
      data chsp(1306) /' '/
      data chsp(1307) /' '/
      data chsp(1308) /' '/
      data chsp(1309) /' '/
      data chsp(1310) /' '/
      data chsp(1311) /' '/
      data chsp(1312) /' '/
      data chsp(1313) /' '/
      data chsp(1314) /' '/
      data chsp(1315) /' '/
      data chsp(1316) /' '/
      data chsp(1317) /' '/
      data chsp(1318) /' '/
      data chsp(1319) /' '/
      data chsp(1320) /' '/
      data chsp(1321) /' '/
      data chsp(1322) /' '/
      data chsp(1323) /' '/
      data chsp(1324) /' '/
      data chsp(1325) /' '/
      data chsp(1326) /' '/
      data chsp(1327) /' '/
      data chsp(1328) /' '/
      data chsp(1329) /' '/
      data chsp(1330) /' '/
      data chsp(1331) /' '/
      data chsp(1332) /' '/
      data chsp(1333) /' '/
      data chsp(1334) /' '/
      data chsp(1335) /' '/
      data chsp(1336) /' '/
      data chsp(1337) /' '/
      data chsp(1338) /' '/
      data chsp(1339) /' '/
      data chsp(1340) /' '/
      data chsp(1341) /' '/
      data chsp(1342) /' '/
      data chsp(1343) /' '/
      data chsp(1344) /' '/
      data chsp(1345) /' '/
      data chsp(1346) /' '/
      data chsp(1347) /' '/
      data chsp(1348) /' '/
      data chsp(1349) /' '/
      data chsp(1350) /' '/
      data chsp(1351) /' '/
      data chsp(1352) /' '/
      data chsp(1353) /' '/
      data chsp(1354) /' '/
      data chsp(1355) /' '/
      data chsp(1356) /' '/
      data chsp(1357) /' '/
      data chsp(1358) /' '/
      data chsp(1359) /' '/
      data chsp(1360) /' '/
      data chsp(1361) /' '/
      data chsp(1362) /' '/
      data chsp(1363) /' '/
      data chsp(1364) /' '/
      data chsp(1365) /' '/
      data chsp(1366) /' '/
      data chsp(1367) /' '/
      data chsp(1368) /' '/
      data chsp(1369) /' '/
      data chsp(1370) /' '/
      data chsp(1371) /' '/
      data chsp(1372) /' '/
      data chsp(1373) /' '/
      data chsp(1374) /' '/
      data chsp(1375) /' '/
      data chsp(1376) /' '/
      data chsp(1377) /' '/
      data chsp(1378) /' '/
      data chsp(1379) /' '/
      data chsp(1380) /' '/
      data chsp(1381) /' '/
      data chsp(1382) /' '/
      data chsp(1383) /' '/
      data chsp(1384) /' '/
      data chsp(1385) /' '/
      data chsp(1386) /' '/
      data chsp(1387) /' '/
      data chsp(1388) /' '/
      data chsp(1389) /' '/
      data chsp(1390) /' '/
      data chsp(1391) /' '/
      data chsp(1392) /' '/
      data chsp(1393) /' '/
      data chsp(1394) /' '/
      data chsp(1395) /' '/
      data chsp(1396) /' '/
      data chsp(1397) /' '/
      data chsp(1398) /' '/
      data chsp(1399) /' '/
      data chsp(1400) /' '/
      data chsp(1401) /' '/
      data chsp(1402) /' '/
      data chsp(1403) /' '/
      data chsp(1404) /' '/
      data chsp(1405) /' '/
      data chsp(1406) /' '/
      data chsp(1407) /' '/
      data chsp(1408) /' '/
      data chsp(1409) /' '/
      data chsp(1410) /' '/
      data chsp(1411) /' '/
      data chsp(1412) /' '/
      data chsp(1413) /' '/
      data chsp(1414) /' '/
      data chsp(1415) /' '/
      data chsp(1416) /' '/
      data chsp(1417) /' '/
      data chsp(1418) /' '/
      data chsp(1419) /' '/
      data chsp(1420) /' '/
      data chsp(1421) /' '/
      data chsp(1422) /' '/
      data chsp(1423) /' '/
      data chsp(1424) /' '/
      data chsp(1425) /' '/
      data chsp(1426) /' '/
      data chsp(1427) /' '/
      data chsp(1428) /' '/
      data chsp(1429) /' '/
      data chsp(1430) /'eta_c(1S)'/
      data chsp(1431) /'J/psi(1S)'/
      data chsp(1432) /'chi_c2(1P)'/
      data chsp(1433) /' '/
      data chsp(1434) /' '/
      data chsp(1435) /' '/
      data chsp(1436) /'psi(4040)'/
      data chsp(1437) /'psi(3836)'/
      data chsp(1438) /' '/
      data chsp(1439) /' '/
      data chsp(1440) /'chi_c0(1P)'/
      data chsp(1441) /'hc(1P)'/
      data chsp(1442) /' '/
      data chsp(1443) /' '/
      data chsp(1444) /' '/
      data chsp(1445) /' '/
      data chsp(1446) /'psi(4160)'/
      data chsp(1447) /' '/
      data chsp(1448) /' '/
      data chsp(1449) /' '/
      data chsp(1450) /' '/
      data chsp(1451) /'chi_c1(1P)'/
      data chsp(1452) /' '/
      data chsp(1453) /' '/
      data chsp(1454) /' '/
      data chsp(1455) /' '/
      data chsp(1456) /'psi(4415)'/
      data chsp(1457) /' '/
      data chsp(1458) /' '/
      data chsp(1459) /' '/
      data chsp(1460) /' '/
      data chsp(1461) /'psi(3770)'/
      data chsp(1462) /' '/
      data chsp(1463) /' '/
      data chsp(1464) /' '/
      data chsp(1465) /' '/
      data chsp(1466) /' '/
      data chsp(1467) /' '/
      data chsp(1468) /' '/
      data chsp(1469) /' '/
      data chsp(1470) /'eta_c(2S)'/
      data chsp(1471) /'psi(2S)'/
      data chsp(1472) /'chi_c2(2P)'/
      data chsp(1473) /' '/
      data chsp(1474) /' '/
      data chsp(1475) /' '/
      data chsp(1476) /' '/
      data chsp(1477) /' '/
      data chsp(1478) /' '/
      data chsp(1479) /' '/
      data chsp(1480) /' '/
      data chsp(1481) /' '/
      data chsp(1482) /' '/
      data chsp(1483) /' '/
      data chsp(1484) /' '/
      data chsp(1485) /' '/
      data chsp(1486) /' '/
      data chsp(1487) /' '/
      data chsp(1488) /' '/
      data chsp(1489) /' '/
      data chsp(1490) /' '/
      data chsp(1491) /' '/
      data chsp(1492) /' '/
      data chsp(1493) /' '/
      data chsp(1494) /' '/
      data chsp(1495) /' '/
      data chsp(1496) /' '/
      data chsp(1497) /' '/
      data chsp(1498) /' '/
      data chsp(1499) /' '/
      data chsp(1500) /' '/
      data chsp(1501) /' '/
      data chsp(1502) /' '/
      data chsp(1503) /' '/
      data chsp(1504) /' '/
      data chsp(1505) /' '/
      data chsp(1506) /' '/
      data chsp(1507) /' '/
      data chsp(1508) /' '/
      data chsp(1509) /' '/
      data chsp(1510) /'B'/
      data chsp(1511) /'B*'/
      data chsp(1512) /'B*_2'/
      data chsp(1513) /' '/
      data chsp(1514) /' '/
      data chsp(1515) /' '/
      data chsp(1516) /' '/
      data chsp(1517) /' '/
      data chsp(1518) /' '/
      data chsp(1519) /' '/
      data chsp(1520) /'B*_0'/
      data chsp(1521) /'B_1(L)'/
      data chsp(1522) /' '/
      data chsp(1523) /' '/
      data chsp(1524) /' '/
      data chsp(1525) /' '/
      data chsp(1526) /' '/
      data chsp(1527) /' '/
      data chsp(1528) /' '/
      data chsp(1529) /' '/
      data chsp(1530) /' '/
      data chsp(1531) /'B_1(H)'/
      data chsp(1532) /' '/
      data chsp(1533) /' '/
      data chsp(1534) /' '/
      data chsp(1535) /' '/
      data chsp(1536) /' '/
      data chsp(1537) /' '/
      data chsp(1538) /' '/
      data chsp(1539) /' '/
      data chsp(1540) /'B'/
      data chsp(1541) /'B*'/
      data chsp(1542) /'B*_2'/
      data chsp(1543) /' '/
      data chsp(1544) /' '/
      data chsp(1545) /' '/
      data chsp(1546) /' '/
      data chsp(1547) /' '/
      data chsp(1548) /' '/
      data chsp(1549) /' '/
      data chsp(1550) /'B*_0'/
      data chsp(1551) /'B_1(L)'/
      data chsp(1552) /' '/
      data chsp(1553) /' '/
      data chsp(1554) /' '/
      data chsp(1555) /' '/
      data chsp(1556) /' '/
      data chsp(1557) /' '/
      data chsp(1558) /' '/
      data chsp(1559) /' '/
      data chsp(1560) /' '/
      data chsp(1561) /'B_1(H)'/
      data chsp(1562) /' '/
      data chsp(1563) /' '/
      data chsp(1564) /' '/
      data chsp(1565) /' '/
      data chsp(1566) /' '/
      data chsp(1567) /' '/
      data chsp(1568) /' '/
      data chsp(1569) /' '/
      data chsp(1570) /'B_s'/
      data chsp(1571) /'B*_s'/
      data chsp(1572) /'B*_s2'/
      data chsp(1573) /' '/
      data chsp(1574) /' '/
      data chsp(1575) /' '/
      data chsp(1576) /' '/
      data chsp(1577) /' '/
      data chsp(1578) /' '/
      data chsp(1579) /' '/
      data chsp(1580) /'B*_s0'/
      data chsp(1581) /'B_s1(L)'/
      data chsp(1582) /' '/
      data chsp(1583) /' '/
      data chsp(1584) /' '/
      data chsp(1585) /' '/
      data chsp(1586) /' '/
      data chsp(1587) /' '/
      data chsp(1588) /' '/
      data chsp(1589) /' '/
      data chsp(1590) /' '/
      data chsp(1591) /'B_s1(H)'/
      data chsp(1592) /' '/
      data chsp(1593) /' '/
      data chsp(1594) /' '/
      data chsp(1595) /' '/
      data chsp(1596) /' '/
      data chsp(1597) /' '/
      data chsp(1598) /' '/
      data chsp(1599) /' '/
      data chsp(1600) /'B_c'/
      data chsp(1601) /'B*_c'/
      data chsp(1602) /'B*_c2'/
      data chsp(1603) /' '/
      data chsp(1604) /' '/
      data chsp(1605) /' '/
      data chsp(1606) /' '/
      data chsp(1607) /' '/
      data chsp(1608) /' '/
      data chsp(1609) /' '/
      data chsp(1610) /'B*_c0'/
      data chsp(1611) /'B_c1(L)'/
      data chsp(1612) /' '/
      data chsp(1613) /' '/
      data chsp(1614) /' '/
      data chsp(1615) /' '/
      data chsp(1616) /' '/
      data chsp(1617) /' '/
      data chsp(1618) /' '/
      data chsp(1619) /' '/
      data chsp(1620) /' '/
      data chsp(1621) /'B_c1(H)'/
      data chsp(1622) /' '/
      data chsp(1623) /' '/
      data chsp(1624) /' '/
      data chsp(1625) /' '/
      data chsp(1626) /' '/
      data chsp(1627) /' '/
      data chsp(1628) /' '/
      data chsp(1629) /' '/
      data chsp(1630) /'eta_b(1S)'/
      data chsp(1631) /'upsilon(1S)'/
      data chsp(1632) /'chi_b2(1P)'/
      data chsp(1633) /'upsilon_3(1D)'/
      data chsp(1634) /' '/
      data chsp(1635) /' '/
      data chsp(1636) /'upsilon(10860)'/
      data chsp(1637) /' '/
      data chsp(1638) /' '/
      data chsp(1639) /' '/
      data chsp(1640) /'chi_b0(1P)'/
      data chsp(1641) /'h_b(1P)'/
      data chsp(1642) /'eta_b2(1D)'/
      data chsp(1643) /' '/
      data chsp(1644) /' '/
      data chsp(1645) /' '/
      data chsp(1646) /'upsilon(11020)'/
      data chsp(1647) /' '/
      data chsp(1648) /' '/
      data chsp(1649) /' '/
      data chsp(1650) /' '/
      data chsp(1651) /'chi_b1(1P)'/
      data chsp(1652) /'upsilon_2(1D)'/
      data chsp(1653) /' '/
      data chsp(1654) /' '/
      data chsp(1655) /' '/
      data chsp(1656) /'upsilon(7S)'/
      data chsp(1657) /' '/
      data chsp(1658) /' '/
      data chsp(1659) /' '/
      data chsp(1660) /' '/
      data chsp(1661) /'upsilon_1(1D)'/
      data chsp(1662) /' '/
      data chsp(1663) /' '/
      data chsp(1664) /' '/
      data chsp(1665) /' '/
      data chsp(1666) /' '/
      data chsp(1667) /' '/
      data chsp(1668) /' '/
      data chsp(1669) /' '/
      data chsp(1670) /'eta_b(2S)'/
      data chsp(1671) /'upsilon(2S)'/
      data chsp(1672) /'chi_b2(2P)'/
      data chsp(1673) /'upsilon_3(2D)'/
      data chsp(1674) /' '/
      data chsp(1675) /' '/
      data chsp(1676) /' '/
      data chsp(1677) /' '/
      data chsp(1678) /' '/
      data chsp(1679) /' '/
      data chsp(1680) /'chi_b0(2P)'/
      data chsp(1681) /'h_b(2P)'/
      data chsp(1682) /'eta_b2(2D)'/
      data chsp(1683) /' '/
      data chsp(1684) /' '/
      data chsp(1685) /' '/
      data chsp(1686) /' '/
      data chsp(1687) /' '/
      data chsp(1688) /' '/
      data chsp(1689) /' '/
      data chsp(1690) /' '/
      data chsp(1691) /'chi_b1(2P)'/
      data chsp(1692) /'upsilon_2(2D)'/
      data chsp(1693) /' '/
      data chsp(1694) /' '/
      data chsp(1695) /' '/
      data chsp(1696) /' '/
      data chsp(1697) /' '/
      data chsp(1698) /' '/
      data chsp(1699) /' '/
      data chsp(1700) /' '/
      data chsp(1701) /'upsilon_1(2D)'/
      data chsp(1702) /' '/
      data chsp(1703) /' '/
      data chsp(1704) /' '/
      data chsp(1705) /' '/
      data chsp(1706) /' '/
      data chsp(1707) /' '/
      data chsp(1708) /' '/
      data chsp(1709) /' '/
      data chsp(1710) /'eta_b(3S)'/
      data chsp(1711) /'upsilon(3S)'/
      data chsp(1712) /'chi_b2(3P)'/
      data chsp(1713) /' '/
      data chsp(1714) /' '/
      data chsp(1715) /' '/
      data chsp(1716) /' '/
      data chsp(1717) /' '/
      data chsp(1718) /' '/
      data chsp(1719) /' '/
      data chsp(1720) /'chi_b0(3P)'/
      data chsp(1721) /'h_b(3P)'/
      data chsp(1722) /' '/
      data chsp(1723) /' '/
      data chsp(1724) /' '/
      data chsp(1725) /' '/
      data chsp(1726) /' '/
      data chsp(1727) /' '/
      data chsp(1728) /' '/
      data chsp(1729) /' '/
      data chsp(1730) /' '/
      data chsp(1731) /'chi_b1(3P)'/
      data chsp(1732) /' '/
      data chsp(1733) /' '/
      data chsp(1734) /' '/
      data chsp(1735) /' '/
      data chsp(1736) /' '/
      data chsp(1737) /' '/
      data chsp(1738) /' '/
      data chsp(1739) /' '/
      data chsp(1740) /' '/
      data chsp(1741) /' '/
      data chsp(1742) /' '/
      data chsp(1743) /' '/
      data chsp(1744) /' '/
      data chsp(1745) /' '/
      data chsp(1746) /' '/
      data chsp(1747) /' '/
      data chsp(1748) /' '/
      data chsp(1749) /' '/
      data chsp(1750) /' '/
      data chsp(1751) /'upsilon(4S)'/
      data chsp(1752) /' '/
      data chsp(1753) /' '/
      data chsp(1754) /' '/
      data chsp(1755) /' '/
      data chsp(1756) /' '/
      data chsp(1757) /' '/
      data chsp(1758) /' '/
      data chsp(1759) /' '/
      data chsp(1760) /' '/
      data chsp(1761) /' '/
      data chsp(1762) /' '/
      data chsp(1763) /' '/
      data chsp(1764) /' '/
      data chsp(1765) /' '/
      data chsp(1766) /' '/
      data chsp(1767) /' '/
      data chsp(1768) /' '/
      data chsp(1769) /' '/
      data chsp(1770) /' '/
      data chsp(1771) /' '/
      data chsp(1772) /' '/
      data chsp(1773) /' '/
      data chsp(1774) /' '/
      data chsp(1775) /' '/
      data chsp(1776) /' '/
      data chsp(1777) /' '/
      data chsp(1778) /' '/
      data chsp(1779) /' '/
      data chsp(1780) /' '/
      data chsp(1781) /' '/
      data chsp(1782) /' '/
      data chsp(1783) /' '/
      data chsp(1784) /' '/
      data chsp(1785) /' '/
      data chsp(1786) /' '/
      data chsp(1787) /' '/
      data chsp(1788) /' '/
      data chsp(1789) /' '/
      data chsp(1790) /'T'/
      data chsp(1791) /'T*'/
      data chsp(1792) /' '/
      data chsp(1793) /' '/
      data chsp(1794) /' '/
      data chsp(1795) /' '/
      data chsp(1796) /' '/
      data chsp(1797) /' '/
      data chsp(1798) /' '/
      data chsp(1799) /' '/
      data chsp(1800) /'T'/
      data chsp(1801) /'T*'/
      data chsp(1802) /' '/
      data chsp(1803) /' '/
      data chsp(1804) /' '/
      data chsp(1805) /' '/
      data chsp(1806) /' '/
      data chsp(1807) /' '/
      data chsp(1808) /' '/
      data chsp(1809) /' '/
      data chsp(1810) /'T_s'/
      data chsp(1811) /'T*_s'/
      data chsp(1812) /' '/
      data chsp(1813) /' '/
      data chsp(1814) /' '/
      data chsp(1815) /' '/
      data chsp(1816) /' '/
      data chsp(1817) /' '/
      data chsp(1818) /' '/
      data chsp(1819) /' '/
      data chsp(1820) /'T_c'/
      data chsp(1821) /'T*_c'/
      data chsp(1822) /' '/
      data chsp(1823) /' '/
      data chsp(1824) /' '/
      data chsp(1825) /' '/
      data chsp(1826) /' '/
      data chsp(1827) /' '/
      data chsp(1828) /' '/
      data chsp(1829) /' '/
      data chsp(1830) /'T_b'/
      data chsp(1831) /'T*_b'/
      data chsp(1832) /' '/
      data chsp(1833) /' '/
      data chsp(1834) /' '/
      data chsp(1835) /' '/
      data chsp(1836) /' '/
      data chsp(1837) /' '/
      data chsp(1838) /' '/
      data chsp(1839) /' '/
      data chsp(1840) /'eta_t'/
      data chsp(1841) /'theta'/
      data chsp(1842) /' '/
      data chsp(1843) /' '/
      data chsp(1844) /' '/
      data chsp(1845) /' '/
      data chsp(1846) /' '/
      data chsp(1847) /' '/
      data chsp(1848) /' '/
      data chsp(1849) /' '/
      data chsp(1850) /'L'/
      data chsp(1851) /'L*'/
      data chsp(1852) /' '/
      data chsp(1853) /' '/
      data chsp(1854) /' '/
      data chsp(1855) /' '/
      data chsp(1856) /' '/
      data chsp(1857) /' '/
      data chsp(1858) /' '/
      data chsp(1859) /' '/
      data chsp(1860) /'L'/
      data chsp(1861) /'L*'/
      data chsp(1862) /' '/
      data chsp(1863) /' '/
      data chsp(1864) /' '/
      data chsp(1865) /' '/
      data chsp(1866) /' '/
      data chsp(1867) /' '/
      data chsp(1868) /' '/
      data chsp(1869) /' '/
      data chsp(1870) /'L_s'/
      data chsp(1871) /'L*_s'/
      data chsp(1872) /' '/
      data chsp(1873) /' '/
      data chsp(1874) /' '/
      data chsp(1875) /' '/
      data chsp(1876) /' '/
      data chsp(1877) /' '/
      data chsp(1878) /' '/
      data chsp(1879) /' '/
      data chsp(1880) /'L_c'/
      data chsp(1881) /'L*_c'/
      data chsp(1882) /' '/
      data chsp(1883) /' '/
      data chsp(1884) /' '/
      data chsp(1885) /' '/
      data chsp(1886) /' '/
      data chsp(1887) /' '/
      data chsp(1888) /' '/
      data chsp(1889) /' '/
      data chsp(1890) /'L_b'/
      data chsp(1891) /'L*_b'/
      data chsp(1892) /' '/
      data chsp(1893) /' '/
      data chsp(1894) /' '/
      data chsp(1895) /' '/
      data chsp(1896) /' '/
      data chsp(1897) /' '/
      data chsp(1898) /' '/
      data chsp(1899) /' '/
      data chsp(1900) /'L_t'/
      data chsp(1901) /'L*_t'/
      data chsp(1902) /' '/
      data chsp(1903) /' '/
      data chsp(1904) /' '/
      data chsp(1905) /' '/
      data chsp(1906) /' '/
      data chsp(1907) /' '/
      data chsp(1908) /' '/
      data chsp(1909) /' '/
      data chsp(1910) /'eta_l'/
      data chsp(1911) /'theta_l'/
      data chsp(1912) /' '/
      data chsp(1913) /' '/
      data chsp(1914) /' '/
      data chsp(1915) /' '/
      data chsp(1916) /' '/
      data chsp(1917) /' '/
      data chsp(1918) /' '/
      data chsp(1919) /' '/
      data chsp(1920) /'H'/
      data chsp(1921) /'H*'/
      data chsp(1922) /' '/
      data chsp(1923) /' '/
      data chsp(1924) /' '/
      data chsp(1925) /' '/
      data chsp(1926) /' '/
      data chsp(1927) /' '/
      data chsp(1928) /' '/
      data chsp(1929) /' '/
      data chsp(1930) /'H'/
      data chsp(1931) /'H*'/
      data chsp(1932) /' '/
      data chsp(1933) /' '/
      data chsp(1934) /' '/
      data chsp(1935) /' '/
      data chsp(1936) /' '/
      data chsp(1937) /' '/
      data chsp(1938) /' '/
      data chsp(1939) /' '/
      data chsp(1940) /'H_s'/
      data chsp(1941) /'H*_s'/
      data chsp(1942) /' '/
      data chsp(1943) /' '/
      data chsp(1944) /' '/
      data chsp(1945) /' '/
      data chsp(1946) /' '/
      data chsp(1947) /' '/
      data chsp(1948) /' '/
      data chsp(1949) /' '/
      data chsp(1950) /'H_c'/
      data chsp(1951) /'H*_c'/
      data chsp(1952) /' '/
      data chsp(1953) /' '/
      data chsp(1954) /' '/
      data chsp(1955) /' '/
      data chsp(1956) /' '/
      data chsp(1957) /' '/
      data chsp(1958) /' '/
      data chsp(1959) /' '/
      data chsp(1960) /'H_b'/
      data chsp(1961) /'H*_b'/
      data chsp(1962) /' '/
      data chsp(1963) /' '/
      data chsp(1964) /' '/
      data chsp(1965) /' '/
      data chsp(1966) /' '/
      data chsp(1967) /' '/
      data chsp(1968) /' '/
      data chsp(1969) /' '/
      data chsp(1970) /'H_t'/
      data chsp(1971) /'H*_t'/
      data chsp(1972) /' '/
      data chsp(1973) /' '/
      data chsp(1974) /' '/
      data chsp(1975) /' '/
      data chsp(1976) /' '/
      data chsp(1977) /' '/
      data chsp(1978) /' '/
      data chsp(1979) /' '/
      data chsp(1980) /'H_l'/
      data chsp(1981) /'H*_l'/
      data chsp(1982) /' '/
      data chsp(1983) /' '/
      data chsp(1984) /' '/
      data chsp(1985) /' '/
      data chsp(1986) /' '/
      data chsp(1987) /' '/
      data chsp(1988) /' '/
      data chsp(1989) /' '/
      data chsp(1990) /'eta_h'/
      data chsp(1991) /'theta_H'/
      data chsp(1992) /' '/
      data chsp(1993) /' '/
      data chsp(1994) /' '/
      data chsp(1995) /' '/
      data chsp(1996) /' '/
      data chsp(1997) /' '/
      data chsp(1998) /' '/
      data chsp(1999) /' '/
      data chsp(2000) /' '/
      data chsp(2001) /'Lambda'/
      data chsp(2002) /'Lambda_c'/
      data chsp(2003) /'Xi_c'/
      data chsp(2004) /'Xi_c'/
      data chsp(2005) /'Lambda_b'/
      data chsp(2006) /'Xi_b'/
      data chsp(2007) /'Xi_b'/
      data chsp(2008) /'Xi_bc'/
      data chsp(2009) /'Xi_bc'/
      data chsp(2010) /'Omega_bc'/
      data chsp(2011) /'Lambda_t'/
      data chsp(2012) /'Xi_t'/
      data chsp(2013) /'Xi_t'/
      data chsp(2014) /'Xi_tc'/
      data chsp(2015) /'Xi_tc'/
      data chsp(2016) /'Omega_tc'/
      data chsp(2017) /'Xi_tb'/
      data chsp(2018) /'Xi_tb'/
      data chsp(2019) /'Omega_tb'/
      data chsp(2020) /'Omega_tbc'/
      data chsp(2021) /' '/
      data chsp(2022) /'n'/
      data chsp(2023) /'p'/
      data chsp(2024) /' '/
      data chsp(2025) /'Sigma'/
      data chsp(2026) /'Sigma'/
      data chsp(2027) /'Sigma'/
      data chsp(2028) /'Xi'/
      data chsp(2029) /'Xi'/
      data chsp(2030) /' '/
      data chsp(2031) /'Sigma_c'/
      data chsp(2032) /'Sigma_c'/
      data chsp(2033) /'Sigma_c'/
      data chsp(2034) /'Xi''_c'/
      data chsp(2035) /'Xi''_c'/
      data chsp(2036) /'Omega_c'/
      data chsp(2037) /'Xi_cc'/
      data chsp(2038) /'Xi_cc'/
      data chsp(2039) /'Omega_cc'/
      data chsp(2040) /' '/
      data chsp(2041) /'Sigma_b'/
      data chsp(2042) /'Sigma_b'/
      data chsp(2043) /'Sigma_b'/
      data chsp(2044) /'Xi''_b'/
      data chsp(2045) /'Xi''_b'/
      data chsp(2046) /'Omega_b'/
      data chsp(2047) /'Xi''_bc'/
      data chsp(2048) /'Xi''_bc'/
      data chsp(2049) /'Omega''_bc'/
      data chsp(2050) /'Omega_bcc'/
      data chsp(2051) /'Xi_bb'/
      data chsp(2052) /'Xi_bb'/
      data chsp(2053) /'Omega_bb'/
      data chsp(2054) /'Omega_bbc'/
      data chsp(2055) /' '/
      data chsp(2056) /'Sigma_t'/
      data chsp(2057) /'Sigma_t'/
      data chsp(2058) /'Sigma_t'/
      data chsp(2059) /'Xi''_t'/
      data chsp(2060) /'Xi''_t'/
      data chsp(2061) /'Omega_t'/
      data chsp(2062) /'Xi''_tc'/
      data chsp(2063) /'Xi''_tc'/
      data chsp(2064) /'Omega''_tc'/
      data chsp(2065) /'Omega_tcc'/
      data chsp(2066) /'Xi''_tb'/
      data chsp(2067) /'Xi''_tb'/
      data chsp(2068) /'Omega''_tb'/
      data chsp(2069) /'Omega''_tbc'/
      data chsp(2070) /'Omega_tbb'/
      data chsp(2071) /'Xi_tt'/
      data chsp(2072) /'Xi_tt'/
      data chsp(2073) /'Omega_tt'/
      data chsp(2074) /'Omega_ttc'/
      data chsp(2075) /'Omega_ttb'/
      data chsp(2076) /' '/
      data chsp(2077) /' '/
      data chsp(2078) /' '/
      data chsp(2079) /' '/
      data chsp(2080) /' '/
      data chsp(2081) /'Delta'/
      data chsp(2082) /'Delta'/
      data chsp(2083) /'Delta'/
      data chsp(2084) /'Delta'/
      data chsp(2085) /'Sigma*'/
      data chsp(2086) /'Sigma*'/
      data chsp(2087) /'Sigma*'/
      data chsp(2088) /'Xi*'/
      data chsp(2089) /'Xi*'/
      data chsp(2090) /'Omega'/
      data chsp(2091) /'Sigma*_c'/
      data chsp(2092) /'Sigma*_c'/
      data chsp(2093) /'Sigma*_c'/
      data chsp(2094) /'Xi*_c'/
      data chsp(2095) /'Xi*_c'/
      data chsp(2096) /'Omega*_c'/
      data chsp(2097) /'Xi*_cc'/
      data chsp(2098) /'Xi*_cc'/
      data chsp(2099) /'Omega*_cc'/
      data chsp(2100) /'Omega*_ccc'/
      data chsp(2101) /'Sigma*_b'/
      data chsp(2102) /'Sigma*_b'/
      data chsp(2103) /'Sigma*_b'/
      data chsp(2104) /'Xi*_b'/
      data chsp(2105) /'Xi*_b'/
      data chsp(2106) /'Omega*_b'/
      data chsp(2107) /'Xi*_bc'/
      data chsp(2108) /'Xi*_bc'/
      data chsp(2109) /'Omega*_bc'/
      data chsp(2110) /'Omega*_bcc'/
      data chsp(2111) /'Xi*_bb'/
      data chsp(2112) /'Xi*_bb'/
      data chsp(2113) /'Omega*_bb'/
      data chsp(2114) /'Omega*_bbc'/
      data chsp(2115) /'Omega*_bbb'/
      data chsp(2116) /'Sigma*_t'/
      data chsp(2117) /'Sigma*_t'/
      data chsp(2118) /'Sigma*_t'/
      data chsp(2119) /'Xi*_t'/
      data chsp(2120) /'Xi*_t'/
      data chsp(2121) /'Omega*_t'/
      data chsp(2122) /'Xi*_tc'/
      data chsp(2123) /'Xi*_tc'/
      data chsp(2124) /'Omega*_tc'/
      data chsp(2125) /'Omega*_tcc'/
      data chsp(2126) /'Xi*_tb'/
      data chsp(2127) /'Xi*_tb'/
      data chsp(2128) /'Omega*_tb'/
      data chsp(2129) /'Omega*_tbc'/
      data chsp(2130) /'Omega*_tbb'/
      data chsp(2131) /'Xi*_tt'/
      data chsp(2132) /'Xi*_tt'/
      data chsp(2133) /'Omega*_tt'/
      data chsp(2134) /'Omega*_ttc'/
      data chsp(2135) /'Omega*_ttb'/
      data chsp(2136) /'Omega*_ttt'/
      data chsp(2137) /' '/
      data chsp(2138) /' '/
      data chsp(2139) /' '/
      data chsp(2140) /' '/
      data chsp(2141) /' '/
      data chsp(2142) /' '/
      data chsp(2143) /' '/
      data chsp(2144) /' '/
      data chsp(2145) /' '/
      data chsp(2146) /' '/
      data chsp(2147) /' '/
      data chsp(2148) /' '/
      data chsp(2149) /' '/
      data chsp(2150) /' '/
      data chsp(2151) /' '/
      data chsp(2152) /' '/
      data chsp(2153) /' '/
      data chsp(2154) /' '/
      data chsp(2155) /' '/
      data chsp(2156) /' '/
      data chsp(2157) /' '/
      data chsp(2158) /' '/
      data chsp(2159) /' '/
      data chsp(2160) /' '/
      data chsp(2161) /' '/
      data chsp(2162) /' '/
      data chsp(2163) /' '/
      data chsp(2164) /' '/
      data chsp(2165) /' '/
      data chsp(2166) /' '/
      data chsp(2167) /' '/
      data chsp(2168) /' '/
      data chsp(2169) /' '/
      data chsp(2170) /' '/
      data chsp(2171) /' '/
      data chsp(2172) /' '/
      data chsp(2173) /' '/
      data chsp(2174) /' '/
      data chsp(2175) /' '/
      data chsp(2176) /' '/
      data chsp(2177) /' '/
      data chsp(2178) /' '/
      data chsp(2179) /' '/
      data chsp(2180) /' '/
      data chsp(2181) /' '/
      data chsp(2182) /' '/
      data chsp(2183) /' '/
      data chsp(2184) /' '/
      data chsp(2185) /' '/
      data chsp(2186) /' '/
      data chsp(2187) /' '/
      data chsp(2188) /' '/
      data chsp(2189) /' '/
      data chsp(2190) /' '/
      data chsp(2191) /' '/
      data chsp(2192) /' '/
      data chsp(2193) /' '/
      data chsp(2194) /' '/
      data chsp(2195) /' '/
      data chsp(2196) /' '/
      data chsp(2197) /' '/
      data chsp(2198) /' '/
      data chsp(2199) /' '/
      data chsp(2200) /' '/
      data chsp(2201) /'Hydrogen'/
      data chsp(2202) /'Deuteron'/
      data chsp(2203) /'Tritium'/
      data chsp(2204) /'He3'/
      data chsp(2205) /'Alpha'/
      data chsp(2206) /' '/
      data chsp(2207) /' '/
      data chsp(2208) /' '/
      data chsp(2209) /' '/
      data chsp(2210) /' '/

      save chsp

      chau=' '
      chaup = chau
      if(ic.le.0 .or. ic.gt.2210) return

      chau=chsp(ic)
      chaup = chau

      return
      end

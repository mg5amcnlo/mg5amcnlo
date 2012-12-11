      subroutine hepnmg(ID,CHAUG)
c									       
c  HEPNMG --
c
C...Purpose: to give the particle/parton name as a character string.
c
c SCCS ID: hepnmg.f 1.1 4/6/92
c						
c Copyright (c) 1991 Universities Research Association, Inc.		        
c All rights reserved.							        
c 									        
c This material resulted from work developed under a Government Contract and    
c is subject to the following license:  The Government r~gh~nins a paid-up,       
c nonexclusive, irrevocable worldwide license to reproduce, prepare derivative  
c works, perform publicly and display publicly by or for the Government,        
c including the right to distribute to other Government contractors.  Neither   
c the United States nor the United States Department of Energy, nor any of      
c their employees, makes any warrenty, express or implied, or assumes any       
c legal liability or responsibility for the accuracy, completeness, or          
c usefulness of any information, apparatus, product, or process disclosed, or   
c represents that its use would not infringe privately owned rights.            
c                                        				        
c Written by Paul Lebrun & Lynn Garren					                                                                   
c
c	Arguments : 
c
c	ID  :Standard particle name See Convention in Particle Data book,
c		Physics LetterB vol 239, p. III.67
c
c	CHAUG : Null terminated Character string, the name of 
c		particle, using Greek fonts, as defined in module StringUtils.c
c		Limited to 30 characters
c	
      IMPLICIT NONE

      INTEGER ID, IDEX, IDA, IDQ1, IDQ2, IDQ3, LEN, LEM, KQN
      INTEGER I, KQ, IDJS, IDCM
      CHARACTER CHAU*25, CHAUG*25
      BYTE ICHAU(25)
      EQUIVALENCE (ICHAU(1), CHAU(1:1))
      character CHSP(1605)*20

      integer hepchg,hepcmp
      external hepchg,hepcmp

      data (CHSP(I), I=1,119) /'d','u','s','c','b','t','l','h',2*' ',
     1'e','~gn~n~de~n','~gm~n','~gn~n~dm~n','~gt~n','~gn~n~dt~n',
     1'~gc~n','~gn~n~dc~n','reggeon','pomeron','g','~gg~n','Z','W',
     2'H~d1~n',' ','~gh~n~dtech~n','LQ','R','Z~d2~n','W~d2~n','H~d2~n',
     3'H~d2~n','H~d2~n','Z~d3~n','H~d3~n','H~d3~n','H~d3~n','H~d4~n',
     4'H~d5~n','L susy d','L susy u','L susy s','L susy c','L susy b',
     5'L susy t','gluino','photino',2*' ','L susy e',
     6'L susy ~gn~n~de~n','L susy ~gm~n','L susy ~gn~n~dm~n',
     7'L susy ~gt~n','L susy ~gn~n~dt~n','susy HL','susy HH','susy HA',
     8' ','R susy d','R susy u','R susy s','R susy c','R susy b',
     9'R susy t','zino','susy Z2','susy Z3','susy Z4','R susy e',
     *'R susy ~gn~n~de~n','R susy ~gm~n','R susy ~gn~n~dm~n',
     1'R susy ~gt~n','R susy ~gn~n~dt~n','wino','susy W2',2*' ',
     2 19*'gen. code',' ','K~ds~n','K~dl~n','diquark','l-baryon',
     3'h-baryon',' ','~gr~n~ddiffr~n','~gp~n~ddiffr~n','~gw~n~ddiffr~n',
     4'~gf~n~ddiffr~n','~gY~n~ddiffr~n','n~ddiffr~n','p~ddiffr~n',
     5'remnant ~gg~n','remnant nucleon',4*' '/
      data (CHSP(I), I=120,219) /'~gp~n','a~d0~n(980)','~gp~n(1300)',
     17*' ','~gr~n(770)','b~d1~n(1235)','a~d1~n(1260)','~gr~n(1450)',
     1'~gr~n(1700)',5*' ','a~d2~n(1320)','~gp~n~d2~n(1670)',8*' ',
     2'~gr~n~d3~n(1690)',19*' ','~gh~n','f~d0~n(975)','~gh~n(1295)',
     3'f~d0~n(1400)','~gh~n(1440)','f~d0~n(1590)','f~d0~n(1710)',3*' ',
     4'~gw~n(783)','h~d1~n(1170)','f~d1~n(1285)','~gw~n(1390)',
     5'f~d1~n(1420)','f~d1~n(1510)','~gw~n(1600)',3*' ','f~d2~n(1270)',
     6' ','f~d2~n(2010)','f~d2~n(2300)','f~d2~n(2340)',5*' ',
     7'~gw~n~d3~n(1670)',9*' ','f~d4~n(2050)',9*' '/
      data (CHSP(I), I=220,369) /'~gh~n''(958)','f''0',8*' ',
     1 '~gf~n(1020)','h~d1~n(1380)','f~d1~n(1420)','~gf~n(1680)',
     1 'f''~d1~n',5*' ','f''~d2~n(1525)', 9*' ','~gf~n~d3~n(1850)',
     2 19*' ','~gh~n~dc~n(1S)','~gc~n~dc0~n(1P)','~gh~n~dc~n(2S)',7*' ',
     3 'J/~gY~n','hc(1P)','~gc~n~dc1~n(1P)','~gY~n(2S)','~gY~n(3770)',
     4 '~gY~n(4040)','~gY~n(4160)','~gY~n(4415)',2*' ',
     4 '~gc~n~dc2~n(1P)',29*' ','~gh~n~db~n(1S)','~gc~n~db0~n(1P)',
     5 '~gh~n~db~n(2S)','~gc~n~db0~n(2P)','~gh~n~db~n(3S)',
     5 5*' ','~gU~n(1S)','h~db~n(1P)','~gc~n~db1~n(1P)',
     6 '~gU~n(2S)','h~db~n(2P)','~gc~n~db1~n(2P)','~gU~n(3S)',
     7 '~gU~n(4S)','~gU~n(10860)','~gU~n(11020)',
     8 '~gc~n~db2~n(1P)','~gc~n~db2~n(2P)',28*' '/
      data (CHSP(I), I=370,419) /'~gh~n~dt~n','~gc~n~dt0~n',8*' ',
     1'~gq~n','h~d1t~n','~gc~n~dt1~n',7*' ','~gc~n~dt2~n',29*' '/
      data (CHSP(I), I=420,449) /'~gh~n~dl~n','~gc~n~dl0~n',' ',
     2'~gq~n~dl~n','h~d1l~n','~gc~n~dl1~n','~gc~n~dl2~n',8*' ',
     3'~gh~n~dh~n','~gc~n~dh0~n',' ','~gq~n~dh~n','h~d1h~n',
     4'~gc~n~dh1~n','~gc~n~h2~n',8*' '/
      data (CHSP(I), I=450,499) /'~gp~n','a~d0~n(980)','~gp~n(1300)',
     1 7*' ','~gr~n(770)','b~d1~n(1235)','a~d1~n(1260)','~gr~n(1450)',
     2 '~gr~n(1700)',5*' ','a~d2~n(1320)','~gp~n~d2~n(1670)',
     2 8*' ','~gr~n~d3~n(1690)',19*' '/
      data (CHSP(I), I=500,599) /'K','K~u*~n~d0~n(1430)',8*' ',
     1'K~u*~n(892)','K~d1~n(1270)','K~d1~n(1400)','K~u*~n(1410)',
     1'K~u*~n(1680)',5*' ','K~u*~n~d2~n(1430)','K~d2~n(1770)',
     2'K~d2~n(1820)',7*' ','K~u*~n~d3~n(1780)',9*' ',
     3 'K~u*~n~d4~n(2045)',9*' ','K','K~u*~n~d0~n(1430)',8*' ',
     3'K~u*~n(892)','K~d1~n(1270)','K~d1~n(1400)','K~u*~n(1410)',
     3'K~u*~n(1680)', 5*' ','K~u*~n~d2~n(1430)','K~d2~n(1770)',
     4'K~d2~n(1820)',7*' ', 'K~u*~n~d3~n(1780)',9*' ',
     5'K~u*~n~d4~n(2045)',9*' '/
      data (CHSP(I), I=600,749) /'D','D~u*~n~d0~n',8*' ','D~u*~n(2010)',
     1'D~d1~n(2420)','D~u*~n~d1~n',7*' ','D~u*~n~d2~n(2460)',29*' ',
     1'D','D~u*~n~d0~n',8*' ','D~u*~n(2010)','D~d1~n(2420)',
     2'D~u*~n~d1~n',7*' ','D~u*~n~d2~n(2460)',29*' ','D~ds~n',
     3'D~u*~n~ds0~n',8*' ','D~u*~n~ds~n','D~ds1~n(2536)',
     4'D~u*~n~ds1~n',7*' ','D~u*~n~ds2~n',29*' '/
      data (CHSP(I), I=750,949) /'B','B~u*~n~d0~n',8*' ','B~u*~n',
     1'B~d1~n','B~u*~n~d1~n',7*' ','B~u*~n~d2~n',29*' ','B',
     1'B~u*~n~d0~n',8*' ','B~u*~n','B~d1~n','B~u*~n~d1~n',7*' ',
     2'B~u*~n~d2~n',29*' ','B~ds~n','B~u*~n~ds0~n',8*' ',
     3'B~u*~n~ds~n','B~ds1~n','B~u*~n~ds1~n',7*' ','B~u*~n~ds2~n',
     3 29*' ','B~dc~n','B~u*~n~dc0~n',8*' ','B~u*~n~dc~n',
     3'B~dc1~n','B~u*~n~dc1~n',7*' ','B~u*~n~dc2~n',29*' '/
      data (CHSP(I), I=950,1199) /'T','T~u*~n~d0~n',8*' ','T~u*~n',
     1'T~d1~n','T~u*~n~d1~n',7*' ','T~u*~n~d2~n',29*' ','T',
     1'T~u*~n~d0~n',8*' ','T~u*~n','T~d1~n','T~u*~n~d1~n',7*' ',
     2'T~u*~n~d2~n',29*' ','T~ds~n','T~u*~n~ds0~n',8*' ',
     3'T~u*~n~ds~n','T~ds1~n','T~u*~n~ds1~n',7*' ','T~u*~n~ds2~n',
     3 29*' ','T~dc~n','T~u*~n~dc0~n',8*' ','T~u*~n~dc~n','T~dc1~n',
     3'T~u*~n~dc1~n',7*' ','T~u*~n~dc2~n',29*' ','T~db~n',
     3 'T~u*~n~db0~n',8*' ','T~u*~n~db~n','T~db1~n','T~u*~n~db1~n',
     4 7*' ','T~u*~n~db2~n',29*' '/
      data (CHSP(I), I=1200,1289) /'L','L~u*~n~d0~n',' ','L~u*~n',
     1'L~d1~n','L~u*~n~d1~n','L~u*~n~d2~n',8*' ','L','L~u*~n~d0~n',
     1' ','L~u*~n','L~d1~n','L~u*~n~d1~n','L~u*~n~d2~n',8*' ','L~ds~n',
     2'L~u*~n~ds0~n',' ','L~u*~n~ds~n','L~ds1~n','L~u*~n~ds1~n',
     2'L~u*~n~ds2~n',8*' ','L~dc~n','L~u*~n~dc0~n',' ','L~u*~n~dc~n',
     3'L~dc1~n','L~u*~n~dc1~n','L~u*~n~dc2~n',8*' ','L~db~n',
     4'L~u*~n~db0~n',' ','L~u*~n~db~n','L~db1~n','L~u*~n~db1~n',
     4'L~u*~n~db2~n',8*' ','L~dt~n','L~u*~n~dt0~n',' ',
     5'L~u*~n~dt~n','L~dt1~n','L~u*~n~dt1~n','L~u*~n~dt2~n',8*' '/
      data (CHSP(I), I=1290,1394) /'H','H~u*~n~d0~n',' ','H~u*~n',
     1'H~d1~n','H~u*~n~d1~n','H~u*~n~d2~n',8*' ','H','H~u*~n~d0~n',
     1' ','H~u*~n','H~d1~n','H~u*~n~d1~n','H~u*~n~d2~n',8*' ','H~ds~n',
     2'H~u*~n~ds0~n',' ','H~u*~n~ds~n','H~ds1~n','H~u*~n~ds1~n',
     2'H~u*~n~ds2~n',8*' ','H~dc~n','H~u*~n~dc0~n',' ','H~u*~n~dc~n',
     3'H~dc1~n','H~u*~n~dc1~n','H~u*~n~dc2~n',8*' ','H~db~n',
     4'H~u*~n~db0~n',' ','H~u*~n~db~n','H~db1~n','H~u*~n~db1~n',
     4'H~u*~n~db2~n',8*' ','H~dt~n','H~u*~n~dt0~n',' ',
     5'H~u*~n~dt~n','H~dt1~n','H~u*~n~dt1~n','H~u*~n~dt2~n',8*' ',
     6'H~dl~n','H~u*~n~dl0~n',' ','H~u*~n~dl~n','H~dl1~n',
     7'H~u*~n~dl1~n','H~u*~n~dl2~n',8*' '/
      data (CHSP(I), I=1395,1415) /' ','~gL~n','~gL~n~dc~n',
     1'~gX~n~dc~n','~gX~n~dc~n','~gL~n~db~n','~gX~n~db~n',
     1'~gX~n~db~n','~gX~n~dbc~n','~gX~n~dbc~n','~gW~n~dbc~n',
     2'~gL~n~dt~n','~gX~n~dt~n','~gX~n~dt~n','~gX~n~dtc~n',
     3'~gX~n~dtc~n','~gW~n~dtc~n','~gX~n~dtb~n','~gX~n~dtb~n',
     4'~gW~n~dtb~n','~gW~n~dtbc~n'/
      data (CHSP(I), I=1416,1475) /' ','n','p',' ','~gS~n','~gS~n',
     1'~gS~n','~gX~n','~gX~n',' ','~gS~n~dc~n','~gS~n~dc~n',
     1'~gS~n~dc~n','~gX~n~dc~n''','~gX~n~dc~n''','~gW~n~dc~n',
     2'~gX~n~dcc~n','~gX~n~dcc~n','~gW~n~dcc~n',' ','~gS~n~db~n',
     3'~gS~n~db~n','~gS~n~db~n','~gX~n~db~n''','~gX~n~db~n''',
     3'~gW~n~db~n','~gX~n~dbc~n''','~gX~n~dbc~n''','~gW~n~dbc~n''',
     4'~gW~n~dbcc~n','~gX~n~dbb~n','~gX~n~dbb~n','~gW~n~dbb~n',
     5'~gW~n~dbbc~n',' ','~gS~n~dt~n','~gS~n~dt~n','~gS~n~dt~n',
     5'~gX~n~dt~n''','~gX~n~dt~n''','~gW~n~dt~n','~gX~n~dtc~n''',
     6'~gX~n~dtc~n''','~gW~n~dtc~n''','~gW~n~dtcc~n','~gX~n~dtb~n''',
     7'~gX~n~dtb~n''','~gW~n~dtb~n''','~gW~n~dtbc~n''','~gW~n~dtbb~n',
     7'~gX~n~dtt~n','~gX~n~dtt~n','~gW~n~dtt~n','~gW~n~dttc~n',
     8'~gW~n~dttb~n',5*' '/
      data (CHSP(I), I=1476,1535) /'~gD~n','~gD~n','~gD~n','~gD~n',
     1'~gS~n~u*~n','~gS~n~u*~n','~gS~n~u*~n','~gX~n~u*~n',
     1'~gX~n~u*~n','~gW~n','~gS~n~u*~n~dc~n','~gS~n~u*~n~dc~n',
     2'~gS~n~u*~n~dc~n','~gX~n~u*~n~dc~n','~gX~n~u*~n~dc~n',
     2'~gW~n~u*~n~dc~n','~gX~n~u*~n~dcc~n','~gX~n~u*~n~dcc~n',
     3'~gW~n~u*~n~dcc~n','~gW~n~u*~n~dccc~n','~gS~n~u*~n~db~n',
     4'~gS~n~u*~n~db~n','~gS~n~u*~n~db~n','~gX~n~u*~n~db~n',
     4'~gX~n~u*~n~db~n','~gW~n~u*~n~db~n','~gX~n~u*~n~dbc~n',
     5'~gX~n~u*~n~dbc~n','~gW~n~u*~n~dbc~n','~gW~n~u*~n~dbcc~n',
     6'~gX~n~u*~n~dbb~n','~gX~n~u*~n~dbb~n','~gW~n~u*~n~dbb~n',
     6'~gW~n~u*~n~dbbc~n','~gW~n~u*~n~dbbb~n','~gS~n~u*~n~dt~n',
     7'~gS~n~u*~n~dt~n','~gS~n~u*~n~dt~n','~gX~n~u*~n~dt~n',
     7'~gX~n~u*~n~dt~n','~gW~n~u*~n~dt~n','~gX~n~u*~n~dtc~n',
     8'~gX~n~u*~n~dtc~n','~gW~n~u*~n~dtc~n','~gW~n~u*~n~dtcc~n',
     8'~gX~n~u*~n~dtb~n','~gX~n~u*~n~dtb~n','~gW~n~u*~n~dtb~n',
     9'~gW~n~u*~n~dtbc~n','~gW~n~u*~n~dtbb~n','~gX~n~u*~n~dtt~n',
     9'~gX~n~u*~n~dtt~n','~gW~n~u*~n~dtt~n','~gW~n~u*~n~dttc~n',
     *'~gW~n~u*~n~dttb~n','~gW~n~u*~n~dttt~n',4*' '/
      data (CHSP(I), I=1536,1600) /65*' '/
      data (CHSP(I), I=1601,1605) /'Hydrogen','Deuteron','Tritium',
     1'He~n~u3~n','Alpha'/


      save CHSP


C...Initial values. Charge (in units of 1/3). Subdivide code.
      CHAU=' '
      LEN=1
      IDA=IABS(ID)
      if(IDA.EQ.0) go to 900

      IDCM=hepcmp(ID)
      if(IDCM.LE.0) go to 900

      KQ=hepchg(ID)
      KQN=MOD(IDA/1000000000,10)
      IDQ3=MOD(IDA/1000,10)
      IDQ2=MOD(IDA/100,10)
      IDQ1=MOD(IDA/10,10)
      IDJS=MOD(IDA,10)
      IDEX=MOD(IDA/10000,10)

C...Read out root name and spin for simple particles and special cases
      if(IDA.LE.100) then
        CHAU=CHSP(IDCM)
        LEN=1
        do 100 LEM=1,20
  100     if(CHAU(LEM:LEM).NE.' ') LEN=LEM
      elseif(KQN.eq.1) then
        CHAU=CHSP(IDCM)
        LEN=0
        do 110 LEM=1,20
  110     if(CHAU(LEM:LEM).NE.' ') LEN=LEM
      elseif(IDJS.EQ.0) then
        CHAU=CHSP(IDCM)
        LEN=1
        do 150 LEM=1,20
  150     if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Construct root name for diquark. Add on spin.
      elseif(IDQ1.EQ.0) then
        CHAU(1:2)=CHSP(IDQ3)(1:1)//CHSP(IDQ2)(1:1)
        if(IDJS.EQ.1) CHAU(3:7)='~d0~n'
        if(IDJS.EQ.3) CHAU(3:7)='~d1~n'
        LEN=7

C...Construct root name for meson.
      elseif(IDQ3.EQ.0) then
        CHAU=CHSP(IDCM)
        LEN=1
        do 200 LEM=1,20
  200     if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Construct root name for Herwig remnant particles
      elseif(IDQ1.eq.9 .and. IDQ2.eq.9 .and. IDQ3.eq.9) then
        CHAU=CHSP(IDCM)
        LEN=1
        do 250 LEM=1,20
  250     if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Construct root name and spin for baryon.
      else
        if(IDQ3.LE.6) then
          CHAU=CHSP(IDCM)
          LEN=1
          do 300 LEM=1,20
  300       if(CHAU(LEM:LEM).NE.' ') LEN=LEM
        else
C...Construct root name and spin for heavy baryon.
          LEN=1
          if(IDQ2.LE.2 .AND. IDQ1.LE.2)then
            CHAU='~gS~n'
            if(IDJS.EQ.4) CHAU='~gS~n~u*~n'
            if(IDQ1.GT.IDQ2) CHAU='~gL~n'
          elseif(IDQ1.LE.2) then
            CHAU='~gX~n'''
            if(IDJS.EQ.4) CHAU='~gX~n~u*~n'
          elseif(IDQ2.LE.2) then
            CHAU='~gX~n'
          elseif(IDQ1.GT.IDQ2) then
            CHAU='~gW~n'
          else
            CHAU='~gW~n'''
            if(IDJS.EQ.4) CHAU='~gW~n~u*~n'
          endif
          do 320 LEM=1,20
  320       if(CHAU(LEM:LEM).NE.' ') LEN=LEM

C...Add on heavy flavour content for heavy baryon.
          CHAU(LEN+1:LEN+5)='~d'//CHSP(IDQ3)(1:1)//'~n'
          LEN=LEN+5
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
        CHAU(LEN+1:LEN+2)='~~'
        LEN=LEN+2
      endif

C...Add on charge where applicable (conventional cases skipped).
      if(IDA.GE.81.AND.IDA.LE.100) then
C...generator specific codes
      elseif(IDCM.eq.114 .or. IDCM.eq.115) then
C...Herwig remnant particles
      elseif(KQN.eq.1) then
C...ions
      elseif(LEN.GT.0)then
C...everything else
        if(abs(KQ).EQ.6)then
          if(KQ.EQ.6) CHAU(LEN+1:LEN+6)='~u++~n'
          if(KQ.EQ.-6) CHAU(LEN+1:LEN+6)='~u--~n'
          LEN = LEN + 6
        elseif(abs(KQ).eq.3)then
          if(KQ.EQ.3) CHAU(LEN+1:LEN+4)='~u+~n'
          if(KQ.EQ.-3) CHAU(LEN+1:LEN+4)='~u-~n'
          LEN = LEN + 4
        endif

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
          CHAU(LEN+1:LEN+5)='~u0~n'
          LEN = LEN + 5
        endif
      endif

900   continue
      ICHAU(LEN+1) = 0
      CHAUG = CHAU
      return
      end

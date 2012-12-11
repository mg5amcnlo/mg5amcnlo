C -------------------------------------------------------------
C
      common/HEPEVTO/NEVHEPO,NHEPO,ISTHEPO(NMXHEP),IDHEPO(NMXHEP),
     1 JMOHEPO(2,NMXHEP),JDAHEPO(2,NMXHEP),
     2 PHEPO(5,NMXHEP),VHEPO(4,NMXHEP)
      integer NEVHEPO,NHEPO,ISTHEPO,IDHEPO,JMOHEPO,JDAHEPO
      real PHEPO,VHEPO
      save /HEPEVTO/
C...      NEVHEPO       - event number
C...      NHEPO         - number of entries in this event
C...      ISTHEPO(..)   - status code
C...      IDHEPO(..)    - particle ID, P.D.G. standard
C...      JMOHEPO(1,..) - position of mother particle in list
C...      JMOHEPO(2,..) - position of second mother particle in list
C...      JDAHEPO(1,..) - position of first daughter in list
C...      JDAHEPO(2,..) - position of last daughter in list
C...      PHEPO(1,..)   - x momentum in GeV/c
C...      PHEPO(2,..)   - y momentum in GeV/c
C...      PHEPO(3,..)   - z momentum in GeV/c
C...      PHEPO(4,..)   - energy in GeV
C...      PHEPO(5,..)   - mass in GeV/c**2
C...      VHEPO(1,..)   - x vertex position in mm
C...      VHEPO(2,..)   - y vertex position in mm
C...      VHEPO(3,..)   - z vertex position in mm
C...      VHEPO(4,..)   - production time in mm/c
C
C -------------------------------------------------------------
C
      common/hepev2o/nmultio,jmultio(NMXHEP)
      integer nmultio,jmultio
      save /hepev2o/
C...      nmultio       - number of interactions in the list
C...      jmultio(..)   - multiople interaction number
C
C -------------------------------------------------------------

C -------------------------------------------------------------
C
C  This is the temporary common block used to get events to and
C  from xdr.  It enables reading multiple stdhep input streams.
C
      common/stdtmp/ phept(5,NMXHEP),vhept(4,NMXHEP),
     1 nevhept,nhept,isthept(NMXHEP),idhept(NMXHEP),
     & jmohept(2,NMXHEP),jdahept(2,NMXHEP)
      integer nevhept,nhept,isthept,idhept,jmohept,jdahept
      double precision phept,vhept
      save /stdtmp/
C...      nevhept       - event number
C...      nhept         - number of entries in this event
C...      isthept(..)   - status code
C...      idhept(..)    - particle ID, P.D.G. standard
C...      jmohept(1,..) - position of mother particle in list
C...      jmohept(2,..) - position of second mother particle in list
C...      jdahept(1,..) - position of first daughter in list
C...      jdahept(2,..) - position of last daughter in list
C...      phept(1,..)   - x momentum in GeV/c
C...      phept(2,..)   - y momentum in GeV/c
C...      phept(3,..)   - z momentum in GeV/c
C...      phept(4,..)   - energy in GeV
C...      phept(5,..)   - mass in GeV/c**2
C...      vhept(1,..)   - x vertex position in mm
C...      vhept(2,..)   - y vertex position in mm
C...      vhept(3,..)   - z vertex position in mm
C...      vhept(4,..)   - production time in mm/c
C
C -------------------------------------------------------------
c -------------------------------------------------------------
c hepev4 holds Les Houches information
c
      double precision eventweightt, scalet
      double precision alphaqedt, alphaqcdt, spint
      integer          icolorflowt, idrupt
c
      common/tmpev4/eventweightt, alphaqedt, alphaqcdt, scalet(10),
     1              spint(3,NMXHEP), icolorflowt(2,NMXHEP), idrupt
      save /tmpev4/

C idrupt            : The identity of the current process, 
C                      as given by the LPRUP codes.
C eventweightt      : The event weight:  
C                      Equal to (total cross section)/(total generated)
C                      for the output of Pythia, Herwig, etc.
C scalet            : Scale Q of the event.
C                      (fact. scale for PDF and energy scale for ISR and FSR)
C alphaqedt         : QED coupling alpha_em.
C alphaqcdt         : QCD coupling alpha_s.
C spint(3,..)       : spin information
C icolorflowt(2,..) : (Anti-)Colour flow.
C
c -------------------------------------------------------------

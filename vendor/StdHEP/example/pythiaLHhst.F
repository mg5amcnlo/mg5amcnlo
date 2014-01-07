      subroutine pythiaflem
C... fill histograms
#include "stdhep.inc"
#include "hepev4.inc"

      integer i

      call hfill(11,float(nevhep),0.,1.)
      call hfill(12,float(nhep),0.,1.)
      do 100 i=1,nhep
        call hfill(13,float(idhep(i)),0.,1.)
        call hfill(14,float(isthep(i)),0.,1.)
        call hfill(15,float(jmohep(1,i)),0.,1.)
        call hfill(16,float(jmohep(2,i)),0.,1.)
        call hfill(17,float(jdahep(1,i)),0.,1.)
        call hfill(18,float(jdahep(2,i)),0.,1.)
        call hfill(21,sngl(phep(1,i)),0.,1.)
        call hfill(22,sngl(phep(2,i)),0.,1.)
        call hfill(23,sngl(phep(3,i)),0.,1.)
        call hfill(24,sngl(phep(4,i)),0.,1.)
        call hfill(25,sngl(phep(5,i)),0.,1.)
        call hfill(26,sngl(vhep(1,i)),0.,1.)
        call hfill(27,sngl(vhep(2,i)),0.,1.)
        call hfill(28,sngl(vhep(3,i)),0.,1.)
        call hfill(29,sngl(vhep(4,i)),0.,1.)
 100  continue
      call hfill(31,sngl(eventweightlh),0.,1.)
      call hfill(32,sngl(alphaqedlh),0.,1.)
      call hfill(33,sngl(alphaqcdlh),0.,1.)
      call hfill(34,sngl(scalelh(1)),0.,1.)
      return
      end

      subroutine pythiabkem
C--- book histograms
C
      IMPLICIT NONE
C---   The HBOOK common
      REAL HMEMOR
      COMMON/PAWC/HMEMOR(50000)
C---   Setup the HBOOK memory limit.
      CALL HLIMIT(50000)
C---   Book histograms.
      call hbook1(11,'Pythia: NEVHEP',100,0.,100.,0.)
      call hbook1(12,'Pythia: NHEP',100,0.,200.,0.)
      call hbook1(13,'Pythia: IDHEP',100,0.,100.,0.)
      call hbook1(14,'Pythia: ISTHEP',20,0.,20.,0.)
      call hbook1(15,'Pythia: JMOHEP(1,)',100,0.,100.,0.)
      call hbook1(16,'Pythia: JMOHEP(2,)',100,0.,100.,0.)
      call hbook1(17,'Pythia: JDAHEP(1,)',100,0.,100.,0.)
      call hbook1(18,'Pythia: JDAHEP(2,)',100,0.,100.,0.)
      call hbook1(21,'Pythia: PHEP(1,)',100,-100.,100.,0.)
      call hbook1(22,'Pythia: PHEP(2,)',100,-100.,100.,0.)
      call hbook1(23,'Pythia: PHEP(3,)',100,-100.,100.,0.)
      call hbook1(24,'Pythia: PHEP(4,)',100,-100.,100.,0.)
      call hbook1(25,'Pythia: PHEP(5,)',100,-100.,100.,0.)
      call hbook1(26,'Pythia: VHEP(1,)',100,-50.,50.,0.)
      call hbook1(27,'Pythia: VHEP(2,)',100,-50.,50.,0.)
      call hbook1(28,'Pythia: VHEP(3,)',100,-50.,50.,0.)
      call hbook1(29,'Pythia: VHEP(4,)',100,0.,100.,0.)
      call hbook1(31,'Pythia: eventweightlh',100,-1.0,1.0,0.)
      call hbook1(32,'Pythia: alphaqedlh',100,-1.0,1.0,0.)
      call hbook1(33,'Pythia: alphaqcdlh',100,-1.0,1.0,0.)
      call hbook1(34,'Pythia: scalelh(1)',100,-10.,10.,0.)
      return
      end

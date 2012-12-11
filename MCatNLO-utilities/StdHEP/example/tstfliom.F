      subroutine tstfliom
C... fill histograms
#include "stdhep.inc"

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
        call hfill(19,float(jmulti(i)),0.,1.)
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
      call hfill(30,float(nmulti),0.,1.)
      do i=1,nmulti
        call hfill(31,float(nevmulti(i)),0.,1.)
        call hfill(32,float(itrkmulti(i)),0.,1.)
        call hfill(33,float(mltstr(i)),0.,1.)
      enddo
      return
      end

      subroutine tstbkiom
C--- book histograms
C
      implicit none
C---   The HBOOK common
      real hmemor
      common/pawc/hmemor(50000)

C---   Setup the HBOOK memory limit.
      call hlimit(50000)
C---   Book histograms.

      call hbook1(11,'nevhep',100,0.0,100.0,0.)
      call hbook1(12,'nhep',100,0.,1000.,0.)
      call hbook1(13,'idhep',100,0.,100.,0.)
      call hbook1(14,'isthep',100,0.,200.,0.)
      call hbook1(15,'jmohep(1,)',100,0.,1000.,0.)
      call hbook1(16,'jmohep(2,)',100,0.,1000.,0.)
      call hbook1(17,'jdahep(1,)',100,0.,1000.,0.)
      call hbook1(18,'jdahep(2,)',100,0.,1000.,0.)
      call hbook1(19,'jmulti',20,0.,10.,0.)
      call hbook1(21,'phep(1,)',100,-100.,100.,0.)
      call hbook1(22,'phep(2,)',100,-100.,100.,0.)
      call hbook1(23,'phep(3,)',100,-1000.,1000.,0.)
      call hbook1(24,'phep(4,)',100,-1000.,1000.,0.)
      call hbook1(25,'phep(5,)',50,-250.,250.,0.)
      call hbook1(26,'vhep(1,)',10,-1.,1.,0.)
      call hbook1(27,'vhep(2,)',10,-1.,1.,0.)
      call hbook1(28,'vhep(3,)',10,-1.,1.,0.)
      call hbook1(29,'vhep(4,)',10,0.,1.,0.)
      call hbook1(30,'nmulti',20,0.,20.,0.)
      call hbook1(31,'nevmulti',100,0.0,100.0,0.)
      call hbook1(32,'itrkmulti',100,0.0,1000.0,0.)
      call hbook1(33,'mltstr',20,0.,20.,0.)
      return
      end


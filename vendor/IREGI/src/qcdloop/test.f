       implicit none
       double complex qlI1,qlI2,qlI3,qlI4,Ival(-2:0)
       double precision p1sq,p2sq,p3sq,p4sq,s12,s23
       double precision m1sq,m2sq,m3sq,m4sq,musq
       integer ep
       call qlinit
       musq=1.e6
       open(unit=301,file="scalar.tmp")
       m1sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=30625.
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=159734.90989225
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=263235.89945104
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=295638.72429289005
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=297274.1172122501
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=323125.28416921
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       m1sq=343145.00348164
       do ep=0,-2,-1
          Ival(ep)=qlI1(m1sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=-449282.61956417596
       m1sq=0.
       m2sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=-449282.61956417596
       m1sq=0.
       m2sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=-449282.61956417596
       m1sq=0.
       m2sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=-218256.45810044397
       m1sq=0.
       m2sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=-218256.45810044397
       m1sq=0.
       m2sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=-218256.45810044397
       m1sq=0.
       m2sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       m1sq=166230.46116769002
       m2sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       m1sq=166230.46116769002
       m2sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=0.
       m2sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=0.
       m2sq=263235.89945104
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=0.
       m2sq=295638.72429289005
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=0.
       m2sq=297274.1172122501
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=0.
       m2sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=0.
       m2sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=0.
       m2sq=323125.28416921
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=30625.
       m2sq=159734.90989225
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       m1sq=30625.
       m2sq=343145.00348164
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=0.
       m2sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=30625.
       m2sq=30625.
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=159734.90989225
       m2sq=159734.90989225
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=166230.46116769002
       m2sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=263235.89945104
       m2sq=263235.89945104
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=295638.72429289005
       m2sq=295638.72429289005
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=297274.1172122501
       m2sq=297274.1172122501
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=301685.77863649005
       m2sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=314854.532161
       m2sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=323125.28416921
       m2sq=323125.28416921
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=1.e6
       m1sq=343145.00348164
       m2sq=343145.00348164
       do ep=0,-2,-1
          Ival(ep)=qlI2(p1sq,m1sq,m2sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=1.e6
       m1sq=0.
       m2sq=0.
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=1.e6
       m1sq=166230.46116769002
       m2sq=301685.77863649005
       m3sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=1.e6
       m1sq=166230.46116769002
       m2sq=314854.532161
       m3sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=1.e6
       m1sq=301685.77863649005
       m2sq=166230.46116769002
       m3sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=1.e6
       m1sq=314854.532161
       m2sq=166230.46116769002
       m3sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-449282.61956417596
       m1sq=166230.46116769002
       m2sq=301685.77863649005
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-449282.61956417596
       m1sq=166230.46116769002
       m2sq=314854.532161
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-449282.61956417596
       m1sq=301685.77863649005
       m2sq=166230.46116769002
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-449282.61956417596
       m1sq=314854.532161
       m2sq=166230.46116769002
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-218256.45810044397
       m1sq=0.
       m2sq=0.
       m3sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-218256.45810044397
       m1sq=0.
       m2sq=0.
       m3sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-218256.45810044397
       m1sq=0.
       m2sq=0.
       m3sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-218256.45810044397
       m1sq=166230.46116769002
       m2sq=301685.77863649005
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-218256.45810044397
       m1sq=166230.46116769002
       m2sq=314854.532161
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-218256.45810044397
       m1sq=301685.77863649005
       m2sq=166230.46116769002
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=-218256.45810044397
       m1sq=314854.532161
       m2sq=166230.46116769002
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=0.
       m2sq=166230.46116769002
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=0.
       m2sq=263235.89945104
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=0.
       m2sq=295638.72429289005
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=0.
       m2sq=297274.1172122501
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=0.
       m2sq=301685.77863649005
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=0.
       m2sq=314854.532161
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=0.
       m2sq=323125.28416921
       m3sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=30625.
       m2sq=159734.90989225
       m3sq=30625.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=30625.
       m2sq=343145.00348164
       m3sq=30625.
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=159734.90989225
       m2sq=30625.
       m3sq=159734.90989225
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=166230.46116769002
       m2sq=0.
       m3sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=263235.89945104
       m2sq=0.
       m3sq=263235.89945104
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=295638.72429289005
       m2sq=0.
       m3sq=295638.72429289005
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=297274.1172122501
       m2sq=0.
       m3sq=297274.1172122501
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=301685.77863649005
       m2sq=0.
       m3sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=314854.532161
       m2sq=0.
       m3sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=323125.28416921
       m2sq=0.
       m3sq=323125.28416921
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=166230.46116769002
       p2sq=166230.46116769002
       p3sq=1.e6
       m1sq=343145.00348164
       m2sq=30625.
       m3sq=343145.00348164
       do ep=0,-2,-1
          Ival(ep)=qlI3(p1sq,p2sq,p3sq,m1sq,m2sq,m3sq,musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=0.
       m4sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=0.
       m4sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=0.
       m4sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-449282.61956417596
       m1sq=166230.46116769002
       m2sq=301685.77863649005
       m3sq=166230.46116769002
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-449282.61956417596
       m1sq=166230.46116769002
       m2sq=314854.532161
       m3sq=166230.46116769002
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-449282.61956417596
       m1sq=301685.77863649005
       m2sq=166230.46116769002
       m3sq=301685.77863649005
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-449282.61956417596
       m1sq=314854.532161
       m2sq=166230.46116769002
       m3sq=314854.532161
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-218256.45810044397
       m1sq=0.
       m2sq=0.
       m3sq=0.
       m4sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-218256.45810044397
       m1sq=0.
       m2sq=0.
       m3sq=0.
       m4sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-218256.45810044397
       m1sq=0.
       m2sq=0.
       m3sq=0.
       m4sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-218256.45810044397
       m1sq=166230.46116769002
       m2sq=301685.77863649005
       m3sq=166230.46116769002
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-218256.45810044397
       m1sq=166230.46116769002
       m2sq=314854.532161
       m3sq=166230.46116769002
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-218256.45810044397
       m1sq=301685.77863649005
       m2sq=166230.46116769002
       m3sq=301685.77863649005
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=0.
       p3sq=166230.46116769002
       p4sq=166230.46116769002
       s12=1.e6
       s23=-218256.45810044397
       m1sq=314854.532161
       m2sq=166230.46116769002
       m3sq=314854.532161
       m4sq=0.
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=0.
       p4sq=166230.46116769002
       s12=-218256.45810044397
       s23=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=166230.46116769002
       m4sq=301685.77863649005
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=0.
       p4sq=166230.46116769002
       s12=-218256.45810044397
       s23=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=166230.46116769002
       m4sq=314854.532161
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=0.
       p4sq=166230.46116769002
       s12=-218256.45810044397
       s23=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=301685.77863649005
       m4sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       p1sq=0.
       p2sq=166230.46116769002
       p3sq=0.
       p4sq=166230.46116769002
       s12=-218256.45810044397
       s23=-449282.61956417596
       m1sq=0.
       m2sq=0.
       m3sq=314854.532161
       m4sq=166230.46116769002
       do ep=0,-2,-1
          Ival(ep)=qlI4(p1sq,p2sq,p3sq,p4sq,s12,s23,m1sq,m2sq,m3sq,m4sq,
     $          musq,ep)
          write(301,*)dreal(Ival(ep))
          write(301,*)dimag(Ival(ep))
       enddo
       close(unit=301,status="KEEP")
       end
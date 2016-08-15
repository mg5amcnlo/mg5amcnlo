

      subroutine avh_olo_mu_set(mu)
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      real(rkind) ,intent(in) :: mu
      call olo_scale( mu )
      end subroutine

      subroutine avh_olo_onshell(thrs)
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      real(rkind) ,intent(in) :: thrs
      call olo_onshell( thrs )
      end subroutine

      subroutine avh_olo_unit( unit_in )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      integer ,intent(in) :: unit_in
      call olo_unit( unit_in ,'all' )
      end subroutine

      subroutine avh_olo_printall( unit_in )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      integer ,intent(in) :: unit_in
      call olo_unit( unit_in ,'printall' )
      end subroutine

      subroutine avh_olo_a0c( rslt ,mm )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      complex(rkind) ,intent(in)  :: mm
      call olo( rslt ,mm )
      end subroutine

      subroutine avh_olo_b0c( rslt ,pp,m1,m2 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      complex(rkind) ,intent(in)  :: pp,m1,m2
      call olo( rslt ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_b11c( b11,b00,b1,b0 ,pp,m1,m2 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
      complex(rkind) ,intent(in)  :: pp,m1,m2
      call olo( b11,b00,b1,b0 ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_c0c( rslt ,p1,p2,p3 ,m1,m2,m3 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      complex(rkind) ,intent(in)  :: p1,p2,p3 ,m1,m2,m3
      call olo( rslt ,p1,p2,p3 ,m1,m2,m3 )
      end subroutine

      subroutine avh_olo_d0c( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      complex(rkind) ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
      call olo( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      end subroutine

      subroutine avh_olo_a0m( rslt ,mm )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      real(rkind)    ,intent(in)  :: mm
      call olo( rslt ,mm )
      end subroutine

      subroutine avh_olo_b0m( rslt ,pp,m1,m2 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      real(rkind)    ,intent(in)  :: pp,m1,m2
      call olo( rslt ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_b11m( b11,b00,b1,b0 ,pp,m1,m2 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
      real(rkind)    ,intent(in)  :: pp,m1,m2
      call olo( b11,b00,b1,b0 ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_c0m( rslt ,p1,p2,p3 ,m1,m2,m3 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      real(rkind)    ,intent(in)  :: p1,p2,p3 ,m1,m2,m3
      call olo( rslt ,p1,p2,p3 ,m1,m2,m3 )
      end subroutine

      subroutine avh_olo_d0m( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      use avh_olo ,rkind=>olo_dp_kind
      implicit none
      complex(rkind) ,intent(out) :: rslt(0:2)
      real(rkind)    ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
      call olo( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      end subroutine

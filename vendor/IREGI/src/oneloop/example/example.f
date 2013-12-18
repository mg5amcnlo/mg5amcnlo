      call test
      end


      subroutine test
!  ********************************************************************
!  ********************************************************************
      use avh_olo
      implicit none
      integer :: nn
      character(4) :: next
      real(kind(1d0)) :: mu
!
      call ltini !|LTVSNge26=yesyes
!#      call ffini !|LTVSNge26=yesno
!      call olo_onshell( 1d-6 )
!      call olo_unit( 6 ,'printall' )
!
      next = 'next'
      do while (next.eq.'next')
        read(5,*) nn ,mu
        call olo_scale( mu )
        call setmudim( mu*mu ) !|LT=yes
        if     (nn.eq.2) then
          call test_2
        elseif (nn.eq.3) then
          call test_3
        else!if(nn.eq.4) then
          call test_4
        endif
        read(5,*) next
      enddo
      end


      subroutine test_4
!  ********************************************************************
!  ********************************************************************
      use avh_olo
      implicit none
      complex(kind(1d0)) :: p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
      complex(kind(1d0)) :: zolo(0:2),zLT,D0C
!
      read(5,*) p1
      read(5,*) p2
      read(5,*) p3
      read(5,*) p4
      read(5,*) p12
      read(5,*) p23
      read(5,*) m1
      read(5,*) m2
      read(5,*) m3
      read(5,*) m4
      write(6,101) 'p1 : (',real(p1),',',aimag(p1),')'
      write(6,101) 'p2 : (',real(p2),',',aimag(p2),')'
      write(6,101) 'p3 : (',real(p3),',',aimag(p3),')'
      write(6,101) 'p4 : (',real(p4),',',aimag(p4),')'
      write(6,101) 'p12: (',real(p12),',',aimag(p12),')'
      write(6,101) 'p23: (',real(p23),',',aimag(p23),')'
      write(6,101) 'm1 : (',real(m1),',',aimag(m1),')'
      write(6,101) 'm2 : (',real(m2),',',aimag(m2),')'
      write(6,101) 'm3 : (',real(m3),',',aimag(m3),')'
      write(6,101) 'm4 : (',real(m4),',',aimag(m4),')'
  101 format(a6,d31.24,a1,d31.24,a1)
!
      call olo( zolo ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
!
      zLT = 0
      if (zolo(1).eq.0)                               !|LT=yes
     &  zLT = D0C( p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ) !|LT=yes
!
      write(6,'(a10)') ' eps^( 0):'
      if (zLT.ne.0)
     &write(6,'(a8,2d24.16)') '     LT:',zLT
      write(6,'(a8,2d24.16)') '    olo:',zolo(0)
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(1)
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(2)
      write(6,*)
!
      end
   
      subroutine test_3
!  ********************************************************************
!  ********************************************************************
      use avh_olo
      implicit none
      complex(kind(1d0)) :: p1,p2,p3,m1,m2,m3 ,zolo(0:2),zLT,C0C
!
      read(5,*) p1
      read(5,*) p2
      read(5,*) p3
      read(5,*) m1
      read(5,*) m2
      read(5,*) m3
      write(6,101) 'p1 : (',real(p1),',',aimag(p1),')'
      write(6,101) 'p2 : (',real(p2),',',aimag(p2),')'
      write(6,101) 'p3 : (',real(p3),',',aimag(p3),')'
      write(6,101) 'm1 : (',real(m1),',',aimag(m1),')'
      write(6,101) 'm2 : (',real(m2),',',aimag(m2),')'
      write(6,101) 'm3 : (',real(m3),',',aimag(m3),')'
  101 format(a6,d31.24,a1,d31.24,a1)
!
      call olo( zolo ,p1,p2,p3 ,m1,m2,m3 )
!
      zLT = 0
      if (zolo(1).eq.0) zLT = C0C( p1,p2,p3 ,m1,m2,m3 ) !|LT=yes
!
      write(6,'(a10)') ' eps^( 0):'
      if (zLT.ne.0)
     &write(6,'(a8,2d24.16)') '     LT:',zLT
      write(6,'(a8,2d24.16)') '    olo:',zolo(0)
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(1)
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(2)
      write(6,*)
!
      end

      subroutine test_2
!  ********************************************************************
!  ********************************************************************
      use avh_olo
      implicit none
      complex(kind(1d0)) :: p1,m1,m2 ,b0(0:2),b1(0:2),b00(0:2),b11(0:2)
      complex(kind(1d0)) :: rslt(0:2)
      complex(kind(1d0)) :: b0LT,b1LT,b00LT,b11LT,b0C,b1C,b00C,b11C
!
      read(5,*) p1
      read(5,*) m1
      read(5,*) m2
      write(6,101) 'p1 : (',real(p1),',',aimag(p1),')'
      write(6,101) 'm1 : (',real(m1),',',aimag(m1),')'
      write(6,101) 'm2 : (',real(m2),',',aimag(m2),')'
  101 format(a6,d31.24,a1,d31.24,a1)
!
      call olo( rslt ,p1 ,m1,m2 )
      call olo( b11,b00,b1,b0 ,p1 ,m1,m2 )
      b0LT  = B0C(  p1 ,m1,m2 ) !|LT=yes
      b1LT  = B1C(  p1 ,m1,m2 ) !|LT=yes
      b00LT = B00C( p1 ,m1,m2 ) !|LT=yes
      b11LT = B11C( p1 ,m1,m2 ) !|LT=yes
!
      write(6,'(a10)') ' eps^( 0):'
      write(6,'(a8,2d24.16)') 'b0   LT:',b0LT !|LT=yes
      write(6,'(a8,2d24.16)') '    olo:',rslt(0)
      write(6,'(a8,2d24.16)') '    olo:',b0(0)
      write(6,'(a8,2d24.16)') 'b1   LT:',b1LT !|LT=yes
      write(6,'(a8,2d24.16)') '    olo:',b1(0)
      write(6,'(a8,2d24.16)') 'b00  LT:',b00LT !|LT=yes
      write(6,'(a8,2d24.16)') '    olo:',b00(0)
      write(6,'(a8,2d24.16)') 'b11  LT:',b11LT !|LT=yes
      write(6,'(a8,2d24.16)') '    olo:',b11(0)
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',b0(1)
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',b0(2)
      write(6,*)
!
      end

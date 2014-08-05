      subroutine f_main
!  ********************************************************************
!  ********************************************************************
      use qdmodule
      use avh_olo
      implicit none
      integer :: nn
      character(4) :: next
      real(kind(1d0)) :: mu
!
!      call olo_onshell( 1d-6 )
!      call olo_unit( 6 ,'printall' )
!
      next = 'next'
      do while (next.eq.'next')
        read(5,*) nn ,mu
        call olo_scale( mu )
        if     (nn.eq.2) then
          call test_2
        elseif (nn.eq.3) then
          call test_3
        else!if(nn.eq.4) then
          call test_4
        endif
        read(5,*) next
      enddo
      end subroutine


      subroutine test_4
!  ********************************************************************
!  ********************************************************************
      use qdmodule
      use avh_olo
      implicit none
      complex(kind(1d0)) :: p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,zdp(0:2)
      type(qd_complex) :: p1_qd,p2_qd,p3_qd,p4_qd,p12_qd,p23_qd
      type(qd_complex) :: m1_qd,m2_qd,m3_qd,m4_qd,zqd(0:2)
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
      p1_qd =p1
      p2_qd =p2
      p3_qd =p3
      p4_qd =p4
      p12_qd=p12
      p23_qd=p23
      m1_qd =m1
      m2_qd =m2
      m3_qd =m3
      m4_qd =m4
      call olo( zdp ,p1,p2,p3,p4,p12,p23  ,m1,m2,m3,m4 )
      call olo( zqd ,p1_qd,p2_qd,p3_qd,p4_qd,p12_qd,p23_qd &
                    ,m1_qd,m2_qd,m3_qd,m4_qd )
!
      write(6,'(a10)') ' eps^( 0):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(0)
      call qdwrite(6,qdreal(zqd(0)))
      call qdwrite(6,aimag(zqd(0)))
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(1)
      call qdwrite(6,qdreal(zqd(1)))
      call qdwrite(6,aimag(zqd(1)))
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(2)
      call qdwrite(6,qdreal(zqd(2)))
      call qdwrite(6,aimag(zqd(2)))
      write(6,*)
!
      end subroutine
   
      subroutine test_3
!  ********************************************************************
!  ********************************************************************
      use qdmodule
      use avh_olo
      implicit none
      complex(kind(1d0)) :: p1,p2,p3,m1,m2,m3,zdp(0:2)
      type(qd_complex) :: p1_qd,p2_qd,p3_qd
      type(qd_complex) :: m1_qd,m2_qd,m3_qd,zqd(0:2)
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
      p1_qd=p1
      p2_qd=p2
      p3_qd=p3
      m1_qd=m1
      m2_qd=m2
      m3_qd=m3
      call olo( zdp ,p1,p2,p3 ,m1,m2,m3 )
      call olo( zqd ,p1_qd,p2_qd,p3_qd ,m1_qd,m2_qd,m3_qd )
!
      write(6,'(a10)') ' eps^( 0):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(0)
      call qdwrite(6,qdreal(zqd(0)))
      call qdwrite(6,aimag(zqd(0)))
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(1)
      call qdwrite(6,qdreal(zqd(1)))
      call qdwrite(6,aimag(zqd(1)))
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(2)
      call qdwrite(6,qdreal(zqd(2)))
      call qdwrite(6,aimag(zqd(2)))
      write(6,*)
!
      end subroutine

      subroutine test_2
!  ********************************************************************
!  ********************************************************************
      use qdmodule
      use avh_olo
      implicit none
      complex(kind(1d0)) :: p1,m1,m2,zdp(0:2)
      type(qd_complex) :: m1_qd,m2_qd,p1_qd,zqd(0:2)
!
      read(5,*) p1
      read(5,*) m1
      read(5,*) m2
      write(6,101) 'p1 : (',real(p1),',',aimag(p1),')'
      write(6,101) 'm1 : (',real(m1),',',aimag(m1),')'
      write(6,101) 'm2 : (',real(m2),',',aimag(m2),')'
  101 format(a6,d31.24,a1,d31.24,a1)
!
      p1_qd = p1
      m1_qd = m1
      m2_qd = m2
      call olo( zdp ,p1 ,m1,m2 )
      call olo( zqd ,p1_qd ,m1_qd,m2_qd )
!
      write(6,'(a10)') ' eps^( 0):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(0)
      call qdwrite(6,qdreal(zqd(0)))
      call qdwrite(6,aimag(zqd(0)))
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(1)
      call qdwrite(6,qdreal(zqd(1)))
      call qdwrite(6,aimag(zqd(1)))
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zdp(2)
      call qdwrite(6,qdreal(zqd(2)))
      call qdwrite(6,aimag(zqd(2)))
      write(6,*)
!
      end subroutine

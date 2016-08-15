      call test
      end


      subroutine test
!  ********************************************************************
!  ********************************************************************
      use avh_olo
      implicit none
      double complex rslt(0:2)
      integer nn
      character(4) next
      double precision mu
!
      call ltini !LTyes
!      call olo_onshell( 1d-6 )
      call olo_a0(rslt,0d0)
      call avh_oni_a0m(rslt,0d0)
!      call avh_oni_unitcuba_set( 0 )
      call avh_oni_maxeval_set( 10 000 000 )
!
      next = 'next'
      do while (next.eq.'next')
        read(5,*) nn ,mu
        call olo_scale( mu )
        call avh_oni_mu_set( mu )
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
      double complex p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
     &,zolo(0:2),zoni(0:2),zLT,ie,D0C
      parameter( ie=(0d0,0d-16) )
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
      write(6,101) 'p1 : (',dreal(p1),',',dimag(p1),')'
      write(6,101) 'p2 : (',dreal(p2),',',dimag(p2),')'
      write(6,101) 'p3 : (',dreal(p3),',',dimag(p3),')'
      write(6,101) 'p4 : (',dreal(p4),',',dimag(p4),')'
      write(6,101) 'p12: (',dreal(p12),',',dimag(p12),')'
      write(6,101) 'p23: (',dreal(p23),',',dimag(p23),')'
      write(6,101) 'm1 : (',dreal(m1),',',dimag(m1),')'
      write(6,101) 'm2 : (',dreal(m2),',',dimag(m2),')'
      write(6,101) 'm3 : (',dreal(m3),',',dimag(m3),')'
      write(6,101) 'm4 : (',dreal(m4),',',dimag(m4),')'
  101 format(a6,d31.24,a1,d31.24,a1)
!
      call olo_d0( zolo ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
!
      write(6,*) 'Executing avh_oni_d0c...'
      call avh_oni_d0c( zoni ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
!
      zLT = dcmplx(0d0)
      if (zolo(1).eq.dcmplx(0d0))                                 !LTyes
     &  zLT = D0C( p1,p2,p3,p4,p12,p23 ,m1-ie,m2-ie,m3-ie,m4-ie ) !LTyes
!
      write(6,'(a10)') ' eps^( 0):'
      if (zLT.ne.dcmplx(0d0))
     &write(6,'(a8,2d24.16)') '     LT:',zLT
      write(6,'(a8,2d24.16)') '    olo:',zolo(0)
      write(6,'(a8,2d24.16)') '    oni:',zoni(0)
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(1)
      write(6,'(a8,2d24.16)') '    oni:',zoni(1)
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(2)
      write(6,'(a8,2d24.16)') '    oni:',zoni(2)
      write(6,*)
!
      end
   
      subroutine test_3
!  ********************************************************************
!  ********************************************************************
      use avh_olo
      implicit none
      double complex p1,p2,p3,m1,m2,m3 ,zolo(0:2),zoni(0:2),zLT,C0C,ie
      parameter( ie=(0d0,0d-16) )
!
      read(5,*) p1
      read(5,*) p2
      read(5,*) p3
      read(5,*) m1
      read(5,*) m2
      read(5,*) m3
      write(6,101) 'p1 : (',dreal(p1),',',dimag(p1),')'
      write(6,101) 'p2 : (',dreal(p2),',',dimag(p2),')'
      write(6,101) 'p3 : (',dreal(p3),',',dimag(p3),')'
      write(6,101) 'm1 : (',dreal(m1),',',dimag(m1),')'
      write(6,101) 'm2 : (',dreal(m2),',',dimag(m2),')'
      write(6,101) 'm3 : (',dreal(m3),',',dimag(m3),')'
  101 format(a6,d31.24,a1,d31.24,a1)
!
      call olo_c0( zolo ,p1,p2,p3 ,m1,m2,m3 )
!
      write(6,*) 'Executing avh_oni_c0c...'
      call avh_oni_c0c( zoni ,p1,p2,p3 ,m1,m2,m3 )
!
      zLT = dcmplx(0d0)
      if (zolo(1).eq.dcmplx(0d0))                !LTyes
     &  zLT = C0C( p1,p2,p3 ,m1-ie,m2-ie,m3-ie ) !LTyes
!
      write(6,'(a10)') ' eps^( 0):'
      if (zLT.ne.dcmplx(0d0))
     &write(6,'(a8,2d24.16)') '     LT:',zLT
      write(6,'(a8,2d24.16)') '    olo:',zolo(0)
      write(6,'(a8,2d24.16)') '    oni:',zoni(0)
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(1)
      write(6,'(a8,2d24.16)') '    oni:',zoni(1)
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(2)
      write(6,'(a8,2d24.16)') '    oni:',zoni(2)
      write(6,*)
!
      end

      subroutine test_2
!  ********************************************************************
!  ********************************************************************
      use avh_olo
      implicit none
      double complex p1,m1,m2 ,zolo(0:2),zoni(0:2),ie
      parameter( ie=(0d0,0d-16) )
!
      read(5,*) p1
      read(5,*) m1
      read(5,*) m2
      write(6,101) 'p1 : (',dreal(p1),',',dimag(p1),')'
      write(6,101) 'm1 : (',dreal(m1),',',dimag(m1),')'
      write(6,101) 'm2 : (',dreal(m2),',',dimag(m2),')'
  101 format(a6,d31.24,a1,d31.24,a1)
!
      call olo_b0( zolo ,p1 ,m1,m2 )
!
      call avh_oni_b0c( zoni ,p1 ,m1,m2 )
!
      write(6,'(a10)') ' eps^( 0):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(0)
      write(6,'(a8,2d24.16)') '    oni:',zoni(0)
      write(6,'(a10)') ' eps^(-1):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(1)
      write(6,'(a8,2d24.16)') '    oni:',zoni(1)
      write(6,'(a10)') ' eps^(-2):'
      write(6,'(a8,2d24.16)') '    olo:',zolo(2)
      write(6,'(a8,2d24.16)') '    oni:',zoni(2)
      write(6,*)
!
      end

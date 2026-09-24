program check_born_sectors
  use mint_module
  use weight_lines
  implicit none
  integer :: pdg_type_d(3,5),fks_i_d(3)
  common/test_fks_info/pdg_type_d,fks_i_d
  integer :: iproc_save(3),eto(2,3),etoi(2,3),maxproc_found
  common/cproc_combination/iproc_save,eto,etoi,maxproc_found
  integer :: ix,iy,sector
  double precision :: b(2),c(2),v(2),expected,mult
  character(len=16) :: arg

  call born_spread_configure(.true.,5,2,3,7)
  call get_command_argument(1,arg)
  if (trim(arg).eq.'load') then
     call born_spread_load_table
     stop
  endif
  ndim=7
  ifold=1
  imode=0
  only_virt=.false.
  call weight_lines_allocated(5,5,1,2)
  icontr=3
  niproc=2
  H_event=.false.
  ifold_cnt=1
  itype(1:3)=[2,3,14]
  iproc_save=2
  eto(1,:)=1
  eto(2,:)=2
  pdg_type_d=21
  fks_i_d=5
  born_spread_phase=1
  ! Interleave two sectors with opposite optima and different Born sizes.
  ! Sector 3 has no samples and must retain a unit table.
  do iy=1,born_spread_ny
     do ix=1,born_spread_nxi
        call born_spread_set_point((ix-0.5d0)/born_spread_nxi,(iy-0.5d0)/born_spread_ny)
        do sector=1,2
           born_spread_current_sector=sector
           nFKS=sector
           mult=10d0**(sector-1)
           b=mult*[1d0,0.5d0]
           c=0d0
           if ((ix.le.born_spread_nxi/2).eqv.(sector.eq.1)) c=-2d0*b
           v=mult*[100d0,-50d0]
           parton_iproc(1:2,1)=b
           parton_iproc(1:2,2)=c
           parton_iproc(1:2,3)=v
           call sum_identical_contributions
           if (any(unwgt_B(1:2,1).ne.b)) stop 1
           if (any(unwgt_noB(1:2,1).ne.c)) stop 2
           if (maxval(abs(unwgt(1:2,1)-(b+c+v))).gt.1d-12) stop 3
           call born_spread_observe_sample(unwgt_B(:,1),unwgt_noB(:,1),2)
        enddo
     enddo
  enddo
  born_spread_phase=0
  call solve_born_spreading_table
  call check_tables
  call born_spread_write_table
  born_spread_factor=-1d0
  call born_spread_load_table
  call check_tables

  ! Generation still excludes the virtual stream only at imode=1.
  born_spread_phase=2
  imode=1
  call sum_identical_contributions
  if (maxval(abs(unwgt(1:2,1)-(b+c))).gt.1d-12) stop 4
  if (any(unwgt_noB(1:2,1).ne.c)) stop 5

  ! Saved Born sector/fold controls application after later maps changed it.
  born_spread_phase=3
  icontr=5
  itype(1:5)=[2,3,14,2,2]
  H_event(4)=.true.
  ifold_cnt(5)=2
  wgt=1d0
  born_spread_bin_fold(1:2)=1
  born_spread_sector_fold(1:2)=[1,2]
  born_spread_current_sector=3
  born_spread_current_bin=1600
  call apply_born_spread_weight(1)
  if (maxval(abs(wgt(1:3,1)-2d0)).gt.1d-10) stop 6
  if (any(wgt(1:3,2:5).ne.1d0)) stop 7
  born_spread_current_sector=3
  call apply_born_spread_weight(2)
  if (maxval(abs(wgt(1:3,5))).gt.1d-10) stop 8
  if (any(wgt(1:3,2:4).ne.1d0)) stop 9

  ! Decimal coefficients leave rounding residues after all slopes cancel.
  ! Every bin has a flat minimum containing one: no bin should get a spike.
  call born_spread_configure(.true.,5,2,3,7)
  born_spread_phase=1
  do ix=1,born_spread_nxi*born_spread_ny
     born_spread_current_bin=ix
     b=[0.1d0,0.2d0]
     c=-b*[0.25d0,0.5d0]
     if (mod(ix,2).eq.0) c=-b*[0.5d0,0.25d0]
     call born_spread_observe_sample(b,c,2)
     b=0.3d0
     c=-0.75d0*b
     call born_spread_observe_sample(b,c,1)
  enddo
  call solve_born_spreading_table
  if (maxval(abs(born_spread_factor-1d0)).gt.1d-10) stop 12
  write(*,*) 'PASS sector fits and S grouping'

contains
  subroutine check_tables
    do sector=1,3
       if (abs(born_spread_normalization(sector)-1d0).gt.1d-10) stop 10
       do ix=1,born_spread_nxi
          expected=0d0
          if ((ix.le.born_spread_nxi/2).eqv.(sector.eq.1)) expected=2d0
          if (sector.eq.3) expected=1d0
          if (maxval(abs(born_spread_factor(ix,:,sector)-expected)).gt.1d-10) stop 11
       enddo
    enddo
  end subroutine check_tables
end program check_born_sectors

double precision function ran2()
  stop 99
end function ran2

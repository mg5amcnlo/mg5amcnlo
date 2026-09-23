      subroutine mc_born_local(p,request,result)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      double precision p(0:3,nexternal-1)
      type(BornRequest) request
      type(BornResult) result
      type(BornModelState),save::state
      integer nr,nc,status,nfksprocess
      common/c_nfksprocess/nfksprocess
      double precision charges(nexternal-1)
      common/c_charges_born/charges
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      call born_model_dimensions(nr,nc)
      call born_resize_model_state(state,nr,nc)
      state%%real_values(1)=g
      state%%real_values(2)=all_g
      state%%complex_values(1:2)=gal
      state%%real_values(3)=mu_r
      state%%real_values(4)=all_mu_r
      state%%real_values(5)=mdl_mb
      state%%real_values(6)=mdl_mh
      state%%real_values(7)=mdl_mt
      state%%real_values(8)=mdl_mta
      state%%real_values(9)=mdl_mw
      state%%real_values(10)=mdl_mz
      state%%real_values(11)=mdl_wh
      state%%real_values(12)=mdl_wt
      state%%real_values(13)=mdl_ww
      state%%real_values(14)=mdl_wz
      state%%complex_values(3)=gc_10
      state%%complex_values(4)=gc_11
      state%%complex_values(5)=gc_1
      state%%complex_values(6)=gc_2
      state%%complex_values(7)=gc_50
      state%%complex_values(8)=gc_58
      state%%real_values(15)=mdl_sqrt__as
      state%%real_values(16)=mdl_g__exp__2
      state%%real_values(17)=mdl_conjg__ckm3x3
      state%%real_values(18)=mdl_conjg__ckm1x1
      state%%real_values(19)=mdl_ckm3x3
      state%%real_values(20)=mdl_mz__exp__2
      state%%real_values(21)=mdl_mz__exp__4
      state%%real_values(22)=mdl_sqrt__2
      state%%real_values(23)=mdl_mh__exp__2
      state%%real_values(24)=mdl_aew
      state%%real_values(25)=mdl_sqrt__aew
      state%%real_values(26)=mdl_ee
      state%%real_values(27)=mdl_mw__exp__2
      state%%real_values(28)=mdl_sw2
      state%%real_values(29)=mdl_cw
      state%%real_values(30)=mdl_sqrt__sw2
      state%%real_values(31)=mdl_sw
      state%%real_values(32)=mdl_g1
      state%%real_values(33)=mdl_gw
      state%%real_values(34)=mdl_vev
      state%%real_values(35)=mdl_vev__exp__2
      state%%real_values(36)=mdl_lam
      state%%real_values(37)=mdl_yb
      state%%real_values(38)=mdl_yt
      state%%real_values(39)=mdl_ytau
      state%%real_values(40)=mdl_muh
      state%%real_values(41)=mdl_ee__exp__2
      state%%real_values(42)=mdl_sw__exp__2
      state%%real_values(43)=mdl_cw__exp__2
      state%%real_values(44)=aewm1
      state%%real_values(45)=mdl_gf
      state%%real_values(46)=as
      state%%real_values(47)=mdl_ymb
      state%%real_values(48)=mdl_ymt
      state%%real_values(49)=mdl_ymtau
      state%%complex_values(9)=mdl_complexi
      state%%complex_values(10)=mdl_i1x33
      state%%complex_values(11)=mdl_i2x33
      state%%complex_values(12)=mdl_i3x33
      state%%complex_values(13)=mdl_i4x33
      request%%sector=nfksprocess
      request%%charges=charges
      call born_evaluate(7,7,p,state,request,result,status)
      if(status.ne.0)then
      write(*,*)'Born support evaluation failed',status
      stop 1
      endif
      end
      subroutine mc_capture_model_state(state)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'coupl.inc'
      type(BornModelState) state
      integer nr,nc
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      call born_model_dimensions(nr,nc)
      call born_resize_model_state(state,nr,nc)
      state%%real_values(1)=g
      state%%real_values(2)=all_g
      state%%complex_values(1:2)=gal
      state%%real_values(3)=mu_r
      state%%real_values(4)=all_mu_r
      state%%real_values(5)=mdl_mb
      state%%real_values(6)=mdl_mh
      state%%real_values(7)=mdl_mt
      state%%real_values(8)=mdl_mta
      state%%real_values(9)=mdl_mw
      state%%real_values(10)=mdl_mz
      state%%real_values(11)=mdl_wh
      state%%real_values(12)=mdl_wt
      state%%real_values(13)=mdl_ww
      state%%real_values(14)=mdl_wz
      state%%complex_values(3)=gc_10
      state%%complex_values(4)=gc_11
      state%%complex_values(5)=gc_1
      state%%complex_values(6)=gc_2
      state%%complex_values(7)=gc_50
      state%%complex_values(8)=gc_58
      state%%real_values(15)=mdl_sqrt__as
      state%%real_values(16)=mdl_g__exp__2
      state%%real_values(17)=mdl_conjg__ckm3x3
      state%%real_values(18)=mdl_conjg__ckm1x1
      state%%real_values(19)=mdl_ckm3x3
      state%%real_values(20)=mdl_mz__exp__2
      state%%real_values(21)=mdl_mz__exp__4
      state%%real_values(22)=mdl_sqrt__2
      state%%real_values(23)=mdl_mh__exp__2
      state%%real_values(24)=mdl_aew
      state%%real_values(25)=mdl_sqrt__aew
      state%%real_values(26)=mdl_ee
      state%%real_values(27)=mdl_mw__exp__2
      state%%real_values(28)=mdl_sw2
      state%%real_values(29)=mdl_cw
      state%%real_values(30)=mdl_sqrt__sw2
      state%%real_values(31)=mdl_sw
      state%%real_values(32)=mdl_g1
      state%%real_values(33)=mdl_gw
      state%%real_values(34)=mdl_vev
      state%%real_values(35)=mdl_vev__exp__2
      state%%real_values(36)=mdl_lam
      state%%real_values(37)=mdl_yb
      state%%real_values(38)=mdl_yt
      state%%real_values(39)=mdl_ytau
      state%%real_values(40)=mdl_muh
      state%%real_values(41)=mdl_ee__exp__2
      state%%real_values(42)=mdl_sw__exp__2
      state%%real_values(43)=mdl_cw__exp__2
      state%%real_values(44)=aewm1
      state%%real_values(45)=mdl_gf
      state%%real_values(46)=as
      state%%real_values(47)=mdl_ymb
      state%%real_values(48)=mdl_ymt
      state%%real_values(49)=mdl_ymtau
      state%%complex_values(9)=mdl_complexi
      state%%complex_values(10)=mdl_i1x33
      state%%complex_values(11)=mdl_i2x33
      state%%complex_values(12)=mdl_i3x33
      state%%complex_values(13)=mdl_i4x33
      end

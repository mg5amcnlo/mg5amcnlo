      subroutine mc_born_local(p,request,result)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      double precision p(0:3,nexternal-1)
      type(BornRequest) request
      type(BornResult) result
      type(BornModelState) state
      integer nr,nc,status,nfksprocess
      common/c_nfksprocess/nfksprocess
      double precision charges(nexternal-1)
      common/c_charges_born/charges
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_R2MIXEDFACTOR_FIN_,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_LHV  ,MDL_CONJG__CKM3X3,MDL_CONJG__CKM22,MDL_CKM3X3,MDL_CKM33  ,MDL_CKM22,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MZ__EXP__2  ,MDL_MZ__EXP__4,MDL_SQRT__2,MDL_MH__EXP__2,MDL_NCOL__EXP__2  ,MDL_MB__EXP__2,MDL_MT__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_V,MDL_V__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP  ,MDL_VECTORZDOWN,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_GW__EXP__2  ,MDL_CW__EXP__2,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_YB__EXP__2  ,MDL_YT__EXP__2,AEWM1,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_R2MIXEDFACTOR_FIN_,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_LHV  ,MDL_CONJG__CKM3X3,MDL_CONJG__CKM22,MDL_CKM3X3,MDL_CKM33  ,MDL_CKM22,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MZ__EXP__2  ,MDL_MZ__EXP__4,MDL_SQRT__2,MDL_MH__EXP__2,MDL_NCOL__EXP__2  ,MDL_MB__EXP__2,MDL_MT__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_V,MDL_V__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP  ,MDL_VECTORZDOWN,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_GW__EXP__2  ,MDL_CW__EXP__2,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_YB__EXP__2  ,MDL_YT__EXP__2,AEWM1,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33,MDL_VECTOR_TBGP,MDL_AXIAL_TBGP,MDL_VECTOR_TBGM  ,MDL_AXIAL_TBGM
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33,MDL_VECTOR_TBGP,MDL_AXIAL_TBGP,MDL_VECTOR_TBGM  ,MDL_AXIAL_TBGM
      call born_model_dimensions(nr,nc)
      allocate(state%%real_values(nr),state%%complex_values(nc))
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
      state%%complex_values(3)=gc_5
      state%%complex_values(4)=r2_bxtw
      state%%complex_values(5)=gc_11
      state%%real_values(15)=mdl_sqrt__as
      state%%real_values(16)=mdl_g__exp__4
      state%%real_values(17)=mdl_g__exp__2
      state%%real_values(18)=mdl_r2mixedfactor_fin_
      state%%real_values(19)=mdl_g__exp__3
      state%%real_values(20)=mdl_mu_r__exp__2
      state%%real_values(21)=mdl_lhv
      state%%real_values(22)=mdl_conjg__ckm3x3
      state%%real_values(23)=mdl_conjg__ckm22
      state%%real_values(24)=mdl_ckm3x3
      state%%real_values(25)=mdl_ckm33
      state%%real_values(26)=mdl_ckm22
      state%%real_values(27)=mdl_ncol
      state%%real_values(28)=mdl_ca
      state%%real_values(29)=mdl_tf
      state%%real_values(30)=mdl_cf
      state%%real_values(31)=mdl_mz__exp__2
      state%%real_values(32)=mdl_mz__exp__4
      state%%real_values(33)=mdl_sqrt__2
      state%%real_values(34)=mdl_mh__exp__2
      state%%real_values(35)=mdl_ncol__exp__2
      state%%real_values(36)=mdl_mb__exp__2
      state%%real_values(37)=mdl_mt__exp__2
      state%%real_values(38)=mdl_aew
      state%%real_values(39)=mdl_sqrt__aew
      state%%real_values(40)=mdl_ee
      state%%real_values(41)=mdl_mw__exp__2
      state%%real_values(42)=mdl_sw2
      state%%real_values(43)=mdl_cw
      state%%real_values(44)=mdl_sqrt__sw2
      state%%real_values(45)=mdl_sw
      state%%real_values(46)=mdl_g1
      state%%real_values(47)=mdl_gw
      state%%real_values(48)=mdl_v
      state%%real_values(49)=mdl_v__exp__2
      state%%real_values(50)=mdl_lam
      state%%real_values(51)=mdl_yb
      state%%real_values(52)=mdl_yt
      state%%real_values(53)=mdl_ytau
      state%%real_values(54)=mdl_muh
      state%%real_values(55)=mdl_axialzup
      state%%real_values(56)=mdl_axialzdown
      state%%real_values(57)=mdl_vectorzup
      state%%real_values(58)=mdl_vectorzdown
      state%%real_values(59)=mdl_vectoraup
      state%%real_values(60)=mdl_vectoradown
      state%%real_values(61)=mdl_vectorwmdxu
      state%%real_values(62)=mdl_axialwmdxu
      state%%real_values(63)=mdl_vectorwpuxd
      state%%real_values(64)=mdl_axialwpuxd
      state%%real_values(65)=mdl_gw__exp__2
      state%%real_values(66)=mdl_cw__exp__2
      state%%real_values(67)=mdl_ee__exp__2
      state%%real_values(68)=mdl_sw__exp__2
      state%%real_values(69)=mdl_yb__exp__2
      state%%real_values(70)=mdl_yt__exp__2
      state%%real_values(71)=aewm1
      state%%real_values(72)=mdl_gf
      state%%real_values(73)=as
      state%%real_values(74)=mdl_ymb
      state%%real_values(75)=mdl_ymt
      state%%real_values(76)=mdl_ymtau
      state%%complex_values(6)=mdl_complexi
      state%%complex_values(7)=mdl_i1x33
      state%%complex_values(8)=mdl_i2x33
      state%%complex_values(9)=mdl_i3x33
      state%%complex_values(10)=mdl_i4x33
      state%%complex_values(11)=mdl_vector_tbgp
      state%%complex_values(12)=mdl_axial_tbgp
      state%%complex_values(13)=mdl_vector_tbgm
      state%%complex_values(14)=mdl_axial_tbgm
      request%%sector=nfksprocess
      request%%charges=charges
      call born_evaluate(2,2,p,state,request,result,status)
      deallocate(state%%real_values,state%%complex_values)
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
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_R2MIXEDFACTOR_FIN_,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_LHV  ,MDL_CONJG__CKM3X3,MDL_CONJG__CKM22,MDL_CKM3X3,MDL_CKM33  ,MDL_CKM22,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MZ__EXP__2  ,MDL_MZ__EXP__4,MDL_SQRT__2,MDL_MH__EXP__2,MDL_NCOL__EXP__2  ,MDL_MB__EXP__2,MDL_MT__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_V,MDL_V__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP  ,MDL_VECTORZDOWN,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_GW__EXP__2  ,MDL_CW__EXP__2,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_YB__EXP__2  ,MDL_YT__EXP__2,AEWM1,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_R2MIXEDFACTOR_FIN_,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_LHV  ,MDL_CONJG__CKM3X3,MDL_CONJG__CKM22,MDL_CKM3X3,MDL_CKM33  ,MDL_CKM22,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MZ__EXP__2  ,MDL_MZ__EXP__4,MDL_SQRT__2,MDL_MH__EXP__2,MDL_NCOL__EXP__2  ,MDL_MB__EXP__2,MDL_MT__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_V,MDL_V__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP  ,MDL_VECTORZDOWN,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_GW__EXP__2  ,MDL_CW__EXP__2,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_YB__EXP__2  ,MDL_YT__EXP__2,AEWM1,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33,MDL_VECTOR_TBGP,MDL_AXIAL_TBGP,MDL_VECTOR_TBGM  ,MDL_AXIAL_TBGM
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33,MDL_VECTOR_TBGP,MDL_AXIAL_TBGP,MDL_VECTOR_TBGM  ,MDL_AXIAL_TBGM
      call born_model_dimensions(nr,nc)
      if(allocated(state%%real_values))deallocate(state%%real_values)
      if(allocated(state%%complex_values))deallocate(state%%complex_values)
      allocate(state%%real_values(nr),state%%complex_values(nc))
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
      state%%complex_values(3)=gc_5
      state%%complex_values(4)=r2_bxtw
      state%%complex_values(5)=gc_11
      state%%real_values(15)=mdl_sqrt__as
      state%%real_values(16)=mdl_g__exp__4
      state%%real_values(17)=mdl_g__exp__2
      state%%real_values(18)=mdl_r2mixedfactor_fin_
      state%%real_values(19)=mdl_g__exp__3
      state%%real_values(20)=mdl_mu_r__exp__2
      state%%real_values(21)=mdl_lhv
      state%%real_values(22)=mdl_conjg__ckm3x3
      state%%real_values(23)=mdl_conjg__ckm22
      state%%real_values(24)=mdl_ckm3x3
      state%%real_values(25)=mdl_ckm33
      state%%real_values(26)=mdl_ckm22
      state%%real_values(27)=mdl_ncol
      state%%real_values(28)=mdl_ca
      state%%real_values(29)=mdl_tf
      state%%real_values(30)=mdl_cf
      state%%real_values(31)=mdl_mz__exp__2
      state%%real_values(32)=mdl_mz__exp__4
      state%%real_values(33)=mdl_sqrt__2
      state%%real_values(34)=mdl_mh__exp__2
      state%%real_values(35)=mdl_ncol__exp__2
      state%%real_values(36)=mdl_mb__exp__2
      state%%real_values(37)=mdl_mt__exp__2
      state%%real_values(38)=mdl_aew
      state%%real_values(39)=mdl_sqrt__aew
      state%%real_values(40)=mdl_ee
      state%%real_values(41)=mdl_mw__exp__2
      state%%real_values(42)=mdl_sw2
      state%%real_values(43)=mdl_cw
      state%%real_values(44)=mdl_sqrt__sw2
      state%%real_values(45)=mdl_sw
      state%%real_values(46)=mdl_g1
      state%%real_values(47)=mdl_gw
      state%%real_values(48)=mdl_v
      state%%real_values(49)=mdl_v__exp__2
      state%%real_values(50)=mdl_lam
      state%%real_values(51)=mdl_yb
      state%%real_values(52)=mdl_yt
      state%%real_values(53)=mdl_ytau
      state%%real_values(54)=mdl_muh
      state%%real_values(55)=mdl_axialzup
      state%%real_values(56)=mdl_axialzdown
      state%%real_values(57)=mdl_vectorzup
      state%%real_values(58)=mdl_vectorzdown
      state%%real_values(59)=mdl_vectoraup
      state%%real_values(60)=mdl_vectoradown
      state%%real_values(61)=mdl_vectorwmdxu
      state%%real_values(62)=mdl_axialwmdxu
      state%%real_values(63)=mdl_vectorwpuxd
      state%%real_values(64)=mdl_axialwpuxd
      state%%real_values(65)=mdl_gw__exp__2
      state%%real_values(66)=mdl_cw__exp__2
      state%%real_values(67)=mdl_ee__exp__2
      state%%real_values(68)=mdl_sw__exp__2
      state%%real_values(69)=mdl_yb__exp__2
      state%%real_values(70)=mdl_yt__exp__2
      state%%real_values(71)=aewm1
      state%%real_values(72)=mdl_gf
      state%%real_values(73)=as
      state%%real_values(74)=mdl_ymb
      state%%real_values(75)=mdl_ymt
      state%%real_values(76)=mdl_ymtau
      state%%complex_values(6)=mdl_complexi
      state%%complex_values(7)=mdl_i1x33
      state%%complex_values(8)=mdl_i2x33
      state%%complex_values(9)=mdl_i3x33
      state%%complex_values(10)=mdl_i4x33
      state%%complex_values(11)=mdl_vector_tbgp
      state%%complex_values(12)=mdl_axial_tbgp
      state%%complex_values(13)=mdl_vector_tbgm
      state%%complex_values(14)=mdl_axial_tbgm
      end

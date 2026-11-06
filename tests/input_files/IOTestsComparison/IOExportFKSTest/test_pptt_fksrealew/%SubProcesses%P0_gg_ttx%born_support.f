      subroutine mc_born_local(p,request,result)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'timing_variables.inc'
      double precision p(0:3,nexternal-1)
      type(BornRequest) request
      type(BornResult) result
      type(BornModelState),save::state
      integer status,nfksprocess
      common/c_nfksprocess/nfksprocess
      double precision charges(nexternal-1)
      common/c_charges_born/charges
      if(timing_sampled)call timing_enter(timing_born)
      call mc_capture_model_state(state)
      request%sector=nfksprocess
      request%charges=charges
      request%helicity_frame=born_active_helicity_frame
      call born_evaluate(5,5,p,state,request,result,status)
      if(status.ne.0)then
      write(*,*)'Born support evaluation failed',status
      stop 1
      endif
      if(timing_sampled)call timing_leave(timing_born)
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
      call mc_prepare_model_configuration(state%configuration_epoch)
      call born_model_dimensions(nr,nc)
      call born_resize_model_state(state,nr,nc)
      state%real_values(1)=g
      state%complex_values(1)=gc_10
      state%complex_values(2)=gc_11
      end
      subroutine mc_prepare_model_configuration(epoch)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'coupl.inc'
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__2,MDL_CONJG__CKM3X3  ,MDL_CONJG__CKM1X1,MDL_CKM3X3,MDL_MZ__EXP__2,MDL_MZ__EXP__4  ,MDL_SQRT__2,MDL_MH__EXP__2,MDL_AEW,MDL_SQRT__AEW,MDL_EE  ,MDL_MW__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW,MDL_G1  ,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YB,MDL_YT,MDL_YTAU  ,MDL_MUH,MDL_EE__EXP__2,MDL_SW__EXP__2,MDL_CW__EXP__2,AEWM1  ,MDL_GF,AS,MDL_YMB,MDL_YMT,MDL_YMTAU
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I1X33,MDL_I2X33,MDL_I3X33  ,MDL_I4X33
      integer(8) epoch,current_revision,mc_born_model_revision
      external mc_born_model_revision
      integer(8),save::saved_revision=-1
      type(BornModelConfiguration),save::configuration,candidate
      integer status
      logical same_configuration
      current_revision=mc_born_model_revision()
      if(saved_revision.ne.current_revision.or.configuration%epoch.eq.0)then
      if(.not.allocated(candidate%real_values))allocate(candidate%real_values(4))
      if(.not.allocated(candidate%complex_values))allocate(candidate%complex_values(7))
      candidate%complex_values(1:2)=gal
      candidate%real_values(1)=mdl_mt
      candidate%real_values(2)=mdl_mz
      candidate%real_values(3)=mdl_wt
      candidate%real_values(4)=mdl_wz
      candidate%complex_values(3)=gc_1
      candidate%complex_values(4)=gc_2
      candidate%complex_values(5)=gc_50
      candidate%complex_values(6)=gc_51
      candidate%complex_values(7)=gc_58
      same_configuration=.false.
      if(configuration%epoch.gt.0)then
      same_configuration=all(candidate%real_values.eq.configuration%real_values).and.all(candidate%complex_values.eq.configuration%complex_values)
      endif
      if(.not.same_configuration)then
      configuration%real_values=candidate%real_values
      configuration%complex_values=candidate%complex_values
      call born_configure_model(configuration,status)
      if(status.ne.0)stop "Invalid native Born model configuration"
      endif
      saved_revision=current_revision
      endif
      if(configuration%epoch.ne.born_active_configuration_epoch)then
      call born_activate_model(configuration,status)
      if(status.ne.0)stop "Cannot reactivate native Born model configuration"
      endif
      epoch=configuration%epoch
      end

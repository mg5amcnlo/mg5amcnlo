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
      call born_evaluate(1,5,p,state,request,result,status)
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
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_YB,MDL_VEP,MDL_LHV  ,MDL_EPSUV,MDL_EPSIR,MDL_CONJG__CKM33,MDL_CONJG__CKM22  ,MDL_CONJG__CKM11,MDL_SCKM33__EXP__2,MDL_SCKM33  ,MDL_SCKM22__EXP__2,MDL_SCKM22,MDL_SCKM11,MDL_QL__EXP__4  ,MDL_QL__EXP__2,MDL_ME__EXP__2,MDL_MU__EXP__2,MDL_MS__EXP__2  ,MDL_MC__EXP__2,MDL_MB__EXP__4,MDL_MB__EXP__2,MDL_LOGSWITCH  ,MDL_GFREDEFINITIONCHOICE,MDL_CMSPARAM,MDL_CKM33,MDL_CKM22  ,MDL_EP,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MW__EXP__2  ,MDL_MZ__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW  ,MDL_MH__EXP__2,MDL_SQRT__2,MDL_I3D,MDL_I3U,MDL_I3L,MDL_I3V  ,MDL_QD,MDL_QU,MDL_QL,MDL_NCOL__EXP__2,MDL_SW__EXP__2  ,MDL_MT__EXP__2,MDL_CW__EXP__2,MDL_MT__EXP__4,MDL_SW__EXP__4  ,MDL_MW__EXP__4,MDL_MT__EXP__3,MDL_MH__EXP__4,MDL_MZ__EXP__4  ,MDL_MT__EXP__6,MDL_MW__EXP__6,MDL_MT__EXP__8,MDL_MW__EXP__8  ,MDL_MT__EXP__10,MDL_MW__EXP__10,MDL_MZ__EXP__6,MDL_SW__EXP__6  ,MDL_SW__EXP__3,MDL_MZ__EXP__3,MDL_CW__EXP__4,MDL_CW__EXP__6  ,MDL_MH__EXP__6,MDL_CW__EXP__3,MDL_SW__EXP__8,MDL_SW__EXP__10  ,MDL_QD__EXP__3,MDL_QD__EXP__2,MDL_I3D__EXP__2,MDL_I3D__EXP__3  ,MDL_QU__EXP__3,MDL_QU__EXP__2,MDL_I3U__EXP__2,MDL_I3U__EXP__3  ,MDL_QL__EXP__3,MDL_I3L__EXP__2,MDL_I3L__EXP__3,MDL_I3V__EXP__3  ,MDL_MW__EXP__3,MDL_QU__EXP__4,MDL_QD__EXP__4,MDL_I3L__EXP__4  ,MDL_I3U__EXP__4,MDL_I3D__EXP__4,MDL_I3V__EXP__4  ,MDL_I3V__EXP__2,MDL_GFBAR,MDL_AEW,AEWM1,MDL_SQRT__AEW,MDL_EE  ,MDL_G1,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YT,MDL_MUH  ,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP,MDL_VECTORZDOWN  ,MDL_AXIALG0UP3,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_EE__EXP__2  ,MDL_EE__EXP__3,MDL_EE__EXP__4,MDL_NTADPOLE,MDL_GF,AS,MDL_YMT
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_YB,MDL_VEP,MDL_LHV  ,MDL_EPSUV,MDL_EPSIR,MDL_CONJG__CKM33,MDL_CONJG__CKM22  ,MDL_CONJG__CKM11,MDL_SCKM33__EXP__2,MDL_SCKM33  ,MDL_SCKM22__EXP__2,MDL_SCKM22,MDL_SCKM11,MDL_QL__EXP__4  ,MDL_QL__EXP__2,MDL_ME__EXP__2,MDL_MU__EXP__2,MDL_MS__EXP__2  ,MDL_MC__EXP__2,MDL_MB__EXP__4,MDL_MB__EXP__2,MDL_LOGSWITCH  ,MDL_GFREDEFINITIONCHOICE,MDL_CMSPARAM,MDL_CKM33,MDL_CKM22  ,MDL_EP,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MW__EXP__2  ,MDL_MZ__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW  ,MDL_MH__EXP__2,MDL_SQRT__2,MDL_I3D,MDL_I3U,MDL_I3L,MDL_I3V  ,MDL_QD,MDL_QU,MDL_QL,MDL_NCOL__EXP__2,MDL_SW__EXP__2  ,MDL_MT__EXP__2,MDL_CW__EXP__2,MDL_MT__EXP__4,MDL_SW__EXP__4  ,MDL_MW__EXP__4,MDL_MT__EXP__3,MDL_MH__EXP__4,MDL_MZ__EXP__4  ,MDL_MT__EXP__6,MDL_MW__EXP__6,MDL_MT__EXP__8,MDL_MW__EXP__8  ,MDL_MT__EXP__10,MDL_MW__EXP__10,MDL_MZ__EXP__6,MDL_SW__EXP__6  ,MDL_SW__EXP__3,MDL_MZ__EXP__3,MDL_CW__EXP__4,MDL_CW__EXP__6  ,MDL_MH__EXP__6,MDL_CW__EXP__3,MDL_SW__EXP__8,MDL_SW__EXP__10  ,MDL_QD__EXP__3,MDL_QD__EXP__2,MDL_I3D__EXP__2,MDL_I3D__EXP__3  ,MDL_QU__EXP__3,MDL_QU__EXP__2,MDL_I3U__EXP__2,MDL_I3U__EXP__3  ,MDL_QL__EXP__3,MDL_I3L__EXP__2,MDL_I3L__EXP__3,MDL_I3V__EXP__3  ,MDL_MW__EXP__3,MDL_QU__EXP__4,MDL_QD__EXP__4,MDL_I3L__EXP__4  ,MDL_I3U__EXP__4,MDL_I3D__EXP__4,MDL_I3V__EXP__4  ,MDL_I3V__EXP__2,MDL_GFBAR,MDL_AEW,AEWM1,MDL_SQRT__AEW,MDL_EE  ,MDL_G1,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YT,MDL_MUH  ,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP,MDL_VECTORZDOWN  ,MDL_AXIALG0UP3,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_EE__EXP__2  ,MDL_EE__EXP__3,MDL_EE__EXP__4,MDL_NTADPOLE,MDL_GF,AS,MDL_YMT
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I233,MDL_I333,MDL_VECTORHUP3  ,MDL_VECTORGPUX3D3,MDL_AXIALGPUX3D3,MDL_VECTORGMDX3U3  ,MDL_AXIALGMDX3U3
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I233,MDL_I333,MDL_VECTORHUP3  ,MDL_VECTORGPUX3D3,MDL_AXIALGPUX3D3,MDL_VECTORGMDX3U3  ,MDL_AXIALGMDX3U3
      call mc_prepare_model_configuration(state%configuration_epoch)
      call born_model_dimensions(nr,nc)
      call born_resize_model_state(state,nr,nc)
      state%real_values(1)=g
      end
      subroutine mc_prepare_model_configuration(epoch)
      use mc_born_support
      use mc_born_types
      implicit none
      include 'coupl.inc'
      DOUBLE PRECISION MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_YB,MDL_VEP,MDL_LHV  ,MDL_EPSUV,MDL_EPSIR,MDL_CONJG__CKM33,MDL_CONJG__CKM22  ,MDL_CONJG__CKM11,MDL_SCKM33__EXP__2,MDL_SCKM33  ,MDL_SCKM22__EXP__2,MDL_SCKM22,MDL_SCKM11,MDL_QL__EXP__4  ,MDL_QL__EXP__2,MDL_ME__EXP__2,MDL_MU__EXP__2,MDL_MS__EXP__2  ,MDL_MC__EXP__2,MDL_MB__EXP__4,MDL_MB__EXP__2,MDL_LOGSWITCH  ,MDL_GFREDEFINITIONCHOICE,MDL_CMSPARAM,MDL_CKM33,MDL_CKM22  ,MDL_EP,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MW__EXP__2  ,MDL_MZ__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW  ,MDL_MH__EXP__2,MDL_SQRT__2,MDL_I3D,MDL_I3U,MDL_I3L,MDL_I3V  ,MDL_QD,MDL_QU,MDL_QL,MDL_NCOL__EXP__2,MDL_SW__EXP__2  ,MDL_MT__EXP__2,MDL_CW__EXP__2,MDL_MT__EXP__4,MDL_SW__EXP__4  ,MDL_MW__EXP__4,MDL_MT__EXP__3,MDL_MH__EXP__4,MDL_MZ__EXP__4  ,MDL_MT__EXP__6,MDL_MW__EXP__6,MDL_MT__EXP__8,MDL_MW__EXP__8  ,MDL_MT__EXP__10,MDL_MW__EXP__10,MDL_MZ__EXP__6,MDL_SW__EXP__6  ,MDL_SW__EXP__3,MDL_MZ__EXP__3,MDL_CW__EXP__4,MDL_CW__EXP__6  ,MDL_MH__EXP__6,MDL_CW__EXP__3,MDL_SW__EXP__8,MDL_SW__EXP__10  ,MDL_QD__EXP__3,MDL_QD__EXP__2,MDL_I3D__EXP__2,MDL_I3D__EXP__3  ,MDL_QU__EXP__3,MDL_QU__EXP__2,MDL_I3U__EXP__2,MDL_I3U__EXP__3  ,MDL_QL__EXP__3,MDL_I3L__EXP__2,MDL_I3L__EXP__3,MDL_I3V__EXP__3  ,MDL_MW__EXP__3,MDL_QU__EXP__4,MDL_QD__EXP__4,MDL_I3L__EXP__4  ,MDL_I3U__EXP__4,MDL_I3D__EXP__4,MDL_I3V__EXP__4  ,MDL_I3V__EXP__2,MDL_GFBAR,MDL_AEW,AEWM1,MDL_SQRT__AEW,MDL_EE  ,MDL_G1,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YT,MDL_MUH  ,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP,MDL_VECTORZDOWN  ,MDL_AXIALG0UP3,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_EE__EXP__2  ,MDL_EE__EXP__3,MDL_EE__EXP__4,MDL_NTADPOLE,MDL_GF,AS,MDL_YMT
      COMMON/PARAMS_R/ MDL_SQRT__AS,MDL_G__EXP__4,MDL_G__EXP__2  ,MDL_G__EXP__3,MDL_MU_R__EXP__2,MDL_YB,MDL_VEP,MDL_LHV  ,MDL_EPSUV,MDL_EPSIR,MDL_CONJG__CKM33,MDL_CONJG__CKM22  ,MDL_CONJG__CKM11,MDL_SCKM33__EXP__2,MDL_SCKM33  ,MDL_SCKM22__EXP__2,MDL_SCKM22,MDL_SCKM11,MDL_QL__EXP__4  ,MDL_QL__EXP__2,MDL_ME__EXP__2,MDL_MU__EXP__2,MDL_MS__EXP__2  ,MDL_MC__EXP__2,MDL_MB__EXP__4,MDL_MB__EXP__2,MDL_LOGSWITCH  ,MDL_GFREDEFINITIONCHOICE,MDL_CMSPARAM,MDL_CKM33,MDL_CKM22  ,MDL_EP,MDL_NCOL,MDL_CA,MDL_TF,MDL_CF,MDL_MW__EXP__2  ,MDL_MZ__EXP__2,MDL_SW2,MDL_CW,MDL_SQRT__SW2,MDL_SW  ,MDL_MH__EXP__2,MDL_SQRT__2,MDL_I3D,MDL_I3U,MDL_I3L,MDL_I3V  ,MDL_QD,MDL_QU,MDL_QL,MDL_NCOL__EXP__2,MDL_SW__EXP__2  ,MDL_MT__EXP__2,MDL_CW__EXP__2,MDL_MT__EXP__4,MDL_SW__EXP__4  ,MDL_MW__EXP__4,MDL_MT__EXP__3,MDL_MH__EXP__4,MDL_MZ__EXP__4  ,MDL_MT__EXP__6,MDL_MW__EXP__6,MDL_MT__EXP__8,MDL_MW__EXP__8  ,MDL_MT__EXP__10,MDL_MW__EXP__10,MDL_MZ__EXP__6,MDL_SW__EXP__6  ,MDL_SW__EXP__3,MDL_MZ__EXP__3,MDL_CW__EXP__4,MDL_CW__EXP__6  ,MDL_MH__EXP__6,MDL_CW__EXP__3,MDL_SW__EXP__8,MDL_SW__EXP__10  ,MDL_QD__EXP__3,MDL_QD__EXP__2,MDL_I3D__EXP__2,MDL_I3D__EXP__3  ,MDL_QU__EXP__3,MDL_QU__EXP__2,MDL_I3U__EXP__2,MDL_I3U__EXP__3  ,MDL_QL__EXP__3,MDL_I3L__EXP__2,MDL_I3L__EXP__3,MDL_I3V__EXP__3  ,MDL_MW__EXP__3,MDL_QU__EXP__4,MDL_QD__EXP__4,MDL_I3L__EXP__4  ,MDL_I3U__EXP__4,MDL_I3D__EXP__4,MDL_I3V__EXP__4  ,MDL_I3V__EXP__2,MDL_GFBAR,MDL_AEW,AEWM1,MDL_SQRT__AEW,MDL_EE  ,MDL_G1,MDL_GW,MDL_VEV,MDL_VEV__EXP__2,MDL_LAM,MDL_YT,MDL_MUH  ,MDL_AXIALZUP,MDL_AXIALZDOWN,MDL_VECTORZUP,MDL_VECTORZDOWN  ,MDL_AXIALG0UP3,MDL_VECTORAUP,MDL_VECTORADOWN,MDL_VECTORWMDXU  ,MDL_AXIALWMDXU,MDL_VECTORWPUXD,MDL_AXIALWPUXD,MDL_EE__EXP__2  ,MDL_EE__EXP__3,MDL_EE__EXP__4,MDL_NTADPOLE,MDL_GF,AS,MDL_YMT
      DOUBLE COMPLEX MDL_COMPLEXI,MDL_I233,MDL_I333,MDL_VECTORHUP3  ,MDL_VECTORGPUX3D3,MDL_AXIALGPUX3D3,MDL_VECTORGMDX3U3  ,MDL_AXIALGMDX3U3
      COMMON/PARAMS_C/ MDL_COMPLEXI,MDL_I233,MDL_I333,MDL_VECTORHUP3  ,MDL_VECTORGPUX3D3,MDL_AXIALGPUX3D3,MDL_VECTORGMDX3U3  ,MDL_AXIALGMDX3U3
      integer(8) epoch,current_revision,mc_born_model_revision
      external mc_born_model_revision
      integer(8),save::saved_revision=-1
      type(BornModelConfiguration),save::configuration,candidate
      integer status
      logical same_configuration
      current_revision=mc_born_model_revision()
      if(saved_revision.ne.current_revision.or.configuration%epoch.eq.0)then
      if(.not.allocated(candidate%real_values))allocate(candidate%real_values(1))
      if(.not.allocated(candidate%complex_values))allocate(candidate%complex_values(5))
      candidate%complex_values(1:2)=gal
      candidate%real_values(1)=mdl_mz
      candidate%complex_values(3)=gc_68
      candidate%complex_values(4)=gc_69
      candidate%complex_values(5)=gc_76
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

// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    op_stack(instr, locals, device);
    break;
case 1:
    op_unstack(instr, locals, device);
    break;
case 2:
    op_stack_sizes(instr, locals, device);
    break;
case 3:
    op_unstack_sizes(instr, locals, device);
    break;
case 4:
    op_pop(instr, locals, device);
    break;
case 5:
    op_batch_cat(instr, locals, device);
    break;
case 6:
    op_batch_split(instr, locals, device);
    break;
case 7:
    op_cat(instr, locals, device);
    break;
case 8:
    op_batch_size(instr, locals, device);
    break;
case 9:
    op_offset_indices(instr, locals, device);
    break;
case 10:
    op_full(instr, locals, device);
    break;
case 11:
    op_squeeze(instr, locals, device);
    break;
case 12:
    op_unsqueeze(instr, locals, device);
    break;
case 13:
    op_accept_norm(instr, locals, device);
    break;
case 14:
    batch_foreach<tensor_foreach_dynamic<kernel_add<CpuTypes>, kernel_add<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 15:
    batch_foreach<tensor_foreach_dynamic<kernel_add_int<CpuTypes>, kernel_add_int<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 16:
    batch_foreach<tensor_foreach_dynamic<kernel_sub<CpuTypes>, kernel_sub<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 17:
    batch_foreach<tensor_foreach_dynamic<kernel_mul<CpuTypes>, kernel_mul<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 18:
    batch_foreach<tensor_foreach_dynamic<kernel_div<CpuTypes>, kernel_div<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 19:
    batch_foreach<tensor_foreach_dynamic<kernel_reduce_sum<CpuTypes>, kernel_reduce_sum<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 20:
    batch_foreach<tensor_foreach_dynamic<kernel_reduce_sum_vector<CpuTypes>, kernel_reduce_sum_vector<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 21:
    op_batch_reduce_mean(instr, locals, device);
    break;
case 22:
    op_batch_reduce_mean_keepdim(instr, locals, device);
    break;
case 23:
    batch_foreach<tensor_foreach<kernel_reduce_product<CpuTypes>, kernel_reduce_product<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 24:
    batch_foreach<tensor_foreach_dynamic<kernel_sqrt<CpuTypes>, kernel_sqrt<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 25:
    batch_foreach<tensor_foreach_dynamic<kernel_square<CpuTypes>, kernel_square<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 26:
    batch_foreach<tensor_foreach_dynamic<kernel_min<CpuTypes>, kernel_min<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 27:
    batch_foreach<tensor_foreach_dynamic<kernel_max<CpuTypes>, kernel_max<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 28:
    batch_foreach<tensor_foreach<kernel_obs_sqrt_s<CpuTypes>, kernel_obs_sqrt_s<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 29:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_e<CpuTypes>, kernel_obs_e<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 30:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_px<CpuTypes>, kernel_obs_px<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 31:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_py<CpuTypes>, kernel_obs_py<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 32:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_pz<CpuTypes>, kernel_obs_pz<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 33:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_mass<CpuTypes>, kernel_obs_mass<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 34:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_pt<CpuTypes>, kernel_obs_pt<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 35:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_p_mag<CpuTypes>, kernel_obs_p_mag<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 36:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_phi<CpuTypes>, kernel_obs_phi<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 37:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_theta<CpuTypes>, kernel_obs_theta<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 38:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_y<CpuTypes>, kernel_obs_y<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 39:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_y_abs<CpuTypes>, kernel_obs_y_abs<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 40:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_eta<CpuTypes>, kernel_obs_eta<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 41:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_eta_abs<CpuTypes>, kernel_obs_eta_abs<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 42:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_eta<CpuTypes>, kernel_obs_delta_eta<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 43:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_phi<CpuTypes>, kernel_obs_delta_phi<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 44:
    batch_foreach<tensor_foreach_dynamic<kernel_obs_delta_r<CpuTypes>, kernel_obs_delta_r<SimdTypes>, 2, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 45:
    batch_foreach<tensor_foreach<kernel_boost_beam<CpuTypes>, kernel_boost_beam<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 46:
    batch_foreach<tensor_foreach<kernel_boost_beam_inverse<CpuTypes>, kernel_boost_beam_inverse<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 47:
    batch_foreach<tensor_foreach<kernel_com_p_in<CpuTypes>, kernel_com_p_in<SimdTypes>, 1, 2, 1, DeviceType>, 1, 2>(instr, locals, device);
    break;
case 48:
    batch_foreach<tensor_foreach<kernel_r_to_x1x2<CpuTypes>, kernel_r_to_x1x2<SimdTypes>, 3, 3, 1, DeviceType>, 3, 3>(instr, locals, device);
    break;
case 49:
    batch_foreach<tensor_foreach<kernel_x1x2_to_r<CpuTypes>, kernel_x1x2_to_r<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 50:
    batch_foreach<tensor_foreach<kernel_diff_cross_section<CpuTypes>, kernel_diff_cross_section<SimdTypes>, 6, 1, 1, DeviceType>, 6, 1>(instr, locals, device);
    break;
case 51:
    batch_foreach<tensor_foreach<kernel_two_body_decay_com<CpuTypes>, kernel_two_body_decay_com<SimdTypes>, 5, 3, 1, DeviceType>, 5, 3>(instr, locals, device);
    break;
case 52:
    batch_foreach<tensor_foreach<kernel_two_body_decay_com_inverse<CpuTypes>, kernel_two_body_decay_com_inverse<SimdTypes>, 2, 6, 1, DeviceType>, 2, 6>(instr, locals, device);
    break;
case 53:
    batch_foreach<tensor_foreach<kernel_two_body_decay<CpuTypes>, kernel_two_body_decay<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 54:
    batch_foreach<tensor_foreach<kernel_two_body_decay_inverse<CpuTypes>, kernel_two_body_decay_inverse<SimdTypes>, 2, 7, 1, DeviceType>, 2, 7>(instr, locals, device);
    break;
case 55:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering_com<CpuTypes>, kernel_two_to_two_particle_scattering_com<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 56:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering_com_inverse<CpuTypes>, kernel_two_to_two_particle_scattering_com_inverse<SimdTypes>, 4, 4, 1, DeviceType>, 4, 4>(instr, locals, device);
    break;
case 57:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering<CpuTypes>, kernel_two_to_two_particle_scattering<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 58:
    batch_foreach<tensor_foreach<kernel_two_to_two_particle_scattering_inverse<CpuTypes>, kernel_two_to_two_particle_scattering_inverse<SimdTypes>, 4, 4, 1, DeviceType>, 4, 4>(instr, locals, device);
    break;
case 59:
    batch_foreach<tensor_foreach<kernel_two_to_three_particle_scattering<CpuTypes>, kernel_two_to_three_particle_scattering<SimdTypes>, 8, 3, 1, DeviceType>, 8, 3>(instr, locals, device);
    break;
case 60:
    batch_foreach<tensor_foreach<kernel_two_to_three_particle_scattering_inverse<CpuTypes>, kernel_two_to_three_particle_scattering_inverse<SimdTypes>, 7, 4, 1, DeviceType>, 7, 4>(instr, locals, device);
    break;
case 61:
    batch_foreach<tensor_foreach<kernel_double_t_scattering<CpuTypes>, kernel_double_t_scattering<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 62:
    batch_foreach<tensor_foreach<kernel_double_t_scattering_inverse<CpuTypes>, kernel_double_t_scattering_inverse<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 63:
    batch_foreach<tensor_foreach<kernel_three_body_decay_com<CpuTypes>, kernel_three_body_decay_com<SimdTypes>, 9, 4, 1, DeviceType>, 9, 4>(instr, locals, device);
    break;
case 64:
    batch_foreach<tensor_foreach<kernel_three_body_decay_com_inverse<CpuTypes>, kernel_three_body_decay_com_inverse<SimdTypes>, 3, 10, 1, DeviceType>, 3, 10>(instr, locals, device);
    break;
case 65:
    batch_foreach<tensor_foreach<kernel_three_body_decay<CpuTypes>, kernel_three_body_decay<SimdTypes>, 10, 4, 1, DeviceType>, 10, 4>(instr, locals, device);
    break;
case 66:
    batch_foreach<tensor_foreach<kernel_three_body_decay_inverse<CpuTypes>, kernel_three_body_decay_inverse<SimdTypes>, 3, 11, 1, DeviceType>, 3, 11>(instr, locals, device);
    break;
case 67:
    batch_foreach<tensor_foreach<kernel_t_inv_min_max<CpuTypes>, kernel_t_inv_min_max<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 68:
    batch_foreach<tensor_foreach<kernel_t_inv_value_and_min_max<CpuTypes>, kernel_t_inv_value_and_min_max<SimdTypes>, 4, 3, 1, DeviceType>, 4, 3>(instr, locals, device);
    break;
case 69:
    batch_foreach<tensor_foreach<kernel_t1_inv_min_max_doublet<CpuTypes>, kernel_t1_inv_min_max_doublet<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 70:
    batch_foreach<tensor_foreach<kernel_t1_inv_value_and_min_max_doublet<CpuTypes>, kernel_t1_inv_value_and_min_max_doublet<SimdTypes>, 5, 3, 1, DeviceType>, 5, 3>(instr, locals, device);
    break;
case 71:
    batch_foreach<tensor_foreach<kernel_t2_inv_min_max_doublet<CpuTypes>, kernel_t2_inv_min_max_doublet<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 72:
    batch_foreach<tensor_foreach<kernel_t2_inv_value_and_min_max_doublet<CpuTypes>, kernel_t2_inv_value_and_min_max_doublet<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 73:
    batch_foreach<tensor_foreach<kernel_s23_min_max<CpuTypes>, kernel_s23_min_max<SimdTypes>, 6, 2, 1, DeviceType>, 6, 2>(instr, locals, device);
    break;
case 74:
    batch_foreach<tensor_foreach<kernel_s23_value_and_min_max<CpuTypes>, kernel_s23_value_and_min_max<SimdTypes>, 6, 3, 1, DeviceType>, 6, 3>(instr, locals, device);
    break;
case 75:
    batch_foreach<tensor_foreach<kernel_invariants_from_momenta<CpuTypes>, kernel_invariants_from_momenta<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 76:
    batch_foreach<tensor_foreach<kernel_sde2_channel_weights<CpuTypes>, kernel_sde2_channel_weights<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 77:
    batch_foreach<tensor_foreach<kernel_subchannel_weights<CpuTypes>, kernel_subchannel_weights<SimdTypes>, 6, 1, 1, DeviceType>, 6, 1>(instr, locals, device);
    break;
case 78:
    batch_foreach<tensor_foreach<kernel_apply_subchannel_weights<CpuTypes>, kernel_apply_subchannel_weights<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 79:
    batch_foreach<tensor_foreach<kernel_pt_eta_phi_x<CpuTypes>, kernel_pt_eta_phi_x<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 80:
    batch_foreach<tensor_foreach<kernel_mirror_momenta<CpuTypes>, kernel_mirror_momenta<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 81:
    batch_foreach<tensor_foreach<kernel_momenta_to_x1x2<CpuTypes>, kernel_momenta_to_x1x2<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 82:
    batch_foreach<tensor_foreach<kernel_uniform_invariant<CpuTypes>, kernel_uniform_invariant<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 83:
    batch_foreach<tensor_foreach<kernel_uniform_invariant_inverse<CpuTypes>, kernel_uniform_invariant_inverse<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 84:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant<CpuTypes>, kernel_breit_wigner_invariant<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 85:
    batch_foreach<tensor_foreach<kernel_breit_wigner_invariant_inverse<CpuTypes>, kernel_breit_wigner_invariant_inverse<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 86:
    batch_foreach<tensor_foreach<kernel_stable_invariant<CpuTypes>, kernel_stable_invariant<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 87:
    batch_foreach<tensor_foreach<kernel_stable_invariant_inverse<CpuTypes>, kernel_stable_invariant_inverse<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 88:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu<CpuTypes>, kernel_stable_invariant_nu<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 89:
    batch_foreach<tensor_foreach<kernel_stable_invariant_nu_inverse<CpuTypes>, kernel_stable_invariant_nu_inverse<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 90:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless<CpuTypes>, kernel_fast_rambo_massless<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 91:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless_inverse<CpuTypes>, kernel_fast_rambo_massless_inverse<SimdTypes>, 2, 3, 1, DeviceType>, 2, 3>(instr, locals, device);
    break;
case 92:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massless_com<CpuTypes>, kernel_fast_rambo_massless_com<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 93:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive<CpuTypes>, kernel_fast_rambo_massive<SimdTypes>, 4, 2, 1, DeviceType>, 4, 2>(instr, locals, device);
    break;
case 94:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive_inverse<CpuTypes>, kernel_fast_rambo_massive_inverse<SimdTypes>, 3, 3, 1, DeviceType>, 3, 3>(instr, locals, device);
    break;
case 95:
    batch_foreach<tensor_foreach<kernel_fast_rambo_massive_com<CpuTypes>, kernel_fast_rambo_massive_com<SimdTypes>, 3, 2, 1, DeviceType>, 3, 2>(instr, locals, device);
    break;
case 96:
    batch_foreach<tensor_foreach<kernel_cut_unphysical<CpuTypes>, kernel_cut_unphysical<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 97:
    batch_foreach<tensor_foreach<kernel_cut_one<CpuTypes>, kernel_cut_one<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 98:
    batch_foreach<tensor_foreach<kernel_cut_all<CpuTypes>, kernel_cut_all<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 99:
    batch_foreach<tensor_foreach<kernel_cut_any<CpuTypes>, kernel_cut_any<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 100:
    batch_foreach<tensor_foreach<kernel_scale_transverse_energy<CpuTypes>, kernel_scale_transverse_energy<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 101:
    batch_foreach<tensor_foreach<kernel_scale_transverse_mass<CpuTypes>, kernel_scale_transverse_mass<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 102:
    batch_foreach<tensor_foreach<kernel_scale_half_transverse_mass<CpuTypes>, kernel_scale_half_transverse_mass<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 103:
    batch_foreach<tensor_foreach<kernel_scale_partonic_energy<CpuTypes>, kernel_scale_partonic_energy<SimdTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 104:
    batch_foreach<tensor_foreach<kernel_chili_forward<CpuTypes>, kernel_chili_forward<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 105:
    batch_foreach<tensor_foreach<kernel_chili_inverse<CpuTypes>, kernel_chili_inverse<SimdTypes>, 5, 2, 1, DeviceType>, 5, 2>(instr, locals, device);
    break;
case 106:
    op_matrix_element(instr, locals, device);
    break;
case 107:
    batch_foreach<tensor_foreach<kernel_collect_channel_weights<CpuTypes>, kernel_collect_channel_weights<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 108:
    batch_foreach<tensor_foreach<kernel_interpolate_pdf<CpuTypes>, kernel_interpolate_pdf<CpuTypes>, 6, 1, 1, DeviceType>, 6, 1>(instr, locals, device);
    break;
case 109:
    batch_foreach<tensor_foreach<kernel_interpolate_alpha_s<CpuTypes>, kernel_interpolate_alpha_s<CpuTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 110:
    op_matmul(instr, locals, device);
    break;
case 111:
    batch_foreach<tensor_foreach_dynamic<kernel_relu<CpuTypes>, kernel_relu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 112:
    batch_foreach<tensor_foreach_dynamic<kernel_leaky_relu<CpuTypes>, kernel_leaky_relu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 113:
    batch_foreach<tensor_foreach_dynamic<kernel_elu<CpuTypes>, kernel_elu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 114:
    batch_foreach<tensor_foreach_dynamic<kernel_gelu<CpuTypes>, kernel_gelu<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 115:
    batch_foreach<tensor_foreach_dynamic<kernel_sigmoid<CpuTypes>, kernel_sigmoid<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 116:
    batch_foreach<tensor_foreach_dynamic<kernel_softplus<CpuTypes>, kernel_softplus<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 117:
    op_rqs_reshape(instr, locals, device);
    break;
case 118:
    batch_foreach<tensor_foreach<kernel_rqs_find_bin<CpuTypes>, kernel_rqs_find_bin<SimdTypes>, 4, 1, 2, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 119:
    batch_foreach<tensor_foreach<kernel_rqs_forward<CpuTypes>, kernel_rqs_forward<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 120:
    batch_foreach<tensor_foreach<kernel_rqs_inverse<CpuTypes>, kernel_rqs_inverse<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 121:
    batch_foreach<tensor_foreach_dynamic<kernel_softmax<CpuTypes>, kernel_softmax<SimdTypes>, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 122:
    batch_foreach<tensor_foreach<kernel_softmax_prior<CpuTypes>, kernel_softmax_prior<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 123:
    batch_foreach<tensor_foreach<kernel_sample_discrete<CpuTypes>, kernel_sample_discrete<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 124:
    batch_foreach<tensor_foreach<kernel_sample_discrete_inverse<CpuTypes>, kernel_sample_discrete_inverse<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 125:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs<CpuTypes>, kernel_sample_discrete_probs<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 126:
    batch_foreach<tensor_foreach<kernel_sample_discrete_probs_inverse<CpuTypes>, kernel_sample_discrete_probs_inverse<SimdTypes>, 2, 2, 1, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 127:
    op_discrete_histogram(instr, locals, device);
    break;
case 128:
    batch_foreach<tensor_foreach<kernel_permute_momenta<CpuTypes>, kernel_permute_momenta<SimdTypes>, 3, 1, 1, DeviceType>, 3, 1>(instr, locals, device);
    break;
case 129:
    batch_foreach<tensor_foreach<kernel_gather<CpuTypes>, kernel_gather<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 130:
    batch_foreach<tensor_foreach<kernel_gather_int<CpuTypes>, kernel_gather_int<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 131:
    batch_foreach<tensor_foreach<kernel_select_int<CpuTypes>, kernel_select_int<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 132:
    batch_foreach<tensor_foreach<kernel_select<CpuTypes>, kernel_select<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 133:
    batch_foreach<tensor_foreach<kernel_select_vector<CpuTypes>, kernel_select_vector<CpuTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 134:
    batch_foreach<tensor_foreach<kernel_argsort<CpuTypes>, kernel_argsort<CpuTypes>, 1, 1, 1, DeviceType>, 1, 1>(instr, locals, device);
    break;
case 135:
    op_quantile(instr, locals, device);
    break;
case 136:
    batch_foreach<tensor_foreach<kernel_one_hot<CpuTypes>, kernel_one_hot<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 137:
    batch_foreach<tensor_foreach<kernel_madnis_abs_weight<CpuTypes>, kernel_madnis_abs_weight<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 138:
    batch_foreach<tensor_foreach<kernel_madnis_softclip<CpuTypes>, kernel_madnis_softclip<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 139:
    batch_foreach<tensor_foreach<kernel_madnis_variance<CpuTypes>, kernel_madnis_variance<SimdTypes>, 4, 1, 1, DeviceType>, 4, 1>(instr, locals, device);
    break;
case 140:
    batch_foreach<tensor_foreach<kernel_madnis_single_channel_variance<CpuTypes>, kernel_madnis_single_channel_variance<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 141:
    batch_foreach<tensor_foreach<kernel_madnis_multi_channel_variance<CpuTypes>, kernel_madnis_multi_channel_variance<SimdTypes>, 2, 1, 1, DeviceType>, 2, 1>(instr, locals, device);
    break;
case 142:
    op_nonzero(instr, locals, device);
    break;
case 143:
    op_batch_gather(instr, locals, device);
    break;
case 144:
    op_batch_scatter(instr, locals, device);
    break;
case 145:
    op_random(instr, locals, device);
    break;
case 146:
    op_random_int(instr, locals, device);
    break;
case 147:
    op_unweight(instr, locals, device);
    break;
case 148:
    batch_foreach<tensor_foreach<kernel_vegas_forward<CpuTypes>, kernel_vegas_forward<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 149:
    batch_foreach<tensor_foreach<kernel_vegas_inverse<CpuTypes>, kernel_vegas_inverse<SimdTypes>, 2, 2, 2, DeviceType>, 2, 2>(instr, locals, device);
    break;
case 150:
    op_vegas_histogram(instr, locals, device);
    break;
case 151:
    op_histogram(instr, locals, device);
    break;

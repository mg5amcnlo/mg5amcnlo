// This file was automatically generated based on instruction_set.yaml
// Do not modify its content directly

case 0:
    backward_op_stack(instr, locals, local_grads, device);
    break;
case 1:
    backward_op_unstack(instr, locals, local_grads, device);
    break;
case 4:
    backward_op_pop(instr, locals, local_grads, device);
    break;
case 5:
    backward_op_batch_cat(instr, locals, local_grads, device);
    break;
case 6:
    backward_op_batch_split(instr, locals, local_grads, device);
    break;
case 7:
    backward_op_cat(instr, locals, local_grads, device);
    break;
case 11:
    backward_op_squeeze(instr, locals, local_grads, device);
    break;
case 12:
    backward_op_unsqueeze(instr, locals, local_grads, device);
    break;
case 14:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_add<CpuTypes>, backward_kernel_add<SimdTypes>, 3, 2, DeviceType>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 16:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sub<CpuTypes>, backward_kernel_sub<SimdTypes>, 3, 2, DeviceType>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 17:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_mul<CpuTypes>, backward_kernel_mul<SimdTypes>, 3, 2, DeviceType>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 18:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_div<CpuTypes>, backward_kernel_div<SimdTypes>, 3, 2, DeviceType>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 21:
    backward_op_batch_reduce_mean(instr, locals, local_grads, device);
    break;
case 22:
    backward_op_batch_reduce_mean_keepdim(instr, locals, local_grads, device);
    break;
case 23:
    backward_batch_foreach<tensor_foreach<backward_kernel_reduce_product<CpuTypes>, backward_kernel_reduce_product<SimdTypes>, 2, 1, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 24:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sqrt<CpuTypes>, backward_kernel_sqrt<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 25:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_square<CpuTypes>, backward_kernel_square<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 110:
    backward_op_matmul(instr, locals, local_grads, device);
    break;
case 111:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_relu<CpuTypes>, backward_kernel_relu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 112:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_leaky_relu<CpuTypes>, backward_kernel_leaky_relu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 113:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_elu<CpuTypes>, backward_kernel_elu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 114:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_gelu<CpuTypes>, backward_kernel_gelu<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 115:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sigmoid<CpuTypes>, backward_kernel_sigmoid<SimdTypes>, 2, 1, DeviceType>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 116:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softplus<CpuTypes>, backward_kernel_softplus<SimdTypes>, 2, 1, DeviceType>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 117:
    backward_op_rqs_reshape(instr, locals, local_grads, device);
    break;
case 118:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_find_bin<CpuTypes>, backward_kernel_rqs_find_bin<SimdTypes>, 5, 4, 2, DeviceType>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 119:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_forward<CpuTypes>, backward_kernel_rqs_forward<SimdTypes>, 4, 2, 2, DeviceType>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 120:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_inverse<CpuTypes>, backward_kernel_rqs_inverse<SimdTypes>, 4, 2, 2, DeviceType>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 121:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softmax<CpuTypes>, backward_kernel_softmax<SimdTypes>, 2, 1, DeviceType>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 122:
    backward_batch_foreach<tensor_foreach<backward_kernel_softmax_prior<CpuTypes>, backward_kernel_softmax_prior<SimdTypes>, 2, 2, 1, DeviceType>, 2, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 126:
    backward_batch_foreach<tensor_foreach<backward_kernel_sample_discrete_probs_inverse<CpuTypes>, backward_kernel_sample_discrete_probs_inverse<SimdTypes>, 4, 2, 1, DeviceType>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 129:
    backward_batch_foreach<tensor_foreach<backward_kernel_gather<CpuTypes>, backward_kernel_gather<SimdTypes>, 2, 2, 1, DeviceType>, 2, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 132:
    backward_batch_foreach<tensor_foreach<backward_kernel_select<CpuTypes>, backward_kernel_select<SimdTypes>, 2, 2, 1, DeviceType>, 2, 1, 1, 0>(instr, locals, local_grads, {1}, {}, device);
    break;
case 137:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_abs_weight<CpuTypes>, backward_kernel_madnis_abs_weight<SimdTypes>, 3, 2, 1, DeviceType>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 138:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_softclip<CpuTypes>, backward_kernel_madnis_softclip<SimdTypes>, 5, 4, 1, DeviceType>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 139:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_variance<CpuTypes>, backward_kernel_madnis_variance<SimdTypes>, 5, 4, 1, DeviceType>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 140:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_single_channel_variance<CpuTypes>, backward_kernel_madnis_single_channel_variance<SimdTypes>, 2, 2, 1, DeviceType>, 2, 1, 1, 0>(instr, locals, local_grads, {1}, {}, device);
    break;
case 141:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_multi_channel_variance<CpuTypes>, backward_kernel_madnis_multi_channel_variance<SimdTypes>, 3, 2, 1, DeviceType>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;

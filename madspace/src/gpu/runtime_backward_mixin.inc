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
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_add<GpuTypes>, 3, 2>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 16:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sub<GpuTypes>, 3, 2>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 17:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_mul<GpuTypes>, 3, 2>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 18:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_div<GpuTypes>, 3, 2>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 21:
    backward_op_batch_reduce_mean(instr, locals, local_grads, device);
    break;
case 22:
    backward_op_batch_reduce_mean_keepdim(instr, locals, local_grads, device);
    break;
case 23:
    backward_batch_foreach<tensor_foreach<backward_kernel_reduce_product<GpuTypes>, 2, 1, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 24:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sqrt<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 25:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_square<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 110:
    backward_op_matmul(instr, locals, local_grads, device);
    break;
case 111:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_relu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 112:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_leaky_relu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 113:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_elu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 114:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_gelu<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 115:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_sigmoid<GpuTypes>, 2, 1>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 116:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softplus<GpuTypes>, 2, 1>, 1, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 117:
    backward_op_rqs_reshape(instr, locals, local_grads, device);
    break;
case 118:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_find_bin<GpuTypes>, 5, 4, 2>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 119:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_forward<GpuTypes>, 4, 2, 2>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 120:
    backward_batch_foreach<tensor_foreach<backward_kernel_rqs_inverse<GpuTypes>, 4, 2, 2>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 121:
    backward_batch_foreach<tensor_foreach_dynamic<backward_kernel_softmax<GpuTypes>, 2, 1>, 1, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 122:
    backward_batch_foreach<tensor_foreach<backward_kernel_softmax_prior<GpuTypes>, 2, 2, 1>, 2, 1, 0, 1>(instr, locals, local_grads, {}, {0}, device);
    break;
case 126:
    backward_batch_foreach<tensor_foreach<backward_kernel_sample_discrete_probs_inverse<GpuTypes>, 4, 2, 1>, 2, 2, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 129:
    backward_batch_foreach<tensor_foreach<backward_kernel_gather<GpuTypes>, 2, 2, 1>, 2, 1, 1, 0>(instr, locals, local_grads, {0}, {}, device);
    break;
case 132:
    backward_batch_foreach<tensor_foreach<backward_kernel_select<GpuTypes>, 2, 2, 1>, 2, 1, 1, 0>(instr, locals, local_grads, {1}, {}, device);
    break;
case 137:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_abs_weight<GpuTypes>, 3, 2, 1>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;
case 138:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_softclip<GpuTypes>, 5, 4, 1>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 139:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_variance<GpuTypes>, 5, 4, 1>, 4, 1, 4, 0>(instr, locals, local_grads, {0,1,2,3}, {}, device);
    break;
case 140:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_single_channel_variance<GpuTypes>, 2, 2, 1>, 2, 1, 1, 0>(instr, locals, local_grads, {1}, {}, device);
    break;
case 141:
    backward_batch_foreach<tensor_foreach<backward_kernel_madnis_multi_channel_variance<GpuTypes>, 3, 2, 1>, 2, 1, 2, 0>(instr, locals, local_grads, {0,1}, {}, device);
    break;

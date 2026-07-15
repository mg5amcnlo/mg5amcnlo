import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx, once_differentiable

from . import _madspace_py_loader as me


class FunctionModule(nn.Module):
    def __init__(
        self,
        function: me.Function,
        context: me.Context = me.default_context(),
    ):
        super().__init__()
        self.global_params = nn.ParameterDict(
            {
                name.replace(".", ":"): nn.Parameter(
                    context.get_global(name).torch(),
                    context.global_requires_grad(name),
                )
                for name, _ in function.globals
            }
        )
        self.runtime = me.FunctionRuntime(function, context)
        self.dummy = torch.zeros(
            1,
            requires_grad=any(
                glob.requires_grad for glob in self.global_params.values()
            ),
        )

    def forward(self, *args: torch.Tensor) -> list[torch.Tensor]:
        if torch.is_grad_enabled():
            return AutogradWrapper.apply(self, self.dummy, *args)
        else:
            outputs = self.runtime.call(args)
            if len(outputs) == 1:
                return torch.from_dlpack(outputs[0])
            else:
                return tuple(torch.from_dlpack(out) for out in outputs)


class AutogradWrapper(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        module: FunctionModule,
        dummy: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        outputs, local_grads, eval_grad = module.runtime.call_with_grad(
            [arg.detach() for arg in args], [arg.requires_grad for arg in args]
        )
        ctx.module = module
        ctx.eval_grad = eval_grad
        ctx.save_for_backward(
            *(None if grad is None else torch.from_dlpack(grad) for grad in local_grads)
        )
        if len(outputs) == 1:
            return torch.from_dlpack(outputs[0])
        else:
            return tuple(torch.from_dlpack(out) for out in outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, *output_grads: torch.Tensor):
        input_grads, global_grads = ctx.module.runtime.call_backward(
            output_grads, ctx.saved_tensors, ctx.eval_grad
        )
        for name, grad in global_grads:
            if grad is None:
                continue
            param = ctx.module.global_params[name.replace(".", ":")]
            grad_torch = torch.from_dlpack(grad)
            if param.grad is None:
                param.grad = grad_torch
            else:
                param.grad += grad_torch
        input_grads_opt = (
            None if grad is None else torch.from_dlpack(grad) for grad in input_grads
        )
        return None, None, *input_grads_opt

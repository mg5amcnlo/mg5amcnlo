import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytest import approx

import madspace as ms
from madspace.torch import FunctionModule

torch.set_default_dtype(torch.float64)
torch.manual_seed(3210)


@pytest.fixture
def mlp():
    return ms.MLP(10, 1, 32, 3, ms.MLP.leaky_relu, "")


@pytest.fixture
def module_context():
    return ms.Context()


@pytest.fixture
def mlp_module(mlp, module_context):
    mlp.initialize_globals(module_context)
    return FunctionModule(mlp.function(), module_context)


def test_properties():
    mlp = ms.MLP(10, 1, 32, 3, ms.MLP.leaky_relu, "")
    assert mlp.input_dim() == 10
    assert mlp.output_dim() == 1


def test_initialization(mlp_module):
    assert torch.all(mlp_module.global_params["layer1:weight"] != 0)
    assert torch.all(mlp_module.global_params["layer1:bias"] != 0)
    assert torch.all(mlp_module.global_params["layer2:weight"] != 0)
    assert torch.all(mlp_module.global_params["layer2:bias"] != 0)
    assert torch.all(mlp_module.global_params["layer3:weight"] == 0)
    assert torch.all(mlp_module.global_params["layer3:bias"] == 0)


@pytest.fixture(params=["relu", "leaky_relu", "elu", "gelu", "sigmoid", "softplus"])
def activation(request):
    return request.param


def test_activation(activation):
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", ms.batch_float_array(10))]),
        ms.NamedTypes([("out", ms.batch_float_array(10))]),
    )
    fb.output(0, getattr(fb, activation)(fb.input(0)))
    func = FunctionModule(fb.function())
    x = 10 * torch.randn((1000, 10))
    x.requires_grad = True

    y_ms = func(x)
    y_ms.sum().backward()
    grad_me = x.grad
    x.grad = None

    y_torch = getattr(F, activation)(x)
    y_torch.sum().backward()
    grad_torch = x.grad

    assert y_ms.detach() == approx(y_torch.detach())
    assert grad_me == approx(grad_torch)


def test_training(mlp, mlp_module, module_context):
    mlp_torch = nn.Sequential(
        nn.Linear(10, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 1),
    )

    with torch.no_grad():
        for i in range(3):
            mlp_torch[2 * i].weight[:] = mlp_module.global_params[f"layer{i+1}:weight"][
                0
            ]
            mlp_torch[2 * i].bias[:] = mlp_module.global_params[f"layer{i+1}:bias"][0]

    ctx = ms.Context()
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", ms.batch_float_array(10))]),
        ms.NamedTypes([("out", ms.single_float), ("mlp_out", ms.batch_float)]),
    )
    fb_in = fb.input(0)
    target = fb.reduce_sum(fb.square(fb_in))
    mlp_out = fb.squeeze(mlp.build_function(fb, [fb_in]).values()[0])
    loss = fb.batch_reduce_mean(fb.square(fb.sub(target, mlp_out)))
    fb.output(0, loss)
    fb.output(1, mlp_out)
    ctx.copy_globals_from(module_context)
    builtin_opt = ms.AdamOptimizer(
        function=fb.function(), context=ctx, learning_rate=1e-2
    )

    opt_ms = torch.optim.Adam(mlp_module.parameters(), lr=1e-2)
    opt_torch = torch.optim.Adam(mlp_torch.parameters(), lr=1e-2)

    for i in range(10):
        x = torch.randn((128, 10))
        y = x.square().sum(dim=1, keepdim=True)

        loss_ms = (y - mlp_module(x)).square().mean()
        opt_ms.zero_grad()
        loss_ms.backward()
        opt_ms.step()

        loss_torch = (y - mlp_torch(x)).square().mean()
        opt_torch.zero_grad()
        loss_torch.backward()
        opt_torch.step()

        loss_builtin, mlp_out = builtin_opt.step([x])

        assert loss_ms.item() == approx(loss_torch.item())
        assert loss_builtin.torch().item() == approx(loss_torch.item())

        with torch.no_grad():
            for i in range(3):
                w_torch = mlp_torch[2 * i].weight.numpy()
                b_torch = mlp_torch[2 * i].bias.numpy()
                assert mlp_module.global_params[f"layer{i+1}:weight"][
                    0
                ].numpy() == approx(w_torch)
                assert mlp_module.global_params[f"layer{i+1}:bias"][
                    0
                ].numpy() == approx(b_torch)
                assert mlp_module.global_params[f"layer{i+1}:weight"].grad[
                    0
                ].numpy() == approx(mlp_torch[2 * i].weight.grad.numpy())
                assert mlp_module.global_params[f"layer{i+1}:bias"].grad[
                    0
                ].numpy() == approx(mlp_torch[2 * i].bias.grad.numpy())
                assert ctx.get_global(f"layer{i+1}.weight").numpy()[0] == approx(
                    w_torch
                )
                assert ctx.get_global(f"layer{i+1}.bias").numpy()[0] == approx(b_torch)

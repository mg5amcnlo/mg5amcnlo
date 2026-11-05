import numpy as np
import pytest

import madspace as ms

PDG_IDS = [21, 21, 6, -6]


@pytest.fixture
def context():
    # several threads, so the function is called from worker threads
    return ms.Context(device=ms.cpu_device(), thread_count=4)


@pytest.fixture
def momenta():
    rng = np.random.default_rng(1234)
    return rng.normal(size=(5000, 4, 4)) * 100.0


def pt(momenta, index):
    return np.hypot(momenta[:, index, 1], momenta[:, index, 2])


SCALE_OUTPUTS = ["ren_scale", "fact_scale1", "fact_scale2"]


def run_custom(context, function, kind, momenta, framework="numpy", **kwargs):
    outputs = SCALE_OUTPUTS if kind == "scale" else [kind]
    parameters = kwargs.pop("parameters", None)
    api = context.add_matrix_element(
        ms.PythonFunction(
            function, ["momenta", "pdg_ids"], outputs, framework=framework, **kwargs
        ),
        parameters=parameters,
    )
    generator = ms.CustomFunction(api, getattr(ms.CustomFunction, kind), 4, [PDG_IDS])
    return ms.FunctionRuntime(generator.function(), context)(momenta)


def test_numpy_cut(context, momenta):
    def cut(p, pdg_ids):
        assert p.shape == (len(p), 4, 4)
        assert (pdg_ids == PDG_IDS).all()
        return (pt(p, 2) > 100.0).astype(np.float64)

    values = run_custom(context, cut, "cut", momenta)
    assert np.array_equal(np.asarray(values), (pt(momenta, 2) > 100.0).astype(float))


def test_numpy_scale(context, momenta):
    def scale(p, pdg_ids):
        return (pt(p, 2) + pt(p, 3)) / 2, pt(p, 2), pt(p, 3)

    ren, fact1, fact2 = run_custom(context, scale, "scale", momenta)
    assert np.allclose(ren, (pt(momenta, 2) + pt(momenta, 3)) / 2)
    assert np.allclose(fact1, pt(momenta, 2))
    assert np.allclose(fact2, pt(momenta, 3))


def test_parameters(context, momenta):
    def cut(p, pdg_ids, params):
        return (pt(p, 2) > params.pt_min).astype(np.float64) * params.factor

    function = ms.PythonFunction(
        cut, ["momenta", "pdg_ids"], ["cut"], parameters={"pt_min": 20.0, "factor": 1.0}
    )
    # pt_min is set, factor keeps its default
    api = context.add_matrix_element(function, parameters={"pt_min": 50.0})
    generator = ms.CustomFunction(api, ms.CustomFunction.cut, 4, [PDG_IDS])
    values = ms.FunctionRuntime(generator.function(), context)(momenta)
    assert np.array_equal(np.asarray(values), (pt(momenta, 2) > 50.0).astype(float))


def test_unknown_parameter(context):
    function = ms.PythonFunction(
        lambda p, ids, params: np.ones(len(p)),
        ["momenta", "pdg_ids"],
        ["cut"],
        parameters={"pt_min": 20.0},
    )
    with pytest.raises(RuntimeError, match="could not set parameter pt_max"):
        context.add_matrix_element(function, parameters={"pt_max": 1.0})


def test_numpy_pdf(context):
    def pdf(beam_index, pdg_id, x, q):
        return np.where(pdg_id == 21, 3.0, 0.5) * np.sqrt(x) * (1 - x) ** 3 + beam_index

    api = context.add_matrix_element(
        ms.PythonFunction(
            pdf, ["beam_index", "pdg_id", "momentum_fraction", "scale"], ["pdf"]
        )
    )
    x = np.array([0.1, 0.5, 0.9])
    for beam_index in (0, 1):
        density = ms.PartonDensity(ms.CustomPdf(api), [21, 2], beam_index)
        values = np.asarray(
            ms.FunctionRuntime(density.function(), context)(x, np.full(3, 100.0))
        )
        toy = np.sqrt(x) * (1 - x) ** 3
        assert np.allclose(values[:, 0], 3 * toy + beam_index)
        assert np.allclose(values[:, 1], 0.5 * toy + beam_index)


def test_multiple_outputs_fixed_particle_count(context, momenta):
    def matrix_element(p):
        return p[:, 2, 0] ** 2, np.full(len(p), 3, dtype=np.int32)

    api = context.add_matrix_element(
        ms.PythonFunction(
            matrix_element,
            ["momenta"],
            ["matrix_element", "helicity_index"],
            particle_count=4,
        )
    )
    generator = ms.MatrixElement(
        api.index(),
        4,
        [ms.MatrixElement.momenta_in],
        [ms.MatrixElement.matrix_element_out, ms.MatrixElement.helicity_index_out],
    )
    me, helicity = ms.FunctionRuntime(generator.function(), context)(momenta)
    assert np.allclose(me, momenta[:, 2, 0] ** 2)
    assert (np.asarray(helicity) == 3).all()


def test_fixed_channel(context, momenta):
    def matrix_element(p, flavor, channel):
        return p[:, 2, 0] * (channel + 10 * flavor)

    api = context.add_matrix_element(
        ms.PythonFunction(
            matrix_element,
            ["momenta", "flavor_index", "channel_index"],
            ["matrix_element"],
            particle_count=4,
        )
    )
    inputs = [
        ms.MatrixElement.momenta_in,
        ms.MatrixElement.flavor_in,
        ms.MatrixElement.channel_in,
    ]
    generator = ms.MatrixElement(
        api.index(), 4, inputs, [ms.MatrixElement.matrix_element_out], channel=3
    )
    assert generator.channel() == 3
    assert generator.external_inputs() == inputs[:2]
    flavor = np.full(len(momenta), 2, dtype=np.int32)
    me = ms.FunctionRuntime(generator.function(), context)(momenta, flavor)
    assert np.allclose(me, momenta[:, 2, 0] * 23)


@pytest.mark.parametrize(
    "result, message",
    [
        (lambda p: np.ones(len(p), dtype=np.float32), "must have dtype float64"),
        (lambda p: np.ones(3), "must have shape"),
        (lambda p: list(np.ones(len(p))), "must be a numpy.ndarray"),
    ],
)
def test_result_checks(context, momenta, result, message):
    with pytest.raises(RuntimeError, match=message):
        run_custom(context, lambda p, pdg_ids: result(p), "cut", momenta)


def test_exception_propagates(context, momenta):
    def broken(p, pdg_ids):
        raise KeyError("something went wrong")

    with pytest.raises(
        RuntimeError, match="Error in Python function broken.*something"
    ):
        run_custom(context, broken, "cut", momenta)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="unknown input"):
        ms.PythonFunction(lambda: None, ["momentum"], ["cut"])
    with pytest.raises(ValueError, match="unknown output"):
        ms.PythonFunction(lambda: None, ["momenta"], ["cuts"])
    with pytest.raises(ValueError, match="framework"):
        ms.PythonFunction(lambda: None, ["momenta"], ["cut"], framework="jax")


def test_release_inputs_on_worker_threads(context, momenta):
    # numpy-backed inputs are freed by madspace, possibly on a worker thread
    def cut(p, pdg_ids):
        return np.ones(len(p))

    api = context.add_matrix_element(
        ms.PythonFunction(cut, ["momenta", "pdg_ids"], ["cut"])
    )
    generator = ms.CustomFunction(api, ms.CustomFunction.cut, 4, [PDG_IDS])
    runtime = ms.FunctionRuntime(generator.function(), context)
    for _ in range(20):
        runtime(momenta.copy())
        runtime.release_inputs()


def test_torch_cpu(context, momenta):
    torch = pytest.importorskip("torch")

    def scale(p, pdg_ids):
        assert isinstance(p, torch.Tensor)
        value = torch.hypot(p[:, 2:, 1], p[:, 2:, 2]).sum(dim=1) / 2
        return value, value, value

    expected = (pt(momenta, 2) + pt(momenta, 3)) / 2
    values = run_custom(context, scale, "scale", momenta, framework="torch")
    assert np.allclose(values[0], expected)
    values = run_custom(
        context, scale, "scale", torch.from_numpy(momenta), framework="torch"
    )
    assert torch.allclose(values[2], torch.from_numpy(expected))


def test_torch_wrong_type(context, momenta):
    pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="must be a torch.Tensor"):
        run_custom(context, lambda p, ids: np.ones(len(p)), "cut", momenta, "torch")


@pytest.mark.parametrize("fact_scale_fixed", [False, True])
def test_energy_scale_three_custom_scales(context, momenta, fact_scale_fixed):
    def scale(p, pdg_ids):
        return pt(p, 2), pt(p, 3), pt(p, 2) + pt(p, 3)

    api = context.add_matrix_element(
        ms.PythonFunction(scale, ["momenta", "pdg_ids"], SCALE_OUTPUTS)
    )
    custom = ms.CustomFunction(api, ms.CustomFunction.scale, 4, [PDG_IDS])
    energy_scale = ms.EnergyScale(
        4,
        ms.EnergyScale.partonic_energy,
        False,
        fact_scale_fixed,
        0.0,
        91.0,
        92.0,
        scale_factor=2.0,
        custom_scale=custom,
    )
    ren, fact1, fact2 = ms.FunctionRuntime(energy_scale.function(), context)(momenta)
    # scale_factor applies to the dynamical scales, a fixed scale is kept
    assert np.allclose(ren, 2 * pt(momenta, 2))
    if fact_scale_fixed:
        assert np.allclose(fact1, 91.0) and np.allclose(fact2, 92.0)
    else:
        assert np.allclose(fact1, 2 * pt(momenta, 3))
        assert np.allclose(fact2, 2 * (pt(momenta, 2) + pt(momenta, 3)))

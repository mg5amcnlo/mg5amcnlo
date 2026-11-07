"""Guard the "expose every constructor parameter" documentation convention.

For each documented ``Mapping`` subclass under ``phasespace/``, construct it
with every parameter its docstring names passed **by keyword**. This fails if a
binding drops a constructor argument (e.g. ``has_cut`` of ``DoubleT``) or
renames one away from the header, keeping the rendered Parameters list honest.
"""

import inspect

import pytest

import madspace as ms

# One representative keyword-argument set per in-scope class. Every key here is
# a parameter the class comment documents; the call must accept all of them.
CASES = {
    "Invariant": dict(power=0.8, mass=0.0, width=0.0),
    "Luminosity": dict(
        s_lab=1.0e8,
        s_hat_min=1.0e4,
        s_hat_max=1.0e8,
        invariant_power=1.0,
        mass=0.0,
        width=0.0,
    ),
    "TwoBodyDecay": dict(com=True),
    "TwoToTwoParticleScattering": dict(
        com=True, invariant_power=0.0, mass=0.0, width=0.0, has_cut=False
    ),
    "DoubleT": dict(
        t1_invariant_power=0.0,
        t1_mass=0.0,
        t1_width=0.0,
        t2_invariant_power=0.0,
        t2_mass=0.0,
        t2_width=0.0,
        has_cut=False,
    ),
    "ThreeBodyDecay": dict(com=True),
    "TwoToThreeParticleScattering": dict(
        t_invariant_power=0.0,
        t_mass=0.0,
        t_width=0.0,
        s_invariant_power=0.0,
        s_mass=0.0,
        s_width=0.0,
        has_cut=False,
    ),
    "ChiliMapping": dict(n_particles=3, y_max=[], pt_min=[]),
    "FastRamboMapping": dict(n_particles=4, massless=True, com=True),
    "TPropagatorMapping": dict(integration_order=[0], invariant_power=0.8, pt_min=[]),
    "ColorOrderedMapping": dict(
        options=ms.ColorOrderedMapping.Options([0, 1, 2, 3]),
        pt_min=[],
        m_inv_min=[],
        dr_min=[],
        y_max=[],
    ),
    "MultiChannelMapping": dict(
        mappings=[ms.TwoBodyDecay(com=True), ms.TwoBodyDecay(com=True)]
    ),
    "VegasMapping": dict(dimension=3, bin_count=64, prefix=""),
    "Flow": dict(
        input_dim=4,
        condition_dim=0,
        prefix="",
        bin_count=10,
        subnet_hidden_dim=32,
        subnet_layers=3,
        subnet_activation=ms.MLP.Activation.leaky_relu,
        invert_spline=True,
    ),
    "DiscreteSampler": dict(option_counts=[3, 4], prefix="", dims_with_prior=[]),
    "DiscreteFlow": dict(
        option_counts=[3, 4],
        prefix="",
        dims_with_prior=[],
        condition_dim=0,
        subnet_hidden_dim=32,
        subnet_layers=3,
        subnet_activation=ms.MLP.Activation.leaky_relu,
    ),
    "MLP": dict(
        input_dim=4,
        output_dim=2,
        hidden_dim=32,
        layers=3,
        activation=ms.MLP.Activation.leaky_relu,
        prefix="",
    ),
    # A few cheaply-constructible FunctionGenerator subclasses.
    "EnergyScale": dict(
        particle_count=4,
        type=ms.EnergyScale.DynamicalScaleType.transverse_energy,
    ),
    "MomentumPreprocessing": dict(particle_count=4),
    "ChannelWeightNetwork": dict(
        channel_count=3,
        particle_count=4,
        hidden_dim=32,
        layers=3,
        activation=ms.MLP.Activation.leaky_relu,
        prefix="",
        include_preprocessing=True,
    ),
    "VegasHistogram": dict(dimension=3, bin_count=64),
    "DiscreteHistogram": dict(option_counts=[3, 4]),
}


@pytest.mark.parametrize("name", ["MultiChannelFunction"])
def test_multi_channel_function_kwargs(name):
    ms.MultiChannelFunction(
        functions=[ms.MLP(2, 2), ms.MLP(2, 2)], return_batch_sizes=False
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_documented_constructor_kwargs_are_accepted(name):
    cls = getattr(ms, name)
    obj = cls(**CASES[name])
    assert isinstance(obj, cls)


def test_phasespace_mapping_both_constructors():
    ms.PhaseSpaceMapping(
        external_masses=[10.0, 10.0, 10.0, 10.0],
        cm_energy=1000.0,
        leptonic=False,
        invariant_power=0.8,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        cuts=None,
        color_ordered=None,
    )


def test_matrix_element_index_is_not_diagram_count():
    """`matrix_element_index` and `diagram_count` are separate parameters."""
    me = ms.MatrixElement(
        matrix_element_index=3,
        particle_count=4,
        inputs=[ms.MatrixElement.MatrixElementInput.momenta_in],
        outputs=[ms.MatrixElement.MatrixElementOutput.matrix_element_out],
        diagram_count=1,
    )
    assert me.matrix_element_index() == 3
    assert me.diagram_count() == 1


def test_every_documented_class_is_importable():
    for name in [*CASES, "PhaseSpaceMapping"]:
        assert inspect.isclass(getattr(ms, name))

import math
import os

import pytest
import torch
from madspace_testing import default_context, to_numpy

import madspace as ms

PDF_SET = "NNPDF40_nlo_as_01180"

try:
    import lhapdf

    lhapdf.setVerbosity(0)
    reference_pdf = lhapdf.mkPDF(PDF_SET, 0)
except ImportError:
    lhapdf = None
    reference_pdf = None
except RuntimeError:
    reference_pdf = None

pytestmark = [
    pytest.mark.skipif(lhapdf is None, reason="lhapdf required to run this test"),
    pytest.mark.skipif(
        reference_pdf is None, reason=f"pdf set {PDF_SET} required to run this test"
    ),
]


def test_pdf():
    ctx = default_context()
    grid = ms.PdfGrid(os.path.join(lhapdf.paths()[0], PDF_SET, f"{PDF_SET}_0000.dat"))
    grid.initialize_globals(ctx)
    pids = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5]
    pdf = ms.PartonDensity(grid, pids)

    xs = torch.logspace(
        math.log10(reference_pdf.xMin) + 1e-6, math.log10(reference_pdf.xMax), 100
    )
    q2s = torch.logspace(
        math.log10(reference_pdf.q2Min) + 1e-6, math.log10(reference_pdf.q2Max), 100
    )
    x_grid, q2_grid = torch.meshgrid(xs, q2s, indexing="xy")
    x, q2 = x_grid.flatten(), q2_grid.flatten()

    result = pdf(x, torch.sqrt(q2))
    reference = torch.tensor(
        reference_pdf.xfxQ2(pids, to_numpy(x).tolist(), to_numpy(q2).tolist())
    )
    torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-12)


def test_alpha_s():
    ctx = default_context()
    grid = ms.AlphaSGrid(os.path.join(lhapdf.paths()[0], PDF_SET, f"{PDF_SET}.info"))
    grid.initialize_globals(ctx)
    alpha_s = ms.RunningCoupling(grid)

    q2 = torch.logspace(
        math.log10(reference_pdf.q2Min) + 1e-6, math.log10(reference_pdf.q2Max), 1000
    )

    result = alpha_s(torch.sqrt(q2))
    reference = torch.tensor(
        [reference_pdf.alphasQ2(q2_item) for q2_item in to_numpy(q2).tolist()]
    )
    torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-12)

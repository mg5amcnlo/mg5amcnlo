################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""mg7/madtrex_driver.py, the standalone MadtRex reweighting driver that
gridpacks ship as bin/reweight, runs without the rest of MadGraph.

It is run as a plain script, in a subprocess where the madgraph package
cannot be imported. Skipped unless madspace and madtrex are installed in
madspace/install and madtrex/install.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

MG_ROOT = Path(__file__).resolve().parents[3]
MG7_DIR = MG_ROOT / "madgraph" / "iolibs" / "template_files" / "mg7"
SCRIPT = MG7_DIR / "madtrex_driver.py"


def madtrex_is_installed():
    return (MG_ROOT / "madspace" / "install" / "madspace").is_dir() and (
        MG_ROOT / "madtrex" / "install" / "madtrex"
    ).is_dir()


@unittest.skipUnless(madtrex_is_installed(), "madspace/madtrex are not installed")
class MadtrexStandaloneTest(unittest.TestCase):

    def run_isolated(self, code):
        """Runs code with -I (no PYTHONPATH, no cwd on sys.path), from a
        directory outside MadGraph, so madgraph cannot be imported."""
        with tempfile.TemporaryDirectory() as tmp:
            return subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp, capture_output=True, text=True,
            )

    def test_script_runs_without_madgraph(self):
        result = self.run_isolated(textwrap.dedent(f"""
            import runpy, sys
            sys.argv = [{str(SCRIPT)!r}, "--help"]
            try:
                runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
            except SystemExit as exc:
                assert exc.code == 0, exc.code
            assert not any(m == "madgraph" or m.startswith("madgraph.")
                           for m in sys.modules), "madgraph was imported"
            import madtrex
            assert madtrex.__file__.startswith({str(MG_ROOT / "madtrex")!r}), madtrex.__file__
        """))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--param-rwgt-card", result.stdout)

    def test_run_card_is_read_without_madgraph(self):
        with tempfile.TemporaryDirectory() as proc_dir:
            os.mkdir(os.path.join(proc_dir, "Cards"))
            with open(os.path.join(proc_dir, "Cards", "run_card.toml"), "w") as f:
                f.write('[run]\nprecision = "all64"\n\n[beam]\npdf1 = "X"\npdf2 = "X"\n')
            result = self.run_isolated(textwrap.dedent(f"""
                import runpy
                ns = runpy.run_path({str(SCRIPT)!r})
                print(ns["_read_run_fptype"]({proc_dir!r}),
                      ns["_read_default_pdf_set"]({proc_dir!r})[0],
                      ns["_read_run_card"]({proc_dir!r} + "/Cards/run_card.toml")["run"]["device"])
            """))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["d", "X", "['cpu']"])

    def test_reweighting_threads(self):
        """MadtRex uses as many threads as the generation: the run card's
        cpu_thread_pool_size, all available cores for -1."""
        with tempfile.TemporaryDirectory() as proc_dir:
            os.mkdir(os.path.join(proc_dir, "Cards"))
            card = os.path.join(proc_dir, "Cards", "run_card.toml")
            result = []
            for threads in (3, -1):
                with open(card, "w") as f:
                    f.write("[run]\ncpu_thread_pool_size = %d\n" % threads)
                result.append(self.run_isolated(textwrap.dedent(f"""
                    import os, runpy
                    ns = runpy.run_path({str(SCRIPT)!r})
                    print(ns["reweighting_threads"]({proc_dir!r}) ==
                          {{3: 3, -1: len(os.sched_getaffinity(0))}}[{threads}])
                """)))
        for each in result:
            self.assertEqual(each.returncode, 0, each.stderr)
            self.assertEqual(each.stdout.strip(), "True")

    def test_generation_run_card_comes_from_the_events(self):
        """The baseline PDF set and the normaliser fptype are those the
        events were generated with (the run card in their LHE header), not
        those of the current Cards/run_card.toml, edited since."""
        import gzip
        with tempfile.TemporaryDirectory() as proc_dir:
            os.mkdir(os.path.join(proc_dir, "Cards"))
            with open(os.path.join(proc_dir, "Cards", "run_card.toml"), "w") as f:
                f.write('[run]\nprecision = "all32"\n\n[beam]\npdf1 = "NEW"\npdf2 = "NEW"\n')
            header = ('<LesHouchesEvents version="3.0">\n<header>\n<MG7RunCard>\n'
                      '[run]\nprecision = &quot;all64&quot;\n\n[beam]\n'
                      'pdf1 = &quot;OLD&quot;\npdf2 = &quot;OLD&quot;\n</MG7RunCard>\n'
                      '</header>\n<init>\n2212 2212 6.5e+03 6.5e+03 0 0 1 1 3 1\n</init>\n')
            npy_run = os.path.join(proc_dir, "Events", "run_01")
            lhe_run = os.path.join(proc_dir, "Events", "run_02")
            for run in (npy_run, lhe_run):
                os.makedirs(run)
            with open(os.path.join(npy_run, "header.lhe"), "w") as f:
                f.write(header + "</LesHouchesEvents>\n")
            with gzip.open(os.path.join(lhe_run, "events.lhe.gz"), "wt") as f:
                f.write(header + "<event>\n</event>\n</LesHouchesEvents>\n")
            result = self.run_isolated(textwrap.dedent(f"""
                import runpy
                ns = runpy.run_path({str(SCRIPT)!r})
                for events in ({npy_run + "/events.npy"!r}, {lhe_run + "/events.lhe.gz"!r}, None):
                    print(ns["_read_run_fptype"]({proc_dir!r}, events),
                          ns["_read_default_pdf_set"]({proc_dir!r}, events)[0])
            """))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["d", "OLD", "d", "OLD", "f", "NEW"])

    def test_classic_cards_are_refused(self):
        """bin/reweight is MadtRex only: a card for the classic module says
        that classic reweighting needs MadGraph."""
        with tempfile.TemporaryDirectory() as tmp:
            cards = {"classic": "change backend classic\nlaunch\nset MT 175\n",
                     "output": "change output 2.0\nlaunch\nset MT 175\n",
                     "madtrex": "change backend madtrex\nlaunch\nset MT 175\n",
                     "auto": "change backend auto\nlaunch\nset MT 175\n"}
            for name, text in cards.items():
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(text)
            result = self.run_isolated(textwrap.dedent(f"""
                import runpy
                ns = runpy.run_path({str(SCRIPT)!r})
                for name in ("classic", "output", "madtrex", "auto"):
                    try:
                        ns["check_card_for_standalone"]({tmp!r} + "/" + name)
                        print(name, "ok")
                    except SystemExit as exc:
                        print(name, "MadGraph" in str(exc))
            """))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["classic", "True", "output", "True",
                                                 "madtrex", "ok", "auto", "ok"])

    def test_gridpack_layout(self):
        """As bin/reweight in a gridpack: madspace/madtrex are found next to
        bin/, the gridpack is the default process directory, and the
        resolved cpu backend comes from grid_run_card.toml."""
        with tempfile.TemporaryDirectory() as gridpack:
            # resolved: the driver locates itself through its resolved path
            # (macOS's /var/folders is a symlink into /private)
            gridpack = Path(gridpack).resolve()
            for pkg in ("madspace", "madtrex"):
                (gridpack / pkg / "install").mkdir(parents=True)
                os.symlink(MG_ROOT / pkg / "install" / pkg,
                           gridpack / pkg / "install" / pkg)
            os.symlink(MG_ROOT / "madtrex" / "install" / "pyrex",
                       gridpack / "madtrex" / "install" / "pyrex")
            (gridpack / "bin").mkdir()
            for name in ("param_reweight_card.py", "pdf_reweight_card.py",
                         "generic_reweight_card.py"):
                shutil.copy(MG7_DIR / name, gridpack / "bin" / name)
            shutil.copy(SCRIPT, gridpack / "bin" / "reweight")
            (gridpack / "SubProcesses").mkdir()
            (gridpack / "SubProcesses" / "subprocesses.json").write_text("[]")
            (gridpack / "Cards").mkdir()
            (gridpack / "Cards" / "run_card.toml").write_text('[run]\ncpu_mode = "auto"\n')
            (gridpack / "Cards" / "grid_run_card.toml").write_text(
                '[run]\ndevice = ["cpu"]\ncpu_mode = "avx512y"\n')
            script = gridpack / "bin" / "reweight"
            result = self.run_isolated(textwrap.dedent(f"""
                import runpy
                ns = runpy.run_path({str(script)!r})
                import madtrex
                print(madtrex.__file__.startswith({str(gridpack)!r}),
                      ns["_default_process_directory"]() == {str(gridpack)!r},
                      ns["_read_devices"]({str(gridpack)!r}))
            """))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["True", "True", "['avx512y']"])


@unittest.skipUnless(madtrex_is_installed(), "madspace/madtrex are not installed")
class RunReweightingRoutingTest(unittest.TestCase):
    """madtrex_driver.run_reweighting() hands each card kind to its runner."""

    def test_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards = {"param": "launch\nset MT 175\n",
                     "pdf": "launch\nset PDF_SET X\n",
                     "generic": "change mode pdf\nlaunch\nset PDF_SET X\n"}
            for kind, text in cards.items():
                with open(os.path.join(tmp, kind + ".dat"), "w") as f:
                    f.write(text)
            result = subprocess.run(
                [sys.executable, "-I", "-c", textwrap.dedent(f"""
                    import importlib.util, sys
                    spec = importlib.util.spec_from_file_location("drv", {str(SCRIPT)!r})
                    drv = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(drv)
                    for name in ("run_parameter_reweighting", "run_pdf_reweighting",
                                 "run_generic_reweighting"):
                        setattr(drv, name, lambda card, _n=name, **kw: _n)
                    out = [drv.run_reweighting({tmp!r} + "/" + k + ".dat")
                           for k in ("param", "pdf", "generic")]
                    out.append(drv.run_reweighting({tmp!r} + "/param.dat", card_kind="generic"))
                    try:
                        drv.run_reweighting({tmp!r} + "/param.dat", card_kind="nlo")
                    except ValueError:
                        out.append("error")
                    print(" ".join(out))
                """)],
                cwd=tmp, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), [
            "run_parameter_reweighting", "run_pdf_reweighting", "run_generic_reweighting",
            "run_generic_reweighting", "error"])


def load_driver_module():
    """mg7/madtrex_driver.py, loaded by path (it imports madspace/madtrex)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("madtrex_driver_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(madtrex_is_installed(), "madspace/madtrex are not installed")
class InPlaceOutputTest(unittest.TestCase):
    """By default the weights go into the event file that was reweighted,
    after the weights already there -- the same file the classic module and
    the other tools use."""

    def setUp(self):
        self.driver = load_driver_module()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    class Result:
        def __init__(self, ids, contexts=()):
            self.weight_ids = ids
            self.weight_context = list(contexts)

    def test_lhe_gz_is_replaced_and_stays_gzipped(self):
        import gzip
        source = os.path.join(self.tmp, "events.lhe.gz")
        with gzip.open(source, "wt") as f:
            f.write("original\n")
        plain = self.driver.prepare_events(source)
        self.assertTrue(os.path.isfile(plain))

        class Driver:
            def write_weights(self, path):
                with open(path, "w") as f:
                    f.write("original + new weights\n")
        out = self.driver.write_weights_in_place(Driver(), self.Result(["mt"]), source, plain)
        self.assertEqual(out, source)
        with gzip.open(source, "rt") as f:
            self.assertEqual(f.read(), "original + new weights\n")
        # the decompressed copy is gone: it would shadow events.lhe.gz
        self.assertEqual(sorted(os.listdir(self.tmp)), ["events.lhe.gz"])

    def test_plain_lhe_is_replaced(self):
        source = os.path.join(self.tmp, "events.lhe")
        with open(source, "w") as f:
            f.write("original\n")

        class Driver:
            def write_weights(self, path):
                with open(path, "w") as f:
                    f.write("new\n")
        self.driver.write_weights_in_place(Driver(), self.Result(["mt"]), source, source)
        with open(source) as f:
            self.assertEqual(f.read(), "new\n")
        self.assertEqual(os.listdir(self.tmp), ["events.lhe"])

    def npy_run(self):
        import numpy as np
        dtype = [("process_id", "<i4"), ("weight", "<f8"), ("rwgt_1", "<f8"),
                 ("rwgt_2", "<f8"), ("part1_pdg_id", "<i4"), ("part1_px", "<f8")]
        events = np.zeros(3, dtype=dtype)
        events["weight"] = [1., 2., 3.]
        events["rwgt_1"] = [1.1, 2.1, 3.1]
        events["part1_px"] = [7., 8., 9.]
        source = os.path.join(self.tmp, "events.npy")
        np.save(source, events)
        with open(os.path.join(self.tmp, "header.lhe"), "w") as f:
            f.write("<LesHouchesEvents>\n<header>\n<initrwgt>\n<weightgroup name='scale'>\n"
                    "<weight id='1'>x</weight>\n</weightgroup>\n</initrwgt>\n</header>\n")
        return source

    def npy_driver(self, weights):
        import numpy as np

        class Driver:
            def write_weights(self, prefix):
                for weight_id, values in weights.items():
                    data = np.zeros(len(values), dtype=[("weight", "<f8")])
                    data["weight"] = values
                    np.save(f"{prefix}_{weight_id}.npy", data)
        return Driver()

    def test_npy_gets_named_columns_after_the_systematics(self):
        import numpy as np
        source = self.npy_run()
        driver = self.npy_driver({"mt175": [10., 20., 30.]})
        self.driver.write_weights_in_place(driver, self.Result(["mt175"], ["set MT 175"]),
                                           source, source)
        events = np.load(source)
        self.assertEqual(events.dtype.names, ("process_id", "weight", "rwgt_1", "rwgt_2",
                                              "rwgt_mt175", "part1_pdg_id", "part1_px"))
        self.assertEqual(list(events["rwgt_mt175"]), [10., 20., 30.])
        self.assertEqual(list(events["rwgt_1"]), [1.1, 2.1, 3.1])
        self.assertEqual(list(events["part1_px"]), [7., 8., 9.])
        with open(os.path.join(self.tmp, "header.lhe")) as f:
            header = f.read()
        self.assertIn("<weight id='1'>x</weight>", header)
        self.assertIn("<weight id='mt175'>set MT 175</weight>", header)
        self.assertLess(header.index("id='1'"), header.index("id='mt175'"))
        self.assertEqual(sorted(os.listdir(self.tmp)), ["events.npy", "header.lhe"])

        # a second reweighting follows the first; the same name again is an error
        driver = self.npy_driver({"mt170": [4., 5., 6.]})
        self.driver.write_weights_in_place(driver, self.Result(["mt170"]), source, source)
        names = np.load(source).dtype.names
        self.assertEqual(names[4:6], ("rwgt_mt175", "rwgt_mt170"))
        driver = self.npy_driver({"mt170": [4., 5., 6.]})
        with self.assertRaisesRegex(ValueError, "already has a weight"):
            self.driver._merge_npy_weights(source, os.path.join(self.tmp, ".madtrex_weights"),
                                           ["mt170"], [""])

    def test_npy_without_systematics(self):
        import numpy as np
        source = os.path.join(self.tmp, "events.npy")
        events = np.zeros(2, dtype=[("weight", "<f8"), ("bias", "<f8"), ("part1_px", "<f8")])
        np.save(source, events)
        self.driver.write_weights_in_place(self.npy_driver({"a": [1., 2.]}), self.Result(["a"]),
                                           source, source)
        self.assertEqual(np.load(source).dtype.names, ("weight", "rwgt_a", "bias", "part1_px"))


class NpyToLheNamedWeightsTest(unittest.TestCase):
    """npy_to_lhe writes the named (MadtRex) weights after the numbered ones."""

    def test_named_weights(self):
        sys.path.insert(0, str(MG_ROOT))
        try:
            from madgraph.iolibs.template_files.mg7.npy_to_lhe import _add_named_weights
        finally:
            sys.path.remove(str(MG_ROOT))
        event = "<event>\n 1 2 3\n<rwgt>\n<wgt id='1'> +1.0e+00 </wgt>\n</rwgt>\n</event>\n"
        out = _add_named_weights(event, ["mt175"], [2.5])
        self.assertIn("<wgt id='1'> +1.0e+00 </wgt>\n<wgt id='mt175'> +2.5000000e+00 </wgt>\n"
                      "</rwgt>", out)
        out = _add_named_weights("<event>\n 1 2 3\n</event>\n", ["a"], [1.])
        self.assertIn("<rwgt>\n<wgt id='a'> +1.0000000e+00 </wgt>\n</rwgt>\n</event>", out)
        self.assertEqual(_add_named_weights(event, [], []), event)


@unittest.skipUnless(madtrex_is_installed(), "madspace/madtrex are not installed")
class RexHeaderParsingTest(unittest.TestCase):
    """The LHE header of an MG7 run embeds the run card unescaped, and its
    comments hold text like "<run>" or "<set>@<member>". Rex's XML parser
    used to take those for unclosed elements and nest the rest of the header
    (<initrwgt> included) inside them, so the existing weight ids were not
    found and MadtRex wrote its own weight names over the first ones."""

    LHE = textwrap.dedent("""\
        <LesHouchesEvents version="3.0">
        <header>
        <MG7RunCard>
        # plot into Events/<run>/plots; "<set>@<member>"; <group>[-<group>]-<observable>.min
        events = 10
        </MG7RunCard>
        <MG7Seed>
        42
        </MG7Seed>
        <initrwgt>
        <weightgroup name="scale" combine="envelope">
        <weight id="1" MUR="0.5"> MUR=0.5 </weight>
        <weight id="2" MUR="2.0"> MUR=2.0 </weight>
        </weightgroup>
        </initrwgt>
        </header>
        <init>
        2212 2212 6.5e+03 6.5e+03 0 0 0 0 3 1
        1.0e+00 1.0e-02 1.0e+00 1
        </init>
        <event>
        4 1 +1.0e+00 1.0e+02 7.8e-03 1.2e-01
        21 -1 0 0 501 502 0.0 0.0 5.0e+02 5.0e+02 0.0 0.0 9.0
        21 -1 0 0 502 503 0.0 0.0 -5.0e+02 5.0e+02 0.0 0.0 9.0
        6 1 1 2 501 0 1.0e+02 0.0 0.0 4.5e+02 1.73e+02 0.0 9.0
        -6 1 1 2 0 503 -1.0e+02 0.0 0.0 4.5e+02 1.73e+02 0.0 9.0
        <rwgt>
        <wgt id='1'> +1.1e+00 </wgt>
        <wgt id='2'> +0.9e+00 </wgt>
        </rwgt>
        </event>
        </LesHouchesEvents>
        """)

    def test_weight_ids_survive_unescaped_card_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.lhe")
            with open(path, "w") as f:
                f.write(self.LHE)
            result = subprocess.run(
                [sys.executable, "-I", "-c", textwrap.dedent(f"""
                    import sys
                    sys.path[:0] = [{str(MG_ROOT / "madtrex" / "install")!r},
                                    {str(MG_ROOT / "madspace" / "install")!r}]
                    import madtrex
                    lhe = madtrex.load_lhe_xml({path!r})
                    lhe.extract_weight_ids()
                    print(list(lhe.weight_ids))
                """)],
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "['1', '2']")


def load_card_module(name):
    """One of the mg7 reweight-card modules, which import nothing from
    MadGraph, loaded by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, MG7_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HelicityReweightingCardTest(unittest.TestCase):
    """"set helicity_reweighting <bool>" in the reweight cards."""

    LAUNCH = "launch --rwgt_name=x\nset MASS 6 1.75e+02\n"

    def setUp(self):
        self.param = load_card_module("param_reweight_card")

    def options(self, text):
        return self.param.parse_reweight_card(text).driver_options

    def test_unset_leaves_the_driver_default(self):
        self.assertIsNone(self.options(self.LAUNCH).helicity_reweighting)

    def test_set_in_the_preamble(self):
        for value, expected in (("false", False), ("off", False), ("true", True), ("1", True)):
            text = f"set helicity_reweighting {value}\n" + self.LAUNCH
            self.assertIs(self.options(text).helicity_reweighting, expected, value)

    def test_use_helicity_is_an_alias(self):
        text = "set use_helicity false\n" + self.LAUNCH
        self.assertIs(self.options(text).helicity_reweighting, False)

    def test_not_allowed_after_a_launch(self):
        with self.assertRaises(ValueError):
            self.param.parse_reweight_card(self.LAUNCH + "set helicity_reweighting false\n")

    def test_not_a_boolean(self):
        with self.assertRaises(ValueError):
            self.param.parse_reweight_card("set helicity_reweighting maybe\n" + self.LAUNCH)

    def test_applied_to_the_driver(self):
        calls = []

        class FakeDriver:
            def set_helicity_reweighting(self, value):
                calls.append(value)

        self.options(self.LAUNCH).apply(FakeDriver())
        self.assertEqual(calls, [])
        self.options("set helicity_reweighting false\n" + self.LAUNCH).apply(FakeDriver())
        self.assertEqual(calls, [False])

    def test_generic_card_parameter_section(self):
        generic = load_card_module("generic_reweight_card")
        sections = generic.split_by_mode(
            "set mode parameter\nset helicity_reweighting false\n" + self.LAUNCH)
        self.assertIs(self.options(sections["parameter"]).helicity_reweighting, False)

    def test_rejected_by_the_pdf_card(self):
        pdf = load_card_module("pdf_reweight_card")
        for name in ("helicity_reweighting", "use_helicity"):
            with self.assertRaisesRegex(ValueError, "parameter"):
                pdf.parse_pdf_reweight_card(
                    f"set {name} true\nlaunch\nset PDF_SET NNPDF40_lo_as_01180\n")


class ClassicVocabularyTest(unittest.TestCase):
    """The MadtRex card readers share the classic reweighting vocabulary."""

    LAUNCH = "launch --rwgt_name=x\nset MASS 6 1.75e+02\n"
    PARAM_CARD = ("BLOCK MASS # masses\n  6 1.730000e+02 # MT\n 25 1.25e+02 # MH\n"
                  "BLOCK SMINPUTS #\n  1 1.3e+02 # aEWM1\nDECAY 6 1.49 # WT\n")

    def setUp(self):
        self.param = load_card_module("param_reweight_card")
        self.pdf = load_card_module("pdf_reweight_card")
        self.generic = load_card_module("generic_reweight_card")

    def test_change_options(self):
        card = self.param.parse_reweight_card(
            "change backend auto\nchange mode LO\nchange helicity False\n"
            "change fptype d\nchange max_batch 10\nchange verbose T\n" + self.LAUNCH)
        options = card.driver_options
        self.assertEqual((options.helicity_reweighting, options.max_batch, options.verbose),
                         (False, 10, True))
        self.assertEqual(card.baseline_fptype, "d")

    def test_thread_options(self):
        options = self.param.parse_reweight_card(
            "change nb_core 4\nchange min_batch 500\n" + self.LAUNCH).driver_options
        self.assertEqual((options.nb_core, options.min_batch), (4, 500))
        # auto (or 0) and an unset min_batch leave the generation's defaults
        options = self.param.parse_reweight_card(
            "change nb_core auto\n" + self.LAUNCH).driver_options
        self.assertEqual((options.nb_core, options.min_batch), (None, None))
        self.assertIsNone(self.param.parse_reweight_card(
            "change nb_core 0\n" + self.LAUNCH).driver_options.nb_core)
        with self.assertRaisesRegex(ValueError, "nb_core"):
            self.param.parse_reweight_card("change nb_core many\n" + self.LAUNCH)
        with self.assertRaisesRegex(ValueError, "preamble"):
            self.param.parse_reweight_card(self.LAUNCH + "change min_batch 10\n")
        # no former "set" spelling, and parameter reweighting only
        self.assertNotIn("nb_core", self.param.DEPRECATED_SET_NAMES)
        with self.assertRaisesRegex(ValueError, "parameter"):
            self.pdf.parse_pdf_reweight_card("change nb_core 2\nlaunch\nset PDF_SET X\n")

    def test_thread_options_are_applied(self):
        calls = []

        class Driver:
            def __getattr__(self, name):
                return lambda *args: calls.append((name,) + args)

        self.param.parse_reweight_card(
            "change nb_core 3\nchange min_batch 50\n" + self.LAUNCH).driver_options.apply(Driver())
        self.assertEqual(calls, [("set_threads", 3), ("set_min_batch", 50)])

    def test_old_spellings_warn(self):
        with self.assertLogs("madgraph7", "WARNING") as logs:
            card = self.param.parse_reweight_card(
                "set FPTYPE f\nset helicity_reweighting false\n" + self.LAUNCH)
        self.assertEqual((card.baseline_fptype, card.driver_options.helicity_reweighting),
                         ("f", False))
        self.assertTrue(all("deprecated" in line for line in logs.output))

    def test_classic_only_lines_are_errors(self):
        for line in ("change output 2.0", "change mode NLO", "change rwgt_dir /tmp",
                     "change particle_in_density_matrix 3", "change process p p > z --add",
                     "change backend classic"):
            with self.assertRaisesRegex(ValueError, "classic", msg=line):
                self.param.parse_reweight_card(line + "\n" + self.LAUNCH)
            with self.assertRaisesRegex(ValueError, "classic", msg=line):
                self.pdf.parse_pdf_reweight_card(line + "\nlaunch\nset PDF_SET X\n")

    def test_pdf_lines_in_a_parameter_card(self):
        for card in ("change mode pdf\n" + self.LAUNCH, self.LAUNCH + "set PDF_SET X\n",
                     "change alpha_s_power 2\n" + self.LAUNCH):
            with self.assertRaisesRegex(ValueError, "pdf", msg=card):
                self.param.parse_reweight_card(card)

    def test_pdf_card_options(self):
        with self.assertLogs("madgraph7", "WARNING"):
            card = self.pdf.parse_pdf_reweight_card(
                "change mode pdf\nchange alpha_s_power 2\nset VERBOSE true\n"
                "change scale_variation_output all\nlaunch\nset PDF_SET X\n")
        self.assertEqual((card.alpha_s_power, card.verbose, card.scale_variation_output),
                         (2, True, "all"))
        for line in ("change helicity False", "change fptype d", "set MASS 6 175"):
            with self.assertRaisesRegex(ValueError, "parameter", msg=line):
                self.pdf.parse_pdf_reweight_card(line + "\nlaunch\nset PDF_SET X\n")

    def test_generic_sections(self):
        sections = self.generic.split_by_mode(
            "change helicity False\n" + self.LAUNCH
            + "change mode pdf\nlaunch\nset PDF_SET X\nchange mode LO\nlaunch\nset MT 170\n")
        self.assertEqual(list(sections), ["parameter", "pdf"])
        self.assertEqual(sections["parameter"].count("launch"), 2)
        self.assertIn("change helicity False", sections["parameter"])
        # a preamble without launches is dropped next to another section
        sections = self.generic.split_by_mode("change helicity False\nchange mode pdf\nlaunch\n")
        self.assertEqual(list(sections), ["pdf"])
        # no mode line: a plain parameter card
        self.assertEqual(list(self.generic.split_by_mode(self.LAUNCH)), ["parameter"])
        with self.assertRaisesRegex(ValueError, "classic"):
            self.generic.split_by_mode("change mode NLO\n" + self.LAUNCH)

    def test_named_parameters(self):
        card = self.param.expand_scans(self.param.parse_reweight_card(
            "launch --rwgt_name=a\nset MT scan:[170, 175]\nset param_card wt 2\n"
            "set aewm1 132\nset MASS 25 130\n"))
        launches = card.groups[0].launches
        self.assertTrue(self.param.has_parameter_names(launches))
        with self.assertRaises(ValueError):
            self.param.render_launches(launches)
        text = self.param.render_launches(
            self.param.resolve_parameter_names(launches, self.PARAM_CARD))
        self.assertEqual(text.splitlines(), [
            "launch --rwgt_name=a_1", "set MASS 6 170", "set DECAY 6 2",
            "set SMINPUTS 1 132", "set MASS 25 130",
            "launch --rwgt_name=a_2", "set MASS 6 175", "set DECAY 6 2",
            "set SMINPUTS 1 132", "set MASS 25 130"])
        with self.assertRaisesRegex(ValueError, "no parameter named"):
            self.param.resolve_parameter_names(
                self.param.parse_reweight_card("launch\nset MW 80\n").groups[0].launches,
                self.PARAM_CARD)

    def test_bang_comments(self):
        """'!' starts a comment, as '#' does (the default card's examples
        use both), at the start of a line or after whitespace."""
        card = self.param.expand_scans(self.param.parse_reweight_card(
            "! a comment line\nlaunch --rwgt_name=a  ! first weight\n"
            "set MT 173 ! named\nset MASS 25 scan:[120, 130]   ! one weight per value\n"))
        launches = card.groups[0].launches
        self.assertEqual([b.name for b in launches], ["a_1", "a_2"])
        self.assertEqual(launches[0].sets, [("MT", None, "173"), ("MASS", "25", "120")])
        pdf = self.pdf.parse_pdf_reweight_card(
            "change mode pdf ! PDF section\nlaunch\nset PDF_SET CT18NNLO ! or BEAM1_PDF_SET\n")
        self.assertEqual(len(pdf.launches), 1)
        self.assertEqual(self.generic.card_kind("launch\nset PDF_SET X ! comment\n"), "pdf")
        self.assertEqual(self.generic.strip_comment("set A 1 # x ! y"), "set A 1 ")
        self.assertEqual(self.generic.strip_comment("set A b!c"), "set A b!c")

    def test_multi_index_parameters(self):
        """'set NMIX 1 2 0.5': every index is kept, also when the
        parameter is given by its param_card name."""
        param_card = ("BLOCK NMIX #\n  1 1 9.0e-01 # NMIX11\n  1 2 1.0e-01 # NMIX12\n"
                      "BLOCK MASS #\n  6 1.73e+02 # MT\n")
        card = self.param.expand_scans(self.param.parse_reweight_card(
            "launch --rwgt_name=a\nset NMIX 1  2 0.5\nset MASS 6 175\nset nmix11 0.3\n"
            "launch --rwgt_name=b\nset NMIX 1 2 scan:[0.1, 0.2]\n"))
        launches = self.param.resolve_parameter_names(card.groups[0].launches, param_card)
        self.assertEqual(self.param.render_launches(launches).splitlines(), [
            "launch --rwgt_name=a", "set NMIX 1 2 0.5", "set MASS 6 175", "set NMIX 1 1 0.3",
            "launch --rwgt_name=b_1", "set NMIX 1 2 0.1",
            "launch --rwgt_name=b_2", "set NMIX 1 2 0.2"])

    def test_card_kind(self):
        kind = self.generic.card_kind
        self.assertEqual(kind(self.LAUNCH), "param")
        self.assertEqual(kind("change fptype d\n" + self.LAUNCH), "param")
        self.assertEqual(kind("launch\nset PDF_SET X\n"), "pdf")
        self.assertEqual(kind("change alpha_s_power 2\nlaunch\n"), "pdf")
        self.assertEqual(kind("change mode pdf\nlaunch\n"), "generic")
        self.assertEqual(kind("set mode parameter\n" + self.LAUNCH), "generic")

    def test_classic_options_in_sync_with_reweight_interface(self):
        sys.path.insert(0, str(MG_ROOT))
        try:
            from madgraph.interface.reweight_interface import (
                REWEIGHT_CHANGE_OPTIONS, DENSITY_CHANGE_OPTIONS)
            from madgraph.various.reweight_backend import SHARED_CHANGE_OPTIONS
        finally:
            sys.path.remove(str(MG_ROOT))
        self.assertEqual(self.generic.CLASSIC_ONLY_CHANGE_OPTIONS,
                         (REWEIGHT_CHANGE_OPTIONS | DENSITY_CHANGE_OPTIONS)
                         - SHARED_CHANGE_OPTIONS)


@unittest.skipUnless(madtrex_is_installed(), "madspace/madtrex are not installed")
class HelicityReweightingDefaultTest(unittest.TestCase):
    """Driver and TupperWare default to helicity reweighting."""

    def test_defaults_and_alias(self):
        result = subprocess.run(
            [sys.executable, "-I", "-c", textwrap.dedent(f"""
                import sys
                sys.path[:0] = [{str(MG_ROOT / "madtrex" / "install")!r},
                                {str(MG_ROOT / "madspace" / "install")!r}]
                import madtrex
                d = madtrex.Driver()
                t = madtrex.TupperWare()
                out = [d.helicity_reweighting(), t.helicity_reweighting()]
                d.set_use_helicity(False)
                t.set_helicity_reweighting(False)
                out += [d.helicity_reweighting(), d.use_helicity(), t.helicity_reweighting()]
                print(out)
            """)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[True, True, False, False, False]")


@unittest.skipUnless(madtrex_is_installed(), "madspace/madtrex are not installed")
class BatchSplittingTest(unittest.TestCase):
    """WareHouse.batch_size_for(): the events are split across the threads,
    in batches of at least min_batch events; an explicit max_batch wins."""

    def test_batch_size_for(self):
        result = subprocess.run(
            [sys.executable, "-I", "-c", textwrap.dedent(f"""
                import sys
                sys.path[:0] = [{str(MG_ROOT / "madtrex" / "install")!r},
                                {str(MG_ROOT / "madspace" / "install")!r}]
                import madtrex
                house = madtrex.WareHouse()
                npos = house.batch_size_for(10 ** 6)  # nothing set: no splitting
                house.set_threads(8)
                out = [house.batch_size_for(10 ** 6) == npos]
                house.set_min_batch(1000)
                out += [house.batch_size_for(n) for n in (80000, 4000, 2000)]
                out.append(house.batch_size_for(1999) == npos)  # under 2 * min_batch
                house.set_threads(1)
                out.append(house.batch_size_for(80000) == npos)  # one thread
                house.set_threads(8).set_max_batch(300)
                out.append(house.batch_size_for(80000))
                d = madtrex.Driver().set_threads(4).set_min_batch(7)
                out += [d.threads(), d.min_batch()]
                print(out)
            """)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(),
                         "[True, 10000, 1000, 1000, True, True, 300, 4, 7]")


if __name__ == "__main__":
    unittest.main()


class GridpackSourceBuildTest(unittest.TestCase):
    """A gridpack made with include_madtrex_source: bin/reweight builds
    madtrex from the included sources when there is no install, and
    rebuilds it when the install was compiled against another madspace.
    madtrex/install.py is a stand-in here, recording how it was called."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "bin").mkdir()
        shutil.copy(SCRIPT, self.root / "bin" / "reweight")
        (self.root / "madtrex").mkdir()
        (self.root / "madtrex" / "install.py").write_text(textwrap.dedent("""
            import sys
            print("INSTALLER", *sys.argv[1:])
            sys.exit(3)
        """))
        self.include = self.root / "madspace" / "install" / "madspace" / "include"
        (self.include / "madspace").mkdir(parents=True)
        (self.include / "madspace" / "umami.h").write_text("// v1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_reweight(self):
        return subprocess.run(
            [sys.executable, "-I", str(self.root / "bin" / "reweight"), "--help"],
            cwd=self.tmp.name, capture_output=True, text=True,
        )

    def install_madtrex(self, digest_of):
        """A madtrex install whose recorded madspace headers are those of
        the directory digest_of."""
        sys.path.insert(0, str(MG_ROOT / "madtrex" / "madtrex"))
        try:
            import _compat
        finally:
            sys.path.pop(0)
        package = self.root / "madtrex" / "install" / "madtrex"
        package.mkdir(parents=True)
        shutil.copy(MG_ROOT / "madtrex" / "madtrex" / "_compat.py", package)
        (package / "_build_info.json").write_text(
            '{"madspace_headers": "%s"}' % _compat.headers_digest(digest_of))

    def test_no_install_builds_from_the_sources(self):
        result = self.run_reweight()
        self.assertIn("this gridpack has no madtrex install: building it", result.stdout)
        self.assertIn("INSTALLER --yes\n", result.stdout)
        self.assertIn("madtrex installation failed", result.stderr)

    def test_install_for_another_madspace_is_rebuilt(self):
        other = self.root / "other"
        (other / "madspace").mkdir(parents=True)
        (other / "madspace" / "umami.h").write_text("// v0\n")
        self.install_madtrex(other)
        result = self.run_reweight()
        self.assertIn("built against another madspace", result.stdout)
        self.assertIn("INSTALLER --yes --clean", result.stdout)

    def test_matching_install_is_not_rebuilt(self):
        self.install_madtrex(self.include)
        self.assertNotIn("INSTALLER", self.run_reweight().stdout)

    def test_precompiled_madspace_cannot_build_madtrex(self):
        shutil.rmtree(self.include)
        result = self.run_reweight()
        self.assertNotIn("INSTALLER", result.stdout)
        self.assertIn("include_madspace_source = true", result.stderr)

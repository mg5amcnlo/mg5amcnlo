# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"


def _run_isolated(
    source: str,
    *,
    script_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(SOURCE_ROOT), existing_pythonpath) if entry
    )
    if script_path is None:
        command = (sys.executable, "-c", textwrap.dedent(source))
    else:
        script_path.write_text(textwrap.dedent(source), encoding="utf-8")
        command = (sys.executable, str(script_path))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def test_root_import_only_adds_symbolica_to_lightweight_public_imports() -> None:
    _run_isolated(
        """
        import importlib.util
        import sys
        import pyamplicol

        assert pyamplicol.__version__
        assert set(pyamplicol.__all__) == {
            *pyamplicol._PUBLIC_EXPORTS,
            "__version__",
        }
        assert "Generator" in dir(pyamplicol)
        assert "pyamplicol.api" not in sys.modules
        assert "pyamplicol.config" not in sys.modules
        assert ("symbolica" in sys.modules) == (
            importlib.util.find_spec("symbolica") is not None
        )
        assert not any(
            name == "pyamplicol.models" or name.startswith("pyamplicol.models.")
            for name in sys.modules
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("ufo_model_loader",)
        )
        """
    )


def test_lightweight_public_exports_do_not_load_model_tooling() -> None:
    _run_isolated(
        """
        import importlib.util
        import sys
        from pyamplicol import (
            ColorConfig,
            ColorFFTBasis,
            CorrelatedRequest,
            Generator,
            ModelSource,
            ProcessRequest,
            ProcessSet,
            Runtime,
        )
        import pyamplicol.api as public_api
        from pyamplicol.api import ModelSource as ApiModelSource
        from pyamplicol.api.requests import ProcessRequest as DirectProcessRequest

        assert set(public_api.__all__) == set(public_api._PUBLIC_EXPORTS)
        request = ProcessRequest.parse("d d~ > z")
        process_set = ProcessSet((request,))
        assert process_set.requests == (request,)
        assert ModelSource is ApiModelSource
        assert ProcessRequest is DirectProcessRequest
        assert ModelSource.built_in_sm().kind == "built-in-sm"
        assert Generator.__module__ == "pyamplicol.api.services"
        assert Runtime.__module__ == "pyamplicol.api.services"
        assert CorrelatedRequest().color_correlation == "born"
        assert ColorConfig().fft_basis is ColorFFTBasis.ADJOINT
        assert "pyamplicol.models.loading" not in sys.modules
        assert ("symbolica" in sys.modules) == (
            importlib.util.find_spec("symbolica") is not None
        )
        assert not any(
            name.startswith("pyamplicol.models.compiler")
            for name in sys.modules
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("ufo_model_loader",)
        )
        """
    )


def test_compiled_model_export_resolves_to_the_canonical_type() -> None:
    _run_isolated(
        """
        from typing import get_type_hints

        from pyamplicol import ModelSource
        from pyamplicol import CompiledModel as RootCompiledModel
        from pyamplicol.api import CompiledModel as ApiCompiledModel
        from pyamplicol.api.models import CompiledModel as ModelCompiledModel

        assert RootCompiledModel is ModelCompiledModel
        assert ApiCompiledModel is ModelCompiledModel
        assert get_type_hints(ModelSource.compile)["return"] is ModelCompiledModel
        assert "pyamplicol.models.loading" not in __import__("sys").modules
        """
    )


def test_cli_help_construction_does_not_load_model_tooling() -> None:
    _run_isolated(
        """
        import importlib.util
        import sys
        from pyamplicol.cli.parser import build_parser

        help_text = build_parser().format_help()
        assert "generate" in help_text
        assert "model" in help_text
        assert "pyamplicol.models.loading" not in sys.modules
        assert ("symbolica" in sys.modules) == (
            importlib.util.find_spec("symbolica") is not None
        )
        assert not any(
            name.startswith("pyamplicol.models.compiler")
            for name in sys.modules
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("ufo_model_loader",)
        )
        """
    )


@pytest.mark.parametrize(
    "registration", ("accepted", "ValueError", "PermissionError", "unavailable")
)
@pytest.mark.parametrize("personal_license", (False, True))
def test_package_registration_preserves_authoritative_personal_and_restricted_fallback(
    registration: str,
    personal_license: bool,
) -> None:
    completed = _run_isolated(
        f"""
        import importlib.abc
        import importlib.util
        import io
        import os
        import sys

        registration = {registration!r}
        personal_license = {personal_license!r}
        calls = []
        if personal_license:
            os.environ["SYMBOLICA_LICENSE"] = "test-personal-license"
        else:
            os.environ.pop("SYMBOLICA_LICENSE", None)
        original_personal = os.environ.get("SYMBOLICA_LICENSE")
        os.environ["SYMBOLICA_HIDE_BANNER"] = "0"

        class SymbolicaLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path, target=None):
                if fullname == "symbolica":
                    return importlib.util.spec_from_loader(fullname, self)

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                assert os.environ["SYMBOLICA_HIDE_BANNER"] == "1"

                def register(key):
                    assert sys._getframe(1).f_globals["__name__"] == "pyamplicol"
                    assert isinstance(key, str) and key
                    calls.append("register")
                    if registration == "ValueError":
                        raise ValueError("test OEM rejection")
                    if registration == "PermissionError":
                        raise PermissionError("test OEM denial")

                def is_licensed():
                    caller = sys._getframe(1).f_globals["__name__"]
                    assert caller == "pyamplicol.licensing"
                    calls.append("is_licensed")
                    return registration == "accepted" or personal_license

                if registration != "unavailable":
                    module.set_library_key = register
                module.is_licensed = is_licensed
                self.original_api = dict(vars(module))

        loader = SymbolicaLoader()
        sys.meta_path.insert(0, loader)
        import pyamplicol

        assert os.environ["SYMBOLICA_HIDE_BANNER"] == "0"
        assert os.environ.get("SYMBOLICA_LICENSE") == original_personal
        assert calls == ([] if registration == "unavailable" else ["register"])
        assert "pyamplicol.config" not in sys.modules
        from pyamplicol.licensing import (
            detect_symbolica_license, symbolica_resource_clamps,
        )
        from pyamplicol.config import RunConfig

        stream = io.StringIO()
        state = detect_symbolica_license(stream=stream)
        expected_licensed = registration == "accepted" or personal_license
        assert state.licensed is expected_licensed
        assert state.restricted is not expected_licensed
        assert calls[-1] == "is_licensed"
        # Only the public registration/check APIs are used; no upstream licence
        # checker or private bypass flag is replaced to manufacture success.
        assert vars(sys.modules["symbolica"]) == loader.original_api
        clamps = symbolica_resource_clamps(RunConfig(action="generate"), state)
        if expected_licensed:
            assert clamps == ()
            assert stream.getvalue() == ""
        else:
            assert [(item.path, item.effective) for item in clamps] == [
                ("generation.workers", 1), ("evaluator.optimization.cores", 1)
            ]
            assert "request-symbolica-trial-license" in stream.getvalue()
        json_stream = io.StringIO()
        assert detect_symbolica_license(json_mode=True, stream=json_stream) == state
        assert json_stream.getvalue() == ""
        assert os.environ["SYMBOLICA_HIDE_BANNER"] == "1"
        """
    )
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_registration_does_not_hide_unexpected_failures() -> None:
    _run_isolated(
        """
        import os
        import sys
        from types import ModuleType

        module = ModuleType("symbolica")
        def fail(key):
            raise RuntimeError("unexpected registration failure")
        module.set_library_key = fail
        sys.modules["symbolica"] = module
        os.environ.pop("SYMBOLICA_HIDE_BANNER", None)
        try:
            import pyamplicol
        except RuntimeError as error:
            assert str(error) == "unexpected registration failure"
        else:
            raise AssertionError("unexpected registration error was swallowed")
        assert "SYMBOLICA_HIDE_BANNER" not in os.environ
        """
    )


@pytest.mark.parametrize(
    "missing_module", ("symbolica", "symbolica.core", "transitive")
)
def test_only_missing_symbolica_is_optional_for_lightweight_imports(
    missing_module: str,
) -> None:
    _run_isolated(
        f"""
        import importlib.abc
        import os
        import sys

        missing_module = {missing_module!r}
        class MissingSymbolica(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "symbolica":
                    raise ModuleNotFoundError(
                        "test missing module", name=missing_module,
                    )

        sys.meta_path.insert(0, MissingSymbolica())
        os.environ["SYMBOLICA_HIDE_BANNER"] = "0"
        try:
            from pyamplicol import Generator, Runtime
            from pyamplicol.config import RunConfig
            from pyamplicol.cli.parser import build_parser
        except ModuleNotFoundError as error:
            assert missing_module != "symbolica"
            assert error.name == missing_module
        else:
            assert missing_module == "symbolica"
            assert "symbolica" not in sys.modules
            assert RunConfig(action="generate").action == "generate"
            assert "generate" in build_parser().format_help()
            # Missing Symbolica is not silently reported as a valid/restricted
            # installation when symbolic work requests the actual license state.
            from pyamplicol.licensing import detect_symbolica_license
            try:
                detect_symbolica_license()
            except ModuleNotFoundError as error:
                assert error.name == "symbolica"
            else:
                raise AssertionError("Symbolica-dependent operation should fail")
        assert os.environ["SYMBOLICA_HIDE_BANNER"] == "0"
        """
    )


def test_spawned_worker_registers_its_own_package_license(tmp_path: Path) -> None:
    _run_isolated(
        """
        import multiprocessing
        import os
        import sys
        from types import ModuleType

        registration_pids = []
        symbolica = ModuleType("symbolica")
        def register(key):
            assert isinstance(key, str) and key
            registration_pids.append(os.getpid())
        symbolica.set_library_key = register
        sys.modules["symbolica"] = symbolica
        import pyamplicol

        def worker(queue):
            queue.put((os.getpid(), registration_pids))

        if __name__ == "__main__":
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            child = context.Process(target=worker, args=(queue,))
            child.start()
            try:
                child_pid, child_registrations = queue.get(timeout=20)
                child.join(timeout=20)
                assert child.exitcode == 0
                assert registration_pids == [os.getpid()]
                assert child_pid != os.getpid()
                assert child_registrations == [child_pid]
            finally:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=5)
                queue.close()
        """,
        script_path=tmp_path / "spawn_license_probe.py",
    )

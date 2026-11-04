# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from pyamplicol.config import PyAmpliColError
from pyamplicol.licensing import detect_symbolica_license
from pyamplicol.reporting import (
    TtyProgressSink,
    close_progress_sink,
    configure_cli_logging,
    get_logger,
    progress_sink,
    render_summary,
    reset_cli_logging,
)

from .handlers import CliServices, DefaultCliServices, dispatch
from .licensing import LicenseRequestInvocation
from .parser import parse_cli
from .utilities import UtilityInvocation, example_card, execute_utility


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(entry) for key, entry in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(entry) for entry in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize CLI result of type {type(value).__name__}")


def write_result(
    value: object,
    *,
    format: str,
    stream: TextIO,
    color: bool = False,
) -> None:
    if value is None:
        return
    plain = _json_value(value)
    if format == "json":
        json.dump(plain, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    elif format == "human":
        rendered = render_summary(value, color=color)
        if rendered is None:
            raise RuntimeError(
                "human table rendering is unavailable; use --json for "
                "machine-readable output"
            )
        stream.write(f"{rendered}\n")
    else:
        raise ValueError(f"unsupported output format {format!r}")
    stream.flush()


def _color_enabled(mode: str, stream: TextIO) -> bool:
    return mode == "always" or (
        mode == "auto" and bool(getattr(stream, "isatty", lambda: False)())
    )


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    services: CliServices | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = sys.stdout if stdout is None else stdout
    diagnostic_stream = sys.stderr if stderr is None else stderr
    input_stream = sys.stdin if stdin is None else stdin
    logging_configured = False
    sink = None
    try:
        invocation = parse_cli(argv)
        if isinstance(invocation, LicenseRequestInvocation):
            result = invocation.run(stdin=input_stream, stdout=output_stream)
            write_result(
                result,
                format=invocation.output_format,
                stream=output_stream,
                color=_color_enabled(invocation.output_color, output_stream),
            )
            return 0
        if isinstance(invocation, UtilityInvocation):
            if invocation.kind == "examples-run":
                assert invocation.name is not None
                arguments = [str(example_card(invocation.name))]
                for override in invocation.overrides:
                    arguments.extend(("--set", override))
                if invocation.output_format == "json":
                    arguments.append("--json")
                arguments.extend(("--color", invocation.output_color))
                invocation = parse_cli(arguments)
                assert not isinstance(
                    invocation, (LicenseRequestInvocation, UtilityInvocation)
                )
            else:
                result = execute_utility(invocation)
                write_result(
                    result,
                    format=invocation.output_format,
                    stream=output_stream,
                    color=_color_enabled(invocation.output_color, output_stream),
                )
                if invocation.kind in {"doctor", "self-test"}:
                    return 0 if bool(getattr(result, "ok", False)) else 1
                return 0
        resolution = invocation.resolve()
        if services is None and resolution.effective.action == "model-compile":
            initial = resolution.effective
            detect_symbolica_license(
                suggest=initial.symbolica.suggest_license,
                json_mode=initial.output.format == "json",
                stream=diagnostic_stream,
            )
        config = resolution.effective
        progress_color = _color_enabled(config.output.color, diagnostic_stream)
        sink = progress_sink(
            config.output.progress,
            stream=diagnostic_stream,
            logger=get_logger("progress"),
            color=progress_color,
        )
        logging_stream = (
            sink.message_stream
            if isinstance(sink, TtyProgressSink)
            else diagnostic_stream
        )
        configure_cli_logging(
            config.output.log_level,
            stream=logging_stream,
            color=progress_color,
        )
        logging_configured = True
        selected_services = (
            DefaultCliServices(resolution=resolution) if services is None else services
        )
        if invocation.correlators is not None:
            from pyamplicol.correlators import CorrelatorConfig

            if invocation.dry_run:
                raise ValueError("--correlators requires generation, not --dry-run")
            if services is not None:
                raise ValueError(
                    "--correlators requires the built-in generation service"
                )
            declarations = CorrelatorConfig.from_json_dict(
                json.loads(invocation.correlators.read_text())
            )
            selected_services = DefaultCliServices(
                resolution=resolution, correlators=declarations
            )
        result = dispatch(
            config,
            selected_services,
            sink,
            dry_run=invocation.dry_run,
            full_physics=invocation.full_physics,
        )
        close_progress_sink(sink)
        sink = None
        color = _color_enabled(config.output.color, output_stream)
        write_result(
            result,
            format=config.output.format,
            stream=output_stream,
            color=color,
        )
        return 130 if bool(getattr(result, "interrupted", False)) else 0
    except (PyAmpliColError, OSError, RuntimeError, TypeError, ValueError) as exc:
        close_progress_sink(sink)
        sink = None
        if logging_configured:
            get_logger("cli").error("%s", exc)
        else:
            diagnostic_stream.write(f"error: {exc}\n")
            diagnostic_stream.flush()
        return 2
    except KeyboardInterrupt:
        close_progress_sink(sink)
        sink = None
        message = "interrupted before a complete result was available"
        if logging_configured:
            get_logger("cli").warning("%s", message)
        else:
            diagnostic_stream.write(f"{message}\n")
            diagnostic_stream.flush()
        return 130
    finally:
        close_progress_sink(sink)
        if logging_configured:
            reset_cli_logging()


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


__all__ = ["main", "run_cli", "write_result"]

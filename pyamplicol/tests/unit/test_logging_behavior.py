# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import io
import logging

from pyamplicol.reporting import (
    configure_cli_logging,
    get_logger,
    reset_cli_logging,
)

logging_module = importlib.import_module("pyamplicol.reporting.logging")


def test_cli_logging_is_package_local_idempotent_and_resettable() -> None:
    root = logging.getLogger()
    root_handlers = tuple(root.handlers)
    root_level = root.level
    stream = io.StringIO()
    reset_cli_logging()
    try:
        logger = configure_cli_logging("warning", stream=stream)
        configure_cli_logging("warning", stream=stream)
        owned = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_pyamplicol_cli_handler", False)
        ]
        assert len(owned) == 1
        get_logger("unit").warning("diagnostic")
        assert "diagnostic" in stream.getvalue()
        assert tuple(root.handlers) == root_handlers
        assert root.level == root_level
    finally:
        reset_cli_logging()
    assert not any(
        getattr(handler, "_pyamplicol_cli_handler", False)
        for handler in get_logger().handlers
    )


def test_cli_logging_restores_embedding_application_state() -> None:
    logger = get_logger()
    reset_cli_logging()
    original_level = logger.level
    original_propagate = logger.propagate
    external_handler = logging.NullHandler()
    logger.addHandler(external_handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    try:
        configure_cli_logging(logging.INFO, stream=io.StringIO())
        assert logger.level == logging.INFO
        assert logger.propagate is False

        reset_cli_logging()

        assert logger.level == logging.ERROR
        assert logger.propagate is False
        assert external_handler in logger.handlers
    finally:
        reset_cli_logging()
        logger.removeHandler(external_handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_cli_logging_colors_terminal_warnings_without_coloring_plain_output() -> None:
    colored = io.StringIO()
    plain = io.StringIO()
    reset_cli_logging()
    try:
        configure_cli_logging("warning", stream=colored, color=True)
        get_logger("generation").warning("skipped unsupported process")
        configure_cli_logging("warning", stream=plain, color=False)
        get_logger("generation").warning("skipped unsupported process")
    finally:
        reset_cli_logging()

    assert colored.getvalue().startswith("\x1b[33mWARNING pyamplicol.generation")
    assert colored.getvalue().endswith("\x1b[0m\n")
    assert plain.getvalue() == (
        "WARNING pyamplicol.generation: skipped unsupported process\n"
    )


def test_cli_logging_falls_back_to_plain_text_without_colorama(monkeypatch) -> None:
    stream = io.StringIO()
    real_import_module = logging_module.importlib.import_module

    def import_without_colorama(name: str) -> object:
        if name == "colorama":
            raise ImportError("colorama unavailable")
        return real_import_module(name)

    monkeypatch.setattr(
        logging_module.importlib,
        "import_module",
        import_without_colorama,
    )
    reset_cli_logging()
    try:
        configure_cli_logging("warning", stream=stream, color=True)
        get_logger("generation").warning("skipped unsupported process")
    finally:
        reset_cli_logging()

    assert stream.getvalue() == (
        "WARNING pyamplicol.generation: skipped unsupported process\n"
    )

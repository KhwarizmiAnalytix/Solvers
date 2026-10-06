#!/usr/bin/env python3
"""Run Solvers' Bazel workflow with the Logging-style dotted CLI.

The Bazel BUILD graph is intentionally kept separate from CMake. Once the
repository has BUILD.bazel targets, these commands are available:
    python Scripts/setup_bazel.py build
    python Scripts/setup_bazel.py build.test.release
    python Scripts/setup_bazel.py build.test.benchmark.coverage.iwyu.spell
    python Scripts/setup_bazel.py clean

Supported tokens:
  build       Build all targets (//...).  Implied by test/coverage/benchmark.
  test        Run unit tests    (//Testing/Cxx/...).
  coverage    Run unit tests with coverage instrumentation (//Testing/Cxx/...).
  benchmark   Run benchmarks    (//Testing/Benchmark/...).  Always a separate
              `bazel test` phase so benchmark timings are never mixed with the
              unit-test or coverage pass.
  release     Optimised build (-c opt).
  debug       Debug build (-c dbg).
  cxx17       Force -std=c++17 (default in .bazelrc).
  cxx20       Force -std=c++20.
  iwyu        Run include-what-you-use analysis as a separate `bazel build
              --config=iwyu //...` phase after all other phases.
  spell       Run codespell on the source tree before any Bazel phase.
  ceres       Accepted for CLI compatibility; Ceres Bazel deps not yet wired up.
  ipopt       Accepted for CLI compatibility; Ipopt Bazel deps not yet wired up.
  petsc       Accepted for CLI compatibility; PETSc Bazel deps not yet wired up.
  clean       Runs `bazel clean` and exits.

Phase order: spell → build → test → coverage → benchmark → iwyu.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_UNIMPLEMENTED_TOKENS = {"ceres", "ipopt", "petsc"}

_TEST_TARGET = "//Testing/Cxx/..."
_BENCH_TARGET = "//Testing/Benchmark/..."
_ALL_TARGET = "//..."


def print_status(message: str, level: str = "INFO") -> None:
    print(f"[{level}] {message}")


def split_tokens(arguments: list[str]) -> set[str]:
    tokens: set[str] = set()
    for argument in arguments:
        tokens.update(part.lower() for part in argument.split(".") if part)
    return tokens


def bazel_command() -> str:
    for name in ("bazelisk", "bazel"):
        if exe := shutil.which(name):
            return exe
    raise RuntimeError("Neither bazelisk nor bazel was found in PATH")


def run(command: list[str]) -> None:
    print_status("$ " + " ".join(command))
    sys.stdout.flush()
    result = subprocess.run(command, cwd=ROOT)
    # Bazel exit code 4: build OK but no tests found — treat as success.
    if result.returncode not in (0, 4):
        result.check_returncode()


def run_spell() -> None:
    codespell = shutil.which("codespell")
    if not codespell:
        print_status("codespell not found in PATH; skipping spell check", "WARN")
        return
    run([codespell, str(ROOT)])


def _base_flags(tokens: set[str]) -> list[str]:
    flags: list[str] = []
    if "release" in tokens:
        flags.extend(["-c", "opt"])
    elif "debug" in tokens:
        flags.extend(["-c", "dbg"])
    if "cxx17" in tokens:
        flags.append("--cxxopt=-std=c++17")
    if "cxx20" in tokens:
        flags.append("--cxxopt=-std=c++20")
    return flags


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("commands", nargs="*", help="Dotted commands and options")
    parser.add_argument("--target", default=_ALL_TARGET, help="Bazel target pattern")
    options = parser.parse_args(arguments)
    tokens = split_tokens(options.commands)
    if not tokens:
        parser.print_help()
        return 0

    for tok in sorted(_UNIMPLEMENTED_TOKENS & tokens):
        print_status(
            f"Token '{tok}': third-party Bazel deps not yet wired up — no effect.",
            "WARN",
        )

    try:
        bazel = bazel_command()

        if "clean" in tokens:
            run([bazel, "clean"])
            return 0

        # ── Phase 0: spell ────────────────────────────────────────────────────
        if "spell" in tokens:
            run_spell()

        flags = _base_flags(tokens)

        # ── Phase 1: build ────────────────────────────────────────────────────
        # Explicit 'build' token, or default when no run-phase token is given.
        run_phases = tokens & {"test", "coverage", "benchmark"}
        if "build" in tokens or not run_phases:
            run([bazel, "build"] + flags + [options.target])

        # ── Phase 2: test (unit tests only) ───────────────────────────────────
        if "test" in tokens:
            target = _TEST_TARGET if options.target == _ALL_TARGET else options.target
            run([bazel, "test"] + flags + [target])

        # ── Phase 3: coverage (unit tests with instrumentation) ───────────────
        if "coverage" in tokens:
            target = _TEST_TARGET if options.target == _ALL_TARGET else options.target
            run([bazel, "coverage"] + flags + [target])

        # ── Phase 4: benchmark (runs the benchmark suite) ─────────────────────
        if "benchmark" in tokens:
            target = _BENCH_TARGET if options.target == _ALL_TARGET else options.target
            run([bazel, "test"] + flags + [target])

        # ── Phase 5: iwyu ─────────────────────────────────────────────────────
        if "iwyu" in tokens:
            run([bazel, "build", "--config=iwyu"] + flags + [options.target])

    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print_status(str(error), "ERROR")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

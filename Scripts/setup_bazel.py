#!/usr/bin/env python3
"""Run Solvers' Bazel workflow with the Logging-style dotted CLI.

The Bazel BUILD graph is intentionally kept separate from CMake. Once the
repository has BUILD.bazel targets, these commands are available:
    python Scripts/setup_bazel.py build
    python Scripts/setup_bazel.py build.test.release
    python Scripts/setup_bazel.py clean
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def print_status(message: str, level: str = "INFO") -> None:
    print(f"[{level}] {message}")


def split_tokens(arguments: list[str]) -> set[str]:
    tokens: set[str] = set()
    for argument in arguments:
        tokens.update(part.lower() for part in argument.split(".") if part)
    return tokens


def bazel_command() -> str:
    for name in ("bazelisk", "bazel"):
        executable = shutil.which(name)
        if executable:
            return executable
    raise RuntimeError("Neither bazelisk nor bazel was found in PATH")


def run(command: list[str]) -> None:
    print_status("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands", nargs="*", help="Dotted commands and options")
    parser.add_argument("--target", default="//...", help="Bazel target pattern")
    options = parser.parse_args(arguments)
    tokens = split_tokens(options.commands)
    if not tokens:
        parser.print_help()
        return 0

    try:
        bazel = bazel_command()
        if "clean" in tokens:
            run([bazel, "clean"])
            return 0

        action = "test" if "test" in tokens else "build"
        command = [bazel, action]
        if "release" in tokens:
            command.extend(["-c", "opt"])
        elif "debug" in tokens:
            command.extend(["-c", "dbg"])
        if "cxx17" in tokens:
            command.append("--cxxopt=-std=c++17")
        if "cxx20" in tokens:
            command.append("--cxxopt=-std=c++20")
        command.append(options.target)
        run(command)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print_status(str(error), "ERROR")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

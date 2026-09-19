#!/usr/bin/env python3
"""Configure, build, and test Solvers with dotted Logging-style commands.

Examples:
    python Scripts/setup.py config.build.test
    python Scripts/setup.py config.build.test.release
    python Scripts/setup.py config.build.test.ceres.nlopt
    python Scripts/setup.py clean
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "build-solvers"


def print_status(message: str, level: str = "INFO") -> None:
    print(f"[{level}] {message}")


def split_tokens(arguments: list[str]) -> set[str]:
    tokens: set[str] = set()
    for argument in arguments:
        tokens.update(part.lower() for part in argument.split(".") if part)
    return tokens


def find_program(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required program not found: {name}")
    return executable


def run(command: list[str], cwd: Path = ROOT) -> None:
    print_status("$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def cmake_args(tokens: set[str], build_dir: Path) -> list[str]:
    build_type = "Release" if "release" in tokens else "RelWithDebInfo" if "relwithdebinfo" in tokens else "Debug"
    # The benchmark target lives in Testing/Cxx, which the top-level CMakeLists
    # only add_subdirectory()s when testing is enabled, so requesting the
    # benchmark implies testing even if the "test" token was left off.
    enable_testing = "test" in tokens or "testing" in tokens or "benchmark" in tokens
    args = [
        "-S", str(ROOT),
        "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DSOLVERS_ENABLE_TESTING={'ON' if enable_testing else 'OFF'}",
        f"-DSOLVERS_ENABLE_CERES={'ON' if 'ceres' in tokens else 'OFF'}",
        f"-DSOLVERS_ENABLE_NLOPT={'ON' if 'nlopt' in tokens else 'OFF'}",
        f"-DSOLVERS_ENABLE_BENCHMARKS={'ON' if 'benchmark' in tokens else 'OFF'}",
        f"-DSOLVERS_ENABLE_SANITIZER={'ON' if 'sanitizer' in tokens or 'asan' in tokens else 'OFF'}",
        f"-DSOLVERS_ENABLE_COVERAGE={'ON' if 'coverage' in tokens else 'OFF'}",
        f"-DBUILD_SHARED_LIBS={'ON' if 'shared' in tokens else 'OFF'}",
    ]
    if "gcc" in tokens:
        args.extend(["-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++"])
    elif "clang" in tokens and platform.system() != "Windows":
        args.extend(["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"])
    return args


def find_built_executable(build_dir: Path, name: str) -> Path:
    exe_name = f"{name}.exe" if platform.system() == "Windows" else name
    # Multi-config generators (Visual Studio, Xcode) nest the binary under a
    # per-config directory; single-config ones (Ninja, Make) do not.
    for candidate in (
        build_dir / "Testing" / "Cxx" / exe_name,
        build_dir / "Testing" / "Cxx" / "Release" / exe_name,
        build_dir / "Testing" / "Cxx" / "RelWithDebInfo" / exe_name,
        build_dir / "Testing" / "Cxx" / "Debug" / exe_name,
    ):
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find built executable {name!r} under {build_dir}; was it built?")


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands", nargs="*", help="Dotted commands and options")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    options = parser.parse_args(arguments)
    tokens = split_tokens(options.commands)
    if not tokens:
        parser.print_help()
        return 0

    build_dir = options.build_dir if options.build_dir.is_absolute() else ROOT / options.build_dir
    try:
        if "clean" in tokens:
            if build_dir.exists():
                shutil.rmtree(build_dir)
            print_status(f"Removed {build_dir}")
            return 0

        find_program("cmake")
        if "config" in tokens or "build" in tokens or "test" in tokens or "coverage" in tokens:
            run(["cmake", *cmake_args(tokens, build_dir)])
        if "build" in tokens:
            run(["cmake", "--build", str(build_dir), "--parallel"])
        if "test" in tokens:
            run(["ctest", "--test-dir", str(build_dir), "--output-on-failure"])
        if "benchmark" in tokens:
            if "build" not in tokens:
                print_status("'benchmark' requires 'build'; add .build to actually compile it", "ERROR")
                return 1
            print_status("Running solver backend benchmark (LM/LBFGS vs. NLopt/Ceres)")
            run([str(find_built_executable(build_dir, "SolversBenchmark"))])
        if "coverage" in tokens:
            print_status("Coverage instrumentation is enabled; use the repository coverage tooling to collect reports.")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print_status(str(error), "ERROR")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

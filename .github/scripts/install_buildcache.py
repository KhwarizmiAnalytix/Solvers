"""Build a pinned buildcache on the runner instead of relying on package aliases."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    source, build = prefix / "source", prefix / "build"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "v0.33.1",
            "--recurse-submodules",
            "https://gitlab.com/bits-n-bites/buildcache.git",
            str(source),
        ],
        check=True,
    )
    # BuildCache v0.33.1 relies on <string> to declare int64_t, which no
    # longer works with GCC 14 on the Windows runner. Patch only this
    # temporary checkout and explicitly include the type's own header.
    header = source / "src" / "base" / "env_utils.hpp"
    header.write_text(
        header.read_text(encoding="utf-8").replace(
            "#include <string>",
            "// Solvers CI compatibility fix for GCC 14.\n#include <stdint.h>\n#include <string>",
            1,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source / "src"),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            # CMake 4 rejects cmake_minimum_required() below 3.5, and warns
            # that below 3.10 is next. The pinned buildcache still vendors
            # mingw-std-threads at 3.0; this raises only that older floor.
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.10",
            # CI uses local storage; avoid runner-specific OpenSSL DLLs.
            "-DDISABLE_SSL=ON",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "buildcache", "--parallel", "4"],
        check=True,
    )
    executable = "buildcache.exe" if os.name == "nt" else "buildcache"
    binaries = [path for path in build.rglob(executable) if path.is_file()]
    if not binaries:
        raise RuntimeError("buildcache built without producing the requested executable")
    destination = prefix / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binaries[0], destination / executable)
    for pattern in ("*.dll", "*.dylib", "*.so*"):
        for library in build.rglob(pattern):
            if library.is_file():
                shutil.copy2(library, destination / library.name)
    subprocess.run([str(destination / executable), "--version"], check=True)
    with Path(os.environ["GITHUB_PATH"]).open("a", encoding="utf-8") as output:
        output.write(str(destination) + "\n")


if __name__ == "__main__":
    main()

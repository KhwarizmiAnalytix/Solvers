# Solvers Example

Standalone project demonstrating `find_package(Solvers)` after installation.

## Steps

1. Build and install Solvers:

```bash
cd /path/to/Solvers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix /path/to/install
```

2. Build the example against the installed package:

```bash
cd /path/to/Solvers/example
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build
```

3. Run:

```bash
./build/root_finding
./build/least_squares
```

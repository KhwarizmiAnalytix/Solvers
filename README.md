# Quarisma Math Library

![C++17](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)

High-performance mathematical functions and numerical algorithms for quantitative finance.

Developed and maintained by [QuarismAnalytix](https://github.com/QuarismAnalytix)

## Features

- Modern C++17 mathematical functions
- Linear algebra operations
- Numerical optimization algorithms
- Special functions for finance
- Matrix operations with GPU support
- Cross-platform compatibility
- Integration with Quarisma quantitative analysis ecosystem

## Source organization

Numerical grid and stencil algorithms are grouped under `numerical_operation/`:

- `discretization`
- `finite_difference`
- `tridiagonal_operations`
- `pentadiagonal_operations`

The `common/` directory is reserved for shared algorithms that do not belong to
a more specific mathematical component. New numerical algorithms should be
placed in the closest named component instead of `common/`.

## Installation

### Using CMake

```cmake
find_package(QuarismaMath REQUIRED)
target_link_libraries(your_target PRIVATE Quarisma::Math)
```

### Building from Source

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Dependencies

- Quarisma::Core
- Quarisma::Serialization
- Quarisma::Vectorization

## License

BSD-3-Clause License

Copyright (c) 2024 QuarismAnalytix

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Author

**QuarismAnalytix**
- GitHub: [@QuarismAnalytix](https://github.com/QuarismAnalytix)
- Organization: [QuarismAnalytix](https://github.com/QuarismAnalytix)

cmake_minimum_required(VERSION 3.20)
project(Solvers VERSION 1.0.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cmake")
option(SOLVERS_ENABLE_TESTING "Build the GoogleTest solver suite" ${PROJECT_IS_TOP_LEVEL})
option(SOLVERS_ENABLE_CERES "Build the optional Ceres backend" OFF)
option(SOLVERS_ENABLE_NLOPT "Build the optional NLopt backend" OFF)
# Ipopt (interior-point NLP) and PETSc/TAO (which also realizes POUNDERS) are
# system libraries discovered through pkg-config rather than bundled submodules
# (docs/reviews/solver-redesign-2026-09-26.md). Native LM never depends on them;
# when OFF the adapters report backend_unavailable at runtime.
option(SOLVERS_ENABLE_IPOPT "Build the optional Ipopt backend" OFF)
option(SOLVERS_ENABLE_PETSC "Build the optional PETSc/TAO backend" OFF)
option(SOLVERS_ENABLE_SANITIZER "Enable address and undefined behavior sanitizers" OFF)
option(SOLVERS_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(SOLVERS_ENABLE_CLANGTIDY "Enable clang-tidy static analysis for Solvers" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(SolversDependencies)

file(GLOB_RECURSE solvers_sources CONFIGURE_DEPENDS "src/*.cpp")
add_library(Solvers ${solvers_sources})
add_library(Solvers::Solvers ALIAS Solvers)
target_include_directories(Solvers
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_compile_features(Solvers PUBLIC cxx_std_17)
target_link_libraries(Solvers PUBLIC Eigen3::Eigen Logging::Logging)
target_compile_definitions(Solvers PUBLIC
  SOLVERS_HAS_CERES=$<BOOL:${SOLVERS_ENABLE_CERES}>
  SOLVERS_HAS_NLOPT=$<BOOL:${SOLVERS_ENABLE_NLOPT}>
  SOLVERS_HAS_IPOPT=$<BOOL:${SOLVERS_ENABLE_IPOPT}>
  SOLVERS_HAS_PETSC=$<BOOL:${SOLVERS_ENABLE_PETSC}>)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(Solvers PUBLIC SOLVERS_SHARED PRIVATE SOLVERS_BUILDING)
  set_target_properties(Solvers PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(SOLVERS_ENABLE_CERES)
  target_link_libraries(Solvers PRIVATE Ceres::ceres)
endif()
if(SOLVERS_ENABLE_NLOPT)
  target_link_libraries(Solvers PRIVATE nlopt)
endif()
if(SOLVERS_ENABLE_IPOPT OR SOLVERS_ENABLE_PETSC)
  find_package(PkgConfig REQUIRED)
endif()
if(SOLVERS_ENABLE_IPOPT)
  # Ipopt ships an ipopt.pc; the import target carries its include dirs and libs.
  pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)
  target_link_libraries(Solvers PRIVATE PkgConfig::IPOPT)
endif()
if(SOLVERS_ENABLE_PETSC)
  # PETSc (with TAO) ships a PETSc.pc / petsc.pc.
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
  target_link_libraries(Solvers PRIVATE PkgConfig::PETSC)
endif()
if(SOLVERS_ENABLE_SANITIZER AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(Solvers PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(Solvers PUBLIC -fsanitize=address,undefined)
endif()
if(SOLVERS_ENABLE_COVERAGE)
  # coverage-tool's Clang path uses source-based coverage (llvm-profdata +
  # llvm-cov over .profraw), so instrument Clang builds with
  # -fprofile-instr-generate -fcoverage-mapping (NOT gcov's --coverage). GCC
  # keeps gcov/lcov. PUBLIC so SolversTests, which links Solvers, is
  # instrumented too and emits a .profraw when run.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
    target_compile_options(Solvers PUBLIC -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(Solvers PUBLIC -fprofile-instr-generate -fcoverage-mapping)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(Solvers PUBLIC --coverage -g -O0 -fprofile-arcs -ftest-coverage)
    target_link_options(Solvers PUBLIC --coverage)
  endif()
  # coverage-tool looks for the module's shared library under <build>/lib.
  set_target_properties(Solvers PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()
if(SOLVERS_ENABLE_CLANGTIDY)
  include(clang_tidy)
  xsigma_target_clang_tidy(Solvers OFF)
endif()
if(SOLVERS_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(Testing/Cxx)
endif()

cmake_minimum_required(VERSION 3.20)
project(Solvers VERSION 1.0.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cmake")
option(SOLVERS_ENABLE_TESTING "Build the GoogleTest solver suite" ${PROJECT_IS_TOP_LEVEL})
option(SOLVERS_ENABLE_CERES "Build the optional Ceres backend" OFF)
# Ipopt (interior-point NLP) and PETSc/TAO (which also realizes POUNDERS) are
# system libraries discovered through pkg-config rather than bundled submodules
# (docs/reviews/solver-redesign-2026-09-26.md). Native LM never depends on them;
# when OFF the adapters report backend_unavailable at runtime.
option(SOLVERS_ENABLE_IPOPT "Build the optional Ipopt backend" OFF)
option(SOLVERS_ENABLE_PETSC "Build the optional PETSc/TAO backend" OFF)
option(SOLVERS_ENABLE_SANITIZER "Enable address and undefined behavior sanitizers" OFF)
option(SOLVERS_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(SOLVERS_ENABLE_CLANGTIDY "Enable clang-tidy static analysis for Solvers" OFF)
option(SOLVERS_ENABLE_IWYU "Enable include-what-you-use analysis for Solvers" OFF)
option(SOLVERS_ENABLE_BENCHMARKS "Build the solver backend comparison benchmark" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(SolversDependencies)

file(GLOB_RECURSE solvers_sources CONFIGURE_DEPENDS "src/*.cpp")
add_library(Solvers ${solvers_sources})
add_library(Solvers::Solvers ALIAS Solvers)
target_include_directories(
    Solvers PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                   "$<INSTALL_INTERFACE:include>"
)
target_compile_features(Solvers PUBLIC cxx_std_17)
target_link_libraries(Solvers PUBLIC Eigen3::Eigen Logging::Logging)
target_compile_definitions(
    Solvers
    PUBLIC SOLVERS_HAS_CERES=$<BOOL:${SOLVERS_ENABLE_CERES}>
           SOLVERS_HAS_IPOPT=$<BOOL:${SOLVERS_ENABLE_IPOPT}>
           SOLVERS_HAS_PETSC=$<BOOL:${SOLVERS_ENABLE_PETSC}>
)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(Solvers PUBLIC SOLVERS_SHARED PRIVATE SOLVERS_BUILDING)
    set_target_properties(Solvers PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(SOLVERS_ENABLE_CERES)
    target_link_libraries(Solvers PRIVATE Ceres::ceres)

    # Create Solvers::CeresAutodiff interface target for clients using the Ceres AD integration
    add_library(Solvers_CeresAutodiff INTERFACE)
    target_link_libraries(Solvers_CeresAutodiff INTERFACE Solvers::Solvers Ceres::ceres)
    target_compile_definitions(Solvers_CeresAutodiff INTERFACE SOLVERS_HAS_CERES=1)
    add_library(Solvers::CeresAutodiff ALIAS Solvers_CeresAutodiff)
    set_target_properties(Solvers_CeresAutodiff PROPERTIES EXPORT_NAME CeresAutodiff)
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
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(Solvers PRIVATE PkgConfig::PETSC MPI::MPI_C)
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
    set_target_properties(
        Solvers PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                           ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
endif()
if(SOLVERS_ENABLE_CLANGTIDY)
    include(clang_tidy)
    xsigma_target_clang_tidy(Solvers OFF)
endif()
if(SOLVERS_ENABLE_IWYU)
    include(iwyu)
    xsigma_apply_iwyu(Solvers)
endif()
if(SOLVERS_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(Testing/Cxx)
endif()

if(SOLVERS_ENABLE_BENCHMARKS)
    add_subdirectory(Testing/Benchmark)
endif()

# ---------------------------------------------------------------------------
# Install & export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_solvers_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/Solvers")

# Eigen3 headers (bundled submodule, header-only). Eigen3 has no proper
# Eigen3Config.cmake installed by this package, so SolversConfig.cmake.in
# keeps a minimal hand-rolled Eigen3::Eigen fallback (see that file) --
# install(EXPORT) cannot manufacture a target for a dependency that was
# never part of this package's own export set.
install(DIRECTORY "${_solvers_tp}/eigen/Eigen" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${_solvers_tp}/eigen/unsupported" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# loguru: Logging links loguru PRIVATE and, by its own documented design
# (Logging/Cmake/LoggingConfig.cmake.in), deliberately never re-exports it --
# correct for a SHARED Logging, where loguru's object code is baked directly
# into libLogging.so. Solvers forces Logging to build STATIC when vendoring
# it (see SolversDependencies.cmake), so loguru's objects are never merged
# into libLogging.a; an external consumer linking the installed static
# Solvers::Solvers/Logging::Logging still needs loguru on the link line, and
# there is no LoguruConfig.cmake to find_dependency() against. Solvers
# re-installs/re-exposes the archive itself as a deliberate, still-necessary
# fallback (see SolversConfig.cmake.in) -- it is not redundant with
# install(EXPORT).
set(_logging_tp "${_solvers_tp}/Logging/ThirdParty")
install(TARGETS loguru ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES "${_logging_tp}/loguru/loguru.hpp" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/loguru")

# Logging itself, magic_enum's headers, and Logging's own CMake package
# files are no longer hand-copied here: Logging is vendored without
# EXCLUDE_FROM_ALL now (SolversDependencies.cmake), so its own
# install(TARGETS Logging EXPORT LoggingTargets ...) / install(DIRECTORY
# include/...) / configure_package_config_file(LoggingConfig...) rules run
# automatically as part of this package's `cmake --install`.

# Solvers library, and Solvers::CeresAutodiff when Ceres is enabled (its
# EXPORT_NAME is set to "CeresAutodiff" where the target is defined above, so
# NAMESPACE Solvers:: below produces the same Solvers::CeresAutodiff name
# used at build time). Both reference targets outside this export set
# (Logging::Logging, Ceres::ceres, ...); CMake permits that in the generated
# *Targets.cmake as long as SolversConfig.cmake.in find_dependency()s /
# pkg_check_modules()s each one into existence before including it -- which
# it does.
install(
    TARGETS Solvers
    EXPORT SolversTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
if(TARGET Solvers_CeresAutodiff)
    install(TARGETS Solvers_CeresAutodiff EXPORT SolversTargets)
endif()
install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

# CMake package files
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SolversConfigVersion.cmake" VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/SolversConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SolversConfig.cmake"
    INSTALL_DESTINATION "${_solvers_install_cmakedir}"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SolversConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/SolversConfigVersion.cmake"
        DESTINATION "${_solvers_install_cmakedir}"
)

install(
    EXPORT SolversTargets
    NAMESPACE Solvers::
    FILE SolversTargets.cmake
    DESTINATION "${_solvers_install_cmakedir}"
)

include(GoogleTest)

# Ceres AD integration tests (only when Ceres is available)
set(CERES_TEST_FILES)
if(SOLVERS_ENABLE_CERES)
    list(APPEND CERES_TEST_FILES TestCeresAutoDiff.cpp)
endif()

add_executable(
    SolversTests TestOptimizationAlgorithm.cpp TestOptimizerWrapper.cpp TestSolverBackends.cpp
                 TestRootFindingAndPolynomialSolvers.cpp TestSolverApiDispatch.cpp
                 TestVerboseAndCornerCases.cpp TestDerivativeProviders.cpp TestLevenbergMarquardtLiterature.cpp
                 TestRncLm.cpp TestEvaluators.cpp TestResults.cpp TestProblemApi.cpp TestLmLinearSolvers.cpp TestRoots.cpp TestApiPaths.cpp ${CERES_TEST_FILES}
)
target_link_libraries(SolversTests PRIVATE Solvers::Solvers GTest::gtest_main)

# Link Ceres for tests when available
if(SOLVERS_ENABLE_CERES AND TARGET Ceres::ceres)
    target_link_libraries(SolversTests PRIVATE Ceres::ceres)
    target_compile_definitions(SolversTests PRIVATE SOLVERS_HAS_CERES=1)
endif()
# PRE_TEST defers gtest enumeration to ctest time instead of running the (here
# coverage-instrumented) test binary as a post-link build step. Under a fully
# parallel build the post-build enumeration competes with in-flight compiles
# and blows the default 5s DISCOVERY_TIMEOUT; PRE_TEST avoids that entirely and
# the longer timeout keeps discovery robust on a loaded machine.
# LABELS "Solvers" lets coverage-tool match this suite to the Solvers module
# by CTest label (its most robust discovery path), independent of exe naming.
gtest_discover_tests(SolversTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 60
    PROPERTIES LABELS "Solvers")

# Allocation check for the native LM iteration loop. Eigen's runtime malloc guard
# (EIGEN_RUNTIME_NO_MALLOC) is a per-program setting, so this executable compiles
# the kernel sources itself instead of linking the Solvers library.
add_executable(
    LmNoAllocCheck
    NoAllocCheckLm.cpp
    ${PROJECT_SOURCE_DIR}/src/solvers/levenberg_marquardt_solver.cpp
    ${PROJECT_SOURCE_DIR}/src/solver_options/solver_options.cpp
    ${PROJECT_SOURCE_DIR}/src/solver_options/solver_options_lm.cpp
    ${PROJECT_SOURCE_DIR}/src/api/evaluators.cpp
    ${PROJECT_SOURCE_DIR}/src/api/derivative_provider.cpp
)
target_include_directories(LmNoAllocCheck PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(LmNoAllocCheck PRIVATE cxx_std_17)
target_compile_definitions(
    LmNoAllocCheck PRIVATE SOLVERS_HAS_CERES=0 SOLVERS_HAS_IPOPT=0 SOLVERS_HAS_PETSC=0
)
# Enable the runtime malloc guard and live Eigen assertions in every TU.
target_compile_options(
    LmNoAllocCheck PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/eigen_no_alloc_config.h
)
target_link_libraries(LmNoAllocCheck PRIVATE Eigen3::Eigen Logging::Logging GTest::gtest_main)
gtest_discover_tests(LmNoAllocCheck DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60
                     PROPERTIES LABELS "Solvers")

option(SOLVERS_ENABLE_BENCHMARKS "Build solver benchmark executables" OFF)
if(SOLVERS_ENABLE_BENCHMARKS)
    add_executable(LmLinearSolversBenchmark BenchmarkLmLinearSolvers.cpp)
    target_link_libraries(LmLinearSolversBenchmark PRIVATE Solvers::Solvers)

    add_executable(RncLmBenchmark BenchmarkRncLm.cpp)
    target_link_libraries(RncLmBenchmark PRIVATE Solvers::Solvers)

    add_executable(SolversBenchmark BenchmarkSolvers.cpp)
    target_link_libraries(SolversBenchmark PRIVATE Solvers::Solvers)

    add_executable(RootFindersBenchmark BenchmarkRootFinders.cpp)
    target_link_libraries(RootFindersBenchmark PRIVATE Solvers::Solvers)

    add_executable(RootFindersVsLMBenchmark BenchmarkRootFindersVsLM.cpp)
    target_link_libraries(RootFindersVsLMBenchmark PRIVATE Solvers::Solvers)

    add_executable(SviCalibrationBenchmark BenchmarkSviCalibration.cpp)
    target_link_libraries(SviCalibrationBenchmark PRIVATE Solvers::Solvers)

    # Ceres AD benchmark (only when Ceres is available)
    if(SOLVERS_ENABLE_CERES AND TARGET Ceres::ceres)
        add_executable(BenchmarkCeresAutoDiff BenchmarkCeresAutoDiff.cpp)
        target_link_libraries(BenchmarkCeresAutoDiff PRIVATE Solvers::Solvers Ceres::ceres)
    endif()
endif()

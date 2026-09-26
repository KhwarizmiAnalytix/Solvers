include(GoogleTest)
add_executable(
    SolversTests TestOptimizationAlgorithm.cpp TestOptimizerWrapper.cpp TestSolverBackends.cpp
                 TestRootFindingAndPolynomialSolvers.cpp TestSolverApiDispatch.cpp
)
target_link_libraries(SolversTests PRIVATE Solvers::Solvers GTest::gtest_main)
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

option(SOLVERS_ENABLE_BENCHMARKS "Build the Ceres/NLopt/native solver benchmark executable" OFF)
if(SOLVERS_ENABLE_BENCHMARKS)
    add_executable(SolversBenchmark BenchmarkSolvers.cpp)
    target_link_libraries(SolversBenchmark PRIVATE Solvers::Solvers)

    add_executable(RootFindersBenchmark BenchmarkRootFinders.cpp)
    target_link_libraries(RootFindersBenchmark PRIVATE Solvers::Solvers)

    add_executable(RootFindersVsLMBenchmark BenchmarkRootFindersVsLM.cpp)
    target_link_libraries(RootFindersVsLMBenchmark PRIVATE Solvers::Solvers)
endif()

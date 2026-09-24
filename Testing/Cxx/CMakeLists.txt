include(GoogleTest)
add_executable(
    SolversTests TestOptimizationAlgorithm.cpp TestOptimizerWrapper.cpp TestSolverBackends.cpp
                 TestRootFindingAndPolynomialSolvers.cpp
)
target_link_libraries(SolversTests PRIVATE Solvers::Solvers GTest::gtest_main)
gtest_discover_tests(SolversTests)

option(SOLVERS_ENABLE_BENCHMARKS "Build the Ceres/NLopt/native solver benchmark executable" OFF)
if(SOLVERS_ENABLE_BENCHMARKS)
    add_executable(SolversBenchmark BenchmarkSolvers.cpp)
    target_link_libraries(SolversBenchmark PRIVATE Solvers::Solvers)

    add_executable(RootFindersBenchmark BenchmarkRootFinders.cpp)
    target_link_libraries(RootFindersBenchmark PRIVATE Solvers::Solvers)

    add_executable(RootFindersVsLMBenchmark BenchmarkRootFindersVsLM.cpp)
    target_link_libraries(RootFindersVsLMBenchmark PRIVATE Solvers::Solvers)
endif()

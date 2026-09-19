include(GoogleTest)
add_executable(SolversTests TestOptimizationAlgorithm.cpp TestOptimizerWrapper.cpp)
target_link_libraries(SolversTests PRIVATE Solvers::Solvers GTest::gtest_main)
gtest_discover_tests(SolversTests)

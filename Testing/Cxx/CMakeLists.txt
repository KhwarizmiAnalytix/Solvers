include(GoogleTest)

add_executable(TestSolversBackend TestSolversBackend.cpp)
target_include_directories(TestSolversBackend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(TestSolversBackend PRIVATE Solvers GTest::gtest_main)
gtest_discover_tests(TestSolversBackend PROPERTIES LABELS Solvers)

add_executable(TestRootFinding TestRootFinding.cpp)
target_include_directories(TestRootFinding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(TestRootFinding PRIVATE Solvers GTest::gtest_main)
gtest_discover_tests(TestRootFinding PROPERTIES LABELS Solvers)

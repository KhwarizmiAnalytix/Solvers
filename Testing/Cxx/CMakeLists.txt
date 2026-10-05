include(GoogleTest)

add_executable(TestSolversBackend TestSolversBackend.cpp)
target_include_directories(TestSolversBackend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(TestSolversBackend PRIVATE Solvers GTest::gtest_main)
gtest_discover_tests(TestSolversBackend)

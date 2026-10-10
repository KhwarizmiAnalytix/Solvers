include(GoogleTest)

# gtest discovery runs the executable at build time. Windows searches for a DLL
# beside that executable, and Solvers.dll is produced in the build root.
function(solvers_add_test target)
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${target} PRIVATE Solvers GTest::gtest_main)
  if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Solvers>
              $<TARGET_FILE_DIR:${target}>)
  endif()
  gtest_discover_tests(${target} PROPERTIES LABELS Solvers)
endfunction()

solvers_add_test(TestSolversBackend)
solvers_add_test(TestRootFinding)
solvers_add_test(TestRncLmSolver)
solvers_add_test(TestLevenbergMarquardt)

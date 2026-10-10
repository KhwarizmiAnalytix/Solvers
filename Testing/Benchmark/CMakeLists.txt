# Use system-installed Google Benchmark via pkg-config
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(BENCHMARK benchmark QUIET IMPORTED_TARGET)
endif()

if(NOT TARGET PkgConfig::BENCHMARK)
  message(STATUS "Google Benchmark not found. Trying find_package...")
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    message(FATAL_ERROR "Google Benchmark not found. Install it via: brew install benchmark")
  endif()
endif()

function(solvers_add_benchmark target)
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  if(TARGET PkgConfig::BENCHMARK)
    target_link_libraries(${target} PRIVATE Solvers PkgConfig::BENCHMARK)
  else()
    target_link_libraries(${target} PRIVATE Solvers benchmark::benchmark)
  endif()
  if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Solvers>
              $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction()

solvers_add_benchmark(BenchmarkNative)
solvers_add_benchmark(BenchmarkExternal)
solvers_add_benchmark(BenchmarkRootFinding)

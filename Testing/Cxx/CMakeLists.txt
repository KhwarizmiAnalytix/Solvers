file(
  GLOB_RECURSE test_sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/Test*.cxx")

set(TestFiles)
foreach(_test_sources IN ITEMS ${test_sources})
    get_filename_component(file_name "${_test_sources}" NAME)
    list(APPEND TestFiles "${file_name}")
endforeach()

if(QUARISMA_GOOGLE_TEST)
    include(GoogleTest)
    quarisma_module_gtest_executable(MathCxxTests ${TestFiles})
    quarisma_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")
else()
    quarisma_add_test_cxx(MathCxxTests tests 
    NO_DATA NO_VALID NO_OUTPUT
    ${TestFiles})
    quarisma_test_cxx_executable(MathCxxTests tests)
endif()

target_include_directories(MathCxxTests BEFORE
                           PRIVATE "${CPUINFO_SOURCE_DIR}/include" "${EIGEN_SOURCE_DIR}")
#target_link_libraries(MathCxxTests PRIVATE cpuinfo)



set(MATH_INCLUDE_LIBS "${QUARISMA_DEPENDENCY_LIBS}")

# -----------------------------------------------------------------------------
# An optional dependency on CUDA
if(QUARISMA_ENABLE_CUDA)
  list(APPEND MATH_INCLUDE_DIRS "${CUDA_INCLUDE_DIRS}")
  list(APPEND MATH_INCLUDE_LIBS "${CUDA_CUDA_LIBRARY}")

  file(
    GLOB_RECURSE CUDA_SOURCES
    LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")

  set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE "CUDA")

  set(GLOB_RECURSE CUDA_HEADERS LIST_DIRECTORIES false
                   "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.h")
endif()

#message("QUARISMA_SMP_IMPLEMENTATION_TYPE: ${QUARISMA_SMP_IMPLEMENTATION_TYPE}")

if(QUARISMA_ENABLE_MKL)
  list(APPEND MATH_INCLUDE_DIRS "${QUARISMA_MKL_INCLUDE_DIRS}")
  list(APPEND MATH_INCLUDE_LIBS "${QUARISMA_MKL_INCLUDE_LIBS}")
endif()
# -----------------------------------------------------------------------------
file(
  GLOB_RECURSE sources
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx")

file(
  GLOB_RECURSE headers
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.inl")

file(
  GLOB_RECURSE templates
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx")

list(FILTER sources EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/Testing")
list(FILTER headers EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/Testing")

if(NOT QUARISMA_ENABLE_CUDA)
  list(FILTER sources EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/cuda/")
  list(FILTER headers EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/cuda/")
endif()

file(
  GLOB_RECURSE wrap_headers
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/distribution/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/copula/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/laplace_inverter/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/interpolator/interpolator_enum.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/quadrature/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/optimization_algorithm/*.h")

quarisma_module_add_module(
  QUARISMA::Math
  SOURCES
  ${sources}
  ${CUDA_SOURCES}
  HEADERS
  ${headers}
  ${CUDA_HEADERS}
  TEMPLATES
  ${templates}
  WRAP_HEADERS
  ${wrap_headers})

if(MATH_INCLUDE_DIRS)
  quarisma_module_include(QUARISMA::Math PUBLIC ${MATH_INCLUDE_DIRS})
endif()
#target_link_libraries(Math PRIVATE quarismaeigen)

quarisma_module_link(QUARISMA::Math PUBLIC ${MATH_INCLUDE_LIBS})

quarisma_module_compile_features(QUARISMA::Math PUBLIC cxx_std_${QUARISMA_CXX_STANDARD})
quarisma_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(Math PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_ARCHITECTURES OFF
                      # CUDA_SEPARABLE_COMPILATION ON
                      )

if (MSVC)
    set_target_properties(Math
                            PROPERTIES
                            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
set_target_properties(Math PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE}")
if(CLANG_TIDY_FOUND)
  quarisma_module_enable_clang_tidy(Math)
endif()


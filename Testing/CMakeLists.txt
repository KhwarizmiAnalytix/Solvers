add_subdirectory(Cxx)

if(QUARISMA_WRAP_PYTHON AND NOT QUARISMA_BUILD_WHEEL)
  add_subdirectory(Python)
endif()

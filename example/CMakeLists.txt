cmake_minimum_required(VERSION 3.20)
project(SolversExample LANGUAGES CXX)

find_package(Solvers REQUIRED)

add_executable(root_finding root_finding.cpp)
target_link_libraries(root_finding PRIVATE Solvers::Solvers)

add_executable(least_squares least_squares.cpp)
target_link_libraries(least_squares PRIVATE Solvers::Solvers)

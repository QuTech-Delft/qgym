# This file is copied from OpenQL (Apache 2.0 License, Copyright [2016] [Nader Khammassi & Imran Ashraf, QuTech, TU Delft]): https://github.com/QuTech-Delft/OpenQL/blob/develop/LICENSE
# For the original file see: https://github.com/QuTech-Delft/OpenQL/blob/develop/python/compat/test-cmake-config/CMakeLists.txt
# No changes were made.

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(CheckBitness C CXX)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/values.cfg.template"
    "${CMAKE_CURRENT_BINARY_DIR}/values.cfg"
)

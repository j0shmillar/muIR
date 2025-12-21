cmake_minimum_required(VERSION 3.15)
project(unpu_runtime_ai8x CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(unpu_runtime
    unpu_runtime.cpp
)

target_include_directories(unpu_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(unpu_ai8x_runner
    main_ai8x.cpp
)

target_link_libraries(unpu_ai8x_runner
    unpu_runtime
)

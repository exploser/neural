cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(neural main.cpp)

if(UNIX)
	target_link_libraries(neural m)
elseif(WIN32)
endif()

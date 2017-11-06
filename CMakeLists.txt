cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(neural STATIC NeuralNetwork.cpp NeuralNetworkLow.cpp)
add_executable(neural-run main.cpp)

if(UNIX)
	target_link_libraries(neural-run m neural)
	install (TARGETS neural DESTINATION bin)
	install (FILES NeuralNetwork.hpp DESTINATION include)
elseif(WIN32)
endif()


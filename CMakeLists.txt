PROJECT(Helloworld)

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Werror -Wall -pedantic -ansi)
 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB SRC_FILES src/*.c src/*.cpp)

file(GLOB HEADER_FILES src/*.hpp src/*.cpp)

foreach(SOURCE_PATH ${SRC_FILES})

    get_filename_component(EXECUTABLE_NAME ${SOURCE_PATH} NAME_WE)

    add_executable(${EXECUTABLE_NAME} ${SOURCE_PATH} ${HEADER_FILES})
    target_link_libraries( ${EXECUTABLE_NAME} ${OpenCV_LIBS} )
endforeach()
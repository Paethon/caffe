# Require CMake 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

# SET(CMAKE_BUILD_TYPE RelWithDebInfo)
SET(CMAKE_BUILD_TYPE Release)

# -fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined -fsanitize=thread
# -funsafe-math-optimizations -ffast-math -funsafe-loop-optimizations -ftree-loop-if-convert-stores -ftree-vectorize -march=native
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wextra -std=c++1y -march=native -fno-omit-frame-pointer -O3")

# Set project name
PROJECT(patch_extract)

# Add a target to generate API documentation with Doxygen
FIND_PACKAGE(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

# Google Test library
# ADD_SUBDIRECTORY(googletest)
# INCLUDE_DIRECTORIES(${gtest_SOURCE_DIR}/include)
# INCLUDE_DIRECTORIES(${gtest_SOURCE_DIR})

# Boost
FIND_PACKAGE( Boost COMPONENTS system filesystem program_options REQUIRED )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

# OpenCV
FIND_PACKAGE(OpenCV 3.0 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})

# Main 
ADD_EXECUTABLE(patch_extract patch_extract.cpp helper.cpp)
TARGET_LINK_LIBRARIES(patch_extract ${OpenCV_LIBS} ${Boost_LIBRARIES})

# V1 test
# ADD_EXECUTABLE(v1test tests/V1.cpp V1.cpp param.cpp blob.cpp)
# TARGET_LINK_LIBRARIES(v1test gtest ${OpenCV_LIBS})

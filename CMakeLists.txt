cmake_minimum_required(VERSION 3.5.1)
project(mujoco_gym)



set(CMAKE_CXX_STANDARD 14)
# It prevents the decay to C++98 when the compiler does not support C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# It disables the use of compiler-specific extensions
# e.g. -std=c++14 rather than -std=gnu++14
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# I believe must be path to Libtorch 
set(Torch_DIR libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

set(NumCpp_DIR NumCpp/cmake)
find_package(NumCpp PATHS ${NumCpp_DIR} NO_DEFAULT REQUIRED)
if (NumCpp_FOUND)
    message(STATUS "NumCpp library found!")
    message(STATUS "    include path: ${NUMCPP_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate NumCpp" \n)
endif()

# find_package(fmt)
# target_link_libraries(fmt/build fmt::fmt)

find_package(fmt REQUIRED)
link_libraries(fmt::fmt)

# Mujoco, glfw3 and GSL are all installed and found on your system 
find_package(mujoco REQUIRED)
link_libraries(mujoco::mujoco)

find_package(glfw3 REQUIRED)
# link_libraries(GLFW::glfw3)

find_package(GSL REQUIRED)
link_libraries(GSL::gsl)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB SOURCE_FILES mul_Tiny_bat_human.cpp)

#CMAKE_PROJECT_NAME is from 'project(mujoco_gym)' on the second line of this script
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries (
        ${CMAKE_PROJECT_NAME}
        ${TORCH_LIBRARIES}
        ${NUMCPP_LIBRARIES}
        ${GSL_LIBRARIES}
        ${mujoco_LIBRARIES}
        # ${fmt_LIBRARIES}
        fmt
        glfw
        
)
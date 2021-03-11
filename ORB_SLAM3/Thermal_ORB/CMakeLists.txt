cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED PATHS /usr/local/lib/python3.7/site-packages/torch/share/cmake/Torch)
if(NOT Torch_FOUND)
   message(FATAL_ERROR "Torch cmake info not found. Edit line 37 of CMakeLists.txt with your path.")
endif()
find_package(OpenCV 4)


add_executable(example-app imageProcess.cc)
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

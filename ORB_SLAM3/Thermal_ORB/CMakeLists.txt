cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED PATHS /home/msaraceno/.local/lib/python3.6/site-packages/torch/share/cmake/Torch)
find_package(OpenCV)


add_executable(example-app imageProcess.cc)
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

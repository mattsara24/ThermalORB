# ThermalORB
## Table of Contents
1. [General Info](#general-info)
2. [Installation](#installation)
3. [Running Instructions](#RunInstructions)
### General Info
***
C++ Wrapper of https://github.com/eric-yyjau/pytorch-superpoint using PyTorch and OpenCV. 
The code is flexible to different SuperPoint weights because it take a Pytorch model file (.pt) as input.

## Installation
***
### Requires
- C++14
- PyTorch >= 1.1 (tested in 1.7.1)
- OpenCV >= 4 (will be extended to 3 soon)
- [OPTIONAL] CUDA >= 10 

```
$ git clone https://github.com/mattsara24/ThermalORB
$ cd ../path/to/the/file
$ mkdir build && cd build
$ cmake ..
$ make
```

### NOTE
In order to properly compile the code you must edit the CMakeLists.txt file to point it to your installation of libtorch.
> The easiest way to do that is to determine where your python package manager installed it.
> 
> Pip ``` pip show torch```
> Once you have that root directory for torch append onto it ``` share/cmake/Torch/ ```. This should be enough to work.

## Running Instructions
***
From inside the build directory you can run the package with
``` ./superPoint ../pretained/combinedSuperPoint.pt ```
##### NOTE
On the initial run you might have to edit line 220 of imageProcess.cc and set your local image path.


#ThermalORB
## Table of Contents
1. [General Info](#general-info)
2. [Requires] (#requires)
3. [Installation](#installation)
4. [Running Instructions] (#RunInstructions)
### General Info
***
C++ Wrapper of https://github.com/eric-yyjau/pytorch-superpoint using Pytorch and OpenCV. 
The code is flexible to different SuperPoint weights because it take a Pytorch model file (.pt) as input.

## Requires
***


## Installation
***
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
> Pip
> ```
> pip show torch
> ```
> Once you have that root directory for torch append onto it ``` share/cmake/Torch/ ```. This should be enough to work.
## Running Instructions
***
Give instructions on how to collaborate with your project.
> Maybe you want to write a quote in this part. 
> Should it encompass several lines?
> This is how you do it.

# MLR robotics course & practical robotics course

This repo is based on RAI code, including its python bindings. See https://github.com/MarcToussaint/rai for a README of the RAI code.

## Table of Contents
- [MLR robotics course & practical robotics course](#mlr-robotics-course--practical-robotics-course)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Documentation](#documentation)
    - [Setup for Robotics Lab Course in Simulation](#setup-for-robotics-lab-course-in-simulation)
- [Further Documentation & Installation Pointers](#further-documentation--installation-pointers)
  - [Installation](#installation)
  - [rai code](#rai-code)
  - [rai examples](#rai-examples)
  - [Tutorials](#tutorials)


## Quick Start

The repo is now used for three lecture formats: the robotics lab
course in simulation, the robotics lab course in real, and the
robotics lectures. Please follow the respective sections.

### Documentation

* [Course material and some documentation of the code base and python bindings](https://marctoussaint.github.io/robotics-course/)


### Setup for Robotics Lab Course in Simulation

This assumes a standard Ubuntu 18.04 or 20.04 machine.

* The following assumes $HOME/git as your git path, and $HOME/opt
to install 3rd-party libs -- please stick to this (no system-wide installs)

* If you'll use python:
```
sudo apt-get install python3 python3-dev python3-numpy python3-pip python3-distutils
echo 'export PATH="${PATH}:$HOME/.local/bin"' >> ~/.bashrc   #add this to your .bashrc, if not done already
pip3 install --user jupyter nbconvert matplotlib pybind11 opencv-python
```

* Clone and compile our robotics-course code:
```
mkdir -p $HOME/git
cd $HOME/git
git clone --recursive https://github.com/MarcToussaint/robotics-course.git

cd robotics-course
make -j4 installUbuntuAll  # calls sudo apt-get install; you can always interrup
```
# Further Documentation & Installation Pointers

## Installation

* [ROS kinectic](http://wiki.ros.org/kinetic/Installation/Ubuntu) (for Ubuntu 16.04) or [ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) (for Ubuntu 18.04)
* [OpenCV](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/localSourceInstalls.md#OpenCV) (from source)
* [PhysX](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/localSourceInstalls.md#PhysX) (from source)
* [Bullet](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/localSourceInstalls.md#Bullet) (from source)
* [qtcreator](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/qtcreator.md) (from source or ubuntu, setting up projects, pretty printers, source parsing)
* Python3:
```
sudo apt-get install python3 python3-dev python3-numpy python3-pip python3-distutils
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
```

## rai code

* [rai::Array and arr](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/arr.md) (tensors, arrays, std::vector)
* [Features and Objectives for KOMO](https://github.com/MarcToussaint/rai-python/blob/master/docs/2-features.ipynb)
* [Graph and `.g` files](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/graph.md) (Python dictionaries, any-type container, file format, logic)
* [Editing robot model/configuration files](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/kinEdit.md)  (URDF, transformations, frame tree)
* [docker](https://github.com/MarcToussaint/rai-maintenance/tree/master/docker) (testing rai within docker, also Ubuntu 18.04)

## rai examples

* [Python examples](https://github.com/MarcToussaint/rai-python/tree/master/docs)
* [Python robotics exercises](https://github.com/MarcToussaint/robotics-course/tree/master/py)
* [Cpp robotics exercises](https://github.com/MarcToussaint/robotics-course/tree/master/cpp)

## Tutorials

1. [Basics:](tutorials/1-basics.ipynb) Configurations, Features & Jacobians
1. [Features:](tutorials/2-features.ipynb) Learn about the language to define and query features and their Jacobians. Including querying collision features (whether and which objects are in collision).

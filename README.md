# Deep-RL for robot object manipulation

This repo contains an implementation of the Deep Reinforcement strategy for learning object manipulation. It has been developed for a Final Project of AI & Robotics: Lab Course at TU Berlin. It uses optimization framework for handling robot dynamics - [K-order Markov Optimization within RAI](https://github.com/MarcToussaint/rai) and learning methods for task execution and planning. 


https://user-images.githubusercontent.com/30003101/221697549-c54a114f-2180-4e89-bfb1-4c8d6802543a.mp4

## Table of Contents
  - [Quick Start](#quick-start)
    - [Documentation: Robot dynamics](#documentation-robot-dynamics)
    - [Setup for Robotics Lab Course in Simulation](#setup-for-robotics-lab-course-in-simulation)
- [Further Documentation & Installation Pointers](#further-documentation--installation-pointers)
  - [Installation](#installation)
  - [rai code](#rai-code)
  - [rai examples](#rai-examples)
  - [Tutorials](#tutorials)


## Quick Start

[scripts/learning_setup.py](https://github.com/VasilyRakche/robotics-course-vr/blob/master/scripts/learning_setup.py) is used to train the network.
Configuration params within the file:
- box_name 
    - "boxc": cube
    - "boxcl": cilynder
    - "boxr": rectangle 
- EVALUATION # for runnig the network in evaluation mode
- WARM_START # for starting the network training from saved model
- Worker object can be initialized with many params (among others):
  - sim_verbose_freq_episodes # define the simulation verbose 


[learning_setup_exec_panda.py](https://github.com/VasilyRakche/robotics-course-vr/blob/master/scripts/learning_setup_exec_panda.py) is used to execute the trained network together with KOMO for PANDA manipulation.
Configuration params within the file:
- box_name 
    - "boxc": cube
    - "boxcl": cilynder
    - "boxr": rectangle 
- EXEC_COMPARISON # run evaluation for all different box shapes (10 times 20 runs) 
- Worker object can be initialized with many params (among others):
  - sim_verbose_freq_episodes # define the simulation verbose 


### Documentation: Robot dynamics

* [RAI Course material and some documentation of the code base and python bindings](https://marctoussaint.github.io/robotics-course/)

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

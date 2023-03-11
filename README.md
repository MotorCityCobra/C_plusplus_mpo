# C++ MPO

The [Maximum A Psoteriori Policy Optimization](https://arxiv.org/pdf/1806.06920v1.pdf) algorithm written in C++ with LibTorch.  
Mostly translated from the implentation in Python by [acyclics](https://github.com/acyclics/MPO).  

To install LibTorch git clone the PyTorch repo, find the LibTorch directory and build from source.

Other dependancies are  
numcpp  
gsl  
mujoco  
fmt  
glfw  


I wrote this in C++ because MuJoCo (the simulator) is in C++ and robotic componant hardware drivers are in C++. Why slow down in Python when we can go 100% C++?  
I found any mujoco models with complexity with many DOF, like the humanoid impossible to train. Maybe it's because I just have one 3090. Maybe it's because there is a problem with the code. I welcome any feedback in the 'issues'.  
It's a very interesting very deep dive into Deep Q MDP to work with this algorithm. I expected more payoff for all the moving parts in the algorithm.

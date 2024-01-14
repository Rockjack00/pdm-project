# Sample Space Shrinking RRT*
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> _#TODO_: come up with a good name

Final Project
TU Delft course RO47005 Planning and Decision Making

##### Contributors: 
> Danish Ansari \
  Jasper van Brakel \
  Tyler Olson \
  Gijs Zijderveld

##### Requirements:
* python >= 3.10
* urdfenvs
* [gym_envs_urdf](https://github.com/maxspahn/gym_envs_urdf)

##### Last Updated:
December 14th, 2023


### Description
This repository contains source code and demos accompanying our [final project report](docs/report.pdf).
In this repository we implement an adaptation of the RRT* algorithm a modified collision checker as a global planner while using a holonomic high-five robot. The robot is being simulated in the `gym_envs_urdf` environment.


### Quick start
First, complete the [Setup Instructions](#setup-instructions).

The project installs to executables in to the venv, `rrt-star-bench-single-run` and `rrt-star-bench-multi-run` (multiple checkpints same sampler).
These executables contain a help accesable via `-h`.
The Nullspace sampler can be enabled using `-NS`.

As an artifact of the sample space updating, it may slow down after around 10 iterations when there is a tradeoff being made between frequently updating the sample space and generating good sample candidates.  After some time however (after around 10% of null space has been covered), it will again speed up as the sample space converges and has already found most of the colliding regions.

It is important to validate a test before doing multi checkpoint runs. This can be done by checking the map using `-vw` and first running the RRT* with the simple sampler (default) to see if it can find a path. (The current world generation allows for some start/goal points on rare occacions, which collide, this causes path finding to fail.)

An example for both Simple and Nullspace sampler.
```bash
# Runs a single checkpoint RRT star test with the Simple Sampler, with all visualizations
rrt-star-bench-single-run -s 41 -i 500 -vw -vp -vs
# Runs a single checkpoint RRT star test with the Nullspace Sampler, with path and simulation visualizations
rrt-star-bench-single-run -s 41 -i 500 -NS -vp -vs
```


## Setup Instructions
Note: These setup instructions have been tested on Ubuntu 20.04 LTS and 22.04 LTS.
On other platforms your mileage may vary.

> An install/setup script is provided, which performs the same steps. Use at your own risks!

We recommend installing this project in a python virtual environment using the 
standard `venv` python package.
1. Create the virtual environment and activate it.
    ```bash
    python -m venv <virtual-environment>
    ./<virtual-environment>/bin/activate
    ```
1. Make sure `pip` is up to date.
    ```bash
    pip install --upgrade pip
    ```

Install the necessary packages with:
```bash
pip install -r requirements.txt
```

If you are using Ubuntu (22.04 LTS, Default Python):
```bash
sudo apt install python3-tk
sudo apt install python3-pil python3-pil.imagetk
```

For development the package can be installed as editable:
```bash
pip install -e .
```

If using TKinter in virtual env (Ubuntu with Deadsnakesppa):
```bash
sudo apt install python3.10-tk
```

# Sample Space Shrinking RRT* 
![Example CI Badge](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

_#TODO_: come up with a good name

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


### Quick start
_#TODO_: quickstart instructions
```bash
python launch_demo.sh
```

Otherwise see the [Setup Instructions](#setup-instructions).


### Description
This repo contains source code and demos accompanying our [final project report](docs/report.pdf).
In this repo we implement an adaptation of the RRT* algorithm a modified
collsision checker as a global planner with using a holonomic high-five robot
in a dynamic(?) environment. The robot is being simulated in the `gym_envs_urdf`
environment.


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

#### NOTES: TO ADD
***TODO***: Integrate and specify this
<!-- If using qt backend install (Ubuntu):
```bash
sudo apt install libxcd-cursor0
``` -->

If using TKinter in virtual env (Ubuntu with Deadsnakesppa):
```bash
sudo apt install python3.10-tk
``` 

## Running the demo
_#TODO_: demo instructions
```bash
python launch_demo.sh
```


# Introduction

RL-Studio allows training, retraining and inference of already created models.
We have called each of these options **modes of operation**, or **modes** for short.

- In the **training mode** we define a specific task to achieve, a specific agent, the algorithm necessary to learn and the simulator where the training will be executed. The final result is a model generated and saved to be able to be used.
- In **retraining mode**, the objective is to use an already generated model to retrain on it and generate new models. It is convenient if you do not want to start a training from scratch.
- In **inference mode**, a previously generated model is loaded and executed in other environments for the same task and the same agent. In this way we check the generalization, robustness and goodness of the models trained with the algorithm used.

## config file

The parameterization of RL-Studio is done with a yaml configuration file. There is a general **config.yaml** file with inside comments and with the following structure that it is necessary to understand well in order to work correctly:

- settings: general parameters as mode, task, algorithm, agent...
- ros: general parameters for ROS
- carla: Carla environment launch params
- inference and retraining models: file to be called in retrained or inference mode
- algorithm parameters: general params for algorithms implemented such as PPO, DQN...
- agents: specific params for every agent, such as sensor configuration
- states: params to define the input state such as image, laser, low-data adquisition, Lidar. There are differents from the agents
- actions: actions definitions for the agent to take. In case of AV could be linear and angular velocity, throttle, brake.
- rewards: reward function
- environment parameters: each environment has its own parameters, which are defined in this place.

It is possible to add new options but you should avoid modifying the existing ones.

Due to the length of the config file, to work with RL-Studio it is convenient to create config files for each task that needs to be done, in order to leave the configuration ready to launch the application in the fastest and most comfortable way. The files must have the same format as the general config.yaml with the file name of the form:

```
config_mode_task_algorithm_agent_simulator.yaml
```

The file must be saved in the directory

```
/PATH/TO/RL-Studio/rl_studio/config/
```

There are several config files to take as example. If you need more information about coding style, please refer to [coding](./CODING.md) file.

## Project diagram

The following graph shows a conceptual diagram of the operation of RL-Studio in training mode. In the case of making inference or retraining, the process is similar

![](../rl_studio/docs/rls-diagram.svg)


# Run RL Studio

## Usage

Open the `config.yaml` file and set the params you need. Then to run RL-Studio, go to directory

```bash
cd ~/PATH/TO/RL-Studio/rl_studio
```

and then just type (depending on how the dependencies are managed):


Pip:

```
python rl-studio.py -f config/<config.yaml>
```
where <config.yaml> can be any config you can create or previosly existed.

When launching Gazebo, if you found and error, try killing all possible previous ROS-Gazebo processes:

```
killall gzserver
killall gzclient
```

## Config.yaml
The config.yaml contains all project hyperparams and configuration needed to execute correctly. In case you want to train a Formula 1 agent in a Follow Lane task in Gazebo, with a PPO algorithm and Tensorflow Deep Learning framework, you can use next example from a config.yaml example file:

```yaml
settings:
  algorithm: PPO
  task: follow_lane
  environment: simple
  mode: training # or inference
  agent: f1
  simulator: gazebo
  framework: tensorflow
```

Remaining params should be adjusted too. There are many working yaml files in config folder to check them.  

> :warning: If you want to use inferencing in a program language other than python, you will
> need extend the rl-studio.py to listen for inputs in a port and execute the loaded brain/algorithm to provide
> outputs in the desired way.

More info about how to config and launch any task, please go to [agents](agents/README.md) section.


## Carla

Two scripts have been developed to get the waypoints of each city in the simulator. In `/envs/carla/utils/`

- `plot_waypoints.py` generates a plot with all the waypoints in the city. By zooming in you can get the road_id, lane_id and waypoint_id of every of them.
Launch it with `python plot_waypoints.py -t <town>` where `<town>` could be Town01, you get next image

![](../rl_studio/docs/Town01_waypoints.png)

- `plot_topology.py` generates the connected waypoints to build routes. Launch it with `python plot_topology.py -t <town>` and generates next image 
  
![](../rl_studio/docs/Town01_topology.png)

- `get_spectator_location.py` print the location where the current spectator is located in carla with `python get_spectator_location.py` 


*Note:* If you encounter the problem in which tour simulator crashes after some training time with:

```
Failed to find symbol file, expected location:
"/opt/carla-simulator/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping.sym"
Malloc Size=131160 LargeMemoryPoolOffset=196744 
Malloc Size=131160 LargeMemoryPoolOffset=327928 
Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.
```

follow the instrictions in [this thread](https://github.com/carla-simulator/carla/issues/2138) to fix the issue.








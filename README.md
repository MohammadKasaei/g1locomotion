# G1 Locomotion in IsaacLab

This repository contains the code for training and evaluating the G1 robot for locomotion tasks using NVIDIA's IsaacLab simulator. The implementation leverages the IsaacGym API for high-performance reinforcement learning.

![G1 Locomotion](imgs/locomotion.gif)





# Rewards
The reward function for the G1 robot is designed to encourage efficient locomotion while minimizing undesirable behaviors. The key components of the reward function include:

- **Track Linear Velocity (XY)**: Encourages the robot to move forward at a desired speed.
- **Track Angular Velocity (Z)**: Encourages the robot to maintain a stable orientation while moving.
- **Symmetry Joint Motion**: Encourages symmetric movement of the robot's joints.
- **Feet Air Time**: Penalizes the robot for having both feet off the ground for too long.
- **Feet Slide**: Penalizes the robot for sliding its feet instead of lifting them.
- **DOF Position Limits**: Penalizes the robot for exceeding joint position limits.
- **Joint Deviation (Hip, Arms, Torso)**: Penalizes the robot for deviating from desired joint positions.
- **Action Rate (L2)**: Penalizes the robot for making rapid changes to its actions.
- **DOF Torques (L2)**: Penalizes the robot for applying excessive torques to its joints.
- **DOF Accelerations (L2)**: Penalizes the robot for making rapid changes to its joint accelerations.
- **Flat Orientation (L2)**: Penalizes the robot for deviating from a flat orientation.

# Optimize the reward function using Optuna
We used [Optuna](https://optuna.org/) to optimize the reward function parameters for training the G1 robot. The optimization process involved running multiple trials to find the best set of parameters that maximize the robot's locomotion performance. 

The optimization script can be found in the `optuna` directory. To run the optimization, execute the following command:

```bash
python optuna/optimize_rewards.py
```


# Use multiple GPUs for training
To leverage multiple GPUs for training the G1 robot, you can use the following command:
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=<number_of_gpus>  scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-G1locomotion-v0 --num_envs 4096 --headless --distributed
```

Replace `<number_of_gpus>` with the number of GPUs you want to use for training.

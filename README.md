# Efficient Language-instructed Skill Acquisition via Reward-Policy Co-Evolution

## About this Repository

This repository is dedicated to implementing and testing the ROSKA algorithm (**R**eward-**O**ptimization-**S**imulation-based **K**inematic **A**daptation) for sim-to-real transfer in robotic tasks. The ROSKA algorithm, introduced in the paper **"Efficient Language-instructed Skill Acquisition via Reward-Policy Co-Evolution,"** enhances the transferability of robotic learning models from simulation to real-world environments by evolving both reward functions and policies simultaneously.

The repository includes:

- **Isaac Gym Environment:** NVIDIA's Isaac Gym serves as the base environment for testing and training reinforcement learning (RL) algorithms under ROSKA.
- **Model Folder:** Pre-trained models for two tasks (AllegroHand and Humanoid) are provided. These include models trained with both the ROSKA and the Eureka algorithms across five rounds of evolution.

## Installation

To begin, you need to install the Isaac Gym environment and other necessary dependencies.

1. **Download and Install Isaac Gym:**
   - Obtain Isaac Gym Preview 4 from the [official NVIDIA website](https://developer.nvidia.com/isaac-gym).
   - Follow the installation guide in the documentation. It's recommended to use a conda environment for easier setup.

2. **Verify Installation:**
   - Ensure that Isaac Gym is installed correctly by running example scripts from the `python/examples` directory, such as `joint_monkey.py`.
   - Address any installation issues by following the troubleshooting steps provided in the Isaac Gym documentation.

3. **Install this Repository:**
   - After setting up Isaac Gym, install this repository using:
     ```bash
     pip install -e .
     ```

## ROSKA Algorithm Overview

ROSKA is designed to optimize the process of transferring learned policies from simulation to reality by evolving reward functions and policies in tandem. This method addresses the limitations of traditional reinforcement learning frameworks, which often struggle with discrepancies between simulated and real-world environments.

### Model Folder Structure

- `model/`
  - `eureka/`: Contains models trained with the Eureka algorithm across five rounds of evolution for the AllegroHand and Humanoid tasks.
  - `roska/`: Contains models trained with the ROSKA algorithm, also across five rounds, for the same tasks.

The evolution of these models demonstrates the robustness and efficiency of ROSKA in adapting policies to new reward functions compared to the baseline Eureka algorithm.

## Running Experiments

To train or evaluate a policy using the ROSKA algorithm in the Isaac Gym environment, follow these steps:

1. **Training a Policy:**
   - Start training with a specific task:
     ```bash
     python train.py task=Cartpole
     ```
   - For locomotion tasks like Ant, use:
     ```bash
     python train.py task=Ant
     ```
   - For headless training (without rendering), execute:
     ```bash
     python train.py task=Ant headless=True
     ```

2. **Loading Trained Models:**
   - To continue training from a checkpoint:
     ```bash
     python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth
     ```
   - To load a trained model for inference:
     ```bash
     python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64
     ```

## Capturing Video

To capture videos of the agent's performance, use the following code snippet:

```python
import gym
import isaacgym
import isaacgymenvs
import torch

num_envs = 64

envs = isaacgymenvs.make(
	seed=0, 
	task="Ant", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless=False,
	multi_gpu=False,
	virtual_screen_capture=True,
	force_render=False,
)
envs.is_vector_env = True
envs = gym.wrappers.RecordVideo(
	envs,
	"./videos",
	step_trigger=lambda step: step % 10000 == 0, # record the videos every 10000 steps
	video_length=100  # record up to 100 steps per video
)
envs.reset()
for _ in range(100):
	actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	envs.step(actions)

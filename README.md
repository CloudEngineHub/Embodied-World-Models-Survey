# A Survey: Learning Embodied Intelligence from Physical Simulators and World Models <!-- omit in toc -->

<div align="center">
<!-- <a href="https://arxiv.org/pdf/2506.11526"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper Badge"/></a> -->
<a href="https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey/stargazers"><img src="https://img.shields.io/github/stars/NJU3DV-LoongGroup/Embodied-World-Models-Survey" alt="Stars Badge"/></a>
<a href="https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey/network/members"><img src="https://img.shields.io/github/forks/NJU3DV-LoongGroup/Embodied-World-Models-Survey" alt="Forks Badge"/></a>
<a href="https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey/pulls"><img src="https://img.shields.io/github/issues-pr/NJU3DV-LoongGroup/Embodied-World-Models-Survey" alt="Pull Requests Badge"/></a>
<a href="https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey/issues"><img src="https://img.shields.io/github/issues/NJU3DV-LoongGroup/Embodied-World-Models-Survey" alt="Issues Badge"/></a>
<a href="https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NJU3DV-LoongGroup/Embodied-World-Models-Survey" alt="License Badge"/></a>
</div>


## Table of Content <!-- omit in toc -->
- [Introduction](#introduction)
- [Levels of Intelligent Robot](#levels-of-intelligent-robot)
- [Robotic Mobility, Dexterity and Interaction](#robotic-mobility-dexterity-and-interaction)
  - [Related Robotic Techniques](#related-robotic-techniques)
  - [Robotic Locomotion](#robotic-locomotion)
- [Simulators](#simulators)
- [World Models](#world-models)
- [World Models for Intelligent Robots](#world-models-for-intelligent-robots)
  - [World Models for Autonomous Driving](#world-models-for-autonomous-driving)
  - [World Models for Articulated Robots](#world-models-for-articulated-robots)

## 1. Introduction

Embodied intelligence provides a foundation for creating robots that can truly understand and reason about the world in a more human-like manner. Central to enabling intelligent behavior in robots are two key technologies: physical simulators and world models. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. While world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. The synergy between them enhances robots' autonomy, adaptability, and task performance across diverse scenarios.

This repository aims to collect and organize research and resources related to learning embodied AI through the integration of physical simulators and world models.


<p align="center">
<img src="assets/1_Survey_outline.png" width="100%" height="auto"/>
</p>

## 2. Levels of Intelligent Robot



## 3. Robotic Mobility, Dexterity and Interaction

### Related Robotic SimulatorsSimulators
<details>
<summary> Model Predictive Control, MPCs </summary>
</details>

<details>
<summary> Whole-Body Control, WBC </summary>
</details>

<details>
<summary> Reinforcement Learning </summary>
</details>

<details>
<summary> Imitation Learning </summary>
</details>

<details>
<summary> Visual-Language-Action Models, VLA </summary>
</details>


### Robotic Locomotion

## 4. Simulators

## 5. World Models

## 6. World Models for Intelligent Robots

### World Models for Autonomous Driving

<details>
<summary> WMs as Neural Simulators for Autonomous Driving </summary>

| Paper | Date | Venue | Code | Application |
|:------|:-----|:------|:-----|:------------|
| [DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777) | 2023-09 | ECCV 2024 | [GitHub](https://github.com/JeffWang987/DriveDreamer) | Scenario Generation |
| [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080) | 2023-09 | arXiv Wayve | - | Scenario Generation |
| [Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion](https://arxiv.org/abs/2311.01017) | 2023-11 | ICLR 2024 | - | Scenario Generation |
| [MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations](https://arxiv.org/abs/2311.11762) | 2023-11 | IV 2025 | - | Scenario Generation |
| [Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving](https://arxiv.org/abs/2311.17918) | 2023-11 | CVPR 2024 | [GitHub](https://github.com/BraveGroup/Drive-WM) | Scenario Generation |
| [Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability](https://arxiv.org/abs/2405.17398) | 2024-03 | NeurIPS 2024 | - | Scenario Generation |
| [MagicDrive: Street View Generation with Diverse 3D Geometry Control](https://arxiv.org/abs/2310.02601) | 2024-05 | arXiv | - | Scenario Generation |
| [DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation](https://arxiv.org/abs/2403.06845) | 2024-05 | AAAI 2025 | [GitHub](https://github.com/f1yfisher/DriveDreamer2) | Scenario Generation |
| [UniScene: Multi-Camera Unified Pre-training via 3D Scene Reconstruction for Autonomous Driving](https://arxiv.org/abs/2305.18829) | 2024-08 | RAL 2024 | - | Scenario Generation |
| [WoVoGen: World Volume-aware Diffusion for Controllable Multi-camera Driving Scene Generation](https://arxiv.org/abs/2312.02934) | 2024-08 | ECCV 2024 | - | Scenario Generation |
| [Panacea+: Panoramic and Controllable Video Generation for Autonomous Driving](https://arxiv.org/abs/2408.07605) | 2024-08 | CVPR 2024 | [GitHub](https://github.com/wenyuqing/panacea) | Scenario Generation |
| [DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving](https://arxiv.org/abs/2408.00415) | 2024-08 | arXiv | [GitHub](https://github.com/PJLab-ADG/DriveArena) | Scenario Generation |
| [DriVerse: Navigation World Model for Driving Simulation via Multimodal Trajectory Prompting and Motion Alignment](https://arxiv.org/abs/2504.18576) | 2024-08 | arXiv | - | Scenario Generation |
| [DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation](https://arxiv.org/abs/2410.13571) | 2024-11 | CVPR 2025 | [GitHub](https://github.com/GigaAI-research/DriveDreamer4D) | Scenario Generation |
| [ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](https://arxiv.org/abs/2411.19548) | 2024-11 | arXiv | - | Scenario Generation |
| [MagicDrive3D: Controllable 3D Generation for Any-View Rendering in Street Scenes](https://arxiv.org/abs/2405.14475) | 2024-11 | arXiv | [GitHub](https://gaoruiyuan.com/magicdrive3d/) | Scenario Generation |
| [MagicDriveDiT: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](https://arxiv.org/abs/2411.13807) | 2024-11 | arXiv | [GitHub](https://gaoruiyuan.com/magicdrive-v2/) | Scenario Generation |
| [ACT-Bench: Towards Action Controllable World Models for Autonomous Driving](https://arxiv.org/abs/2412.05337) | 2024-12 | arXiv | - | Scenario Generation |
| [GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control](https://arxiv.org/abs/2412.11198) | 2024-12 | CVPR 2025 | [GitHub](https://github.com/vita-epfl/GEM) | Scenario Generation |
| [SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](https://arxiv.org/abs/2412.12129) | 2024-12 | CVPR 2025 | - | Scenario Generation |
| [DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT](https://arxiv.org/abs/2412.19505) | 2024-12 | arXiv | [GitHub](https://github.com/YvanYin/DrivingWorld) | Scenario Generation |
| [Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving](https://arxiv.org/abs/2408.14197) | 2025-01 | AAAI 2025 | [GitHub](https://github.com/yuyang-cloud/Drive-OccWorld) | Scenario Generation |
| [DualDiff+: Dual-Branch Diffusion for High-Fidelity Video Generation with Reward Guidance](https://arxiv.org/abs/2503.03689) | 2025-03 | ICRA 2025 | [GitHub](https://github.com/yangzhaojason/DualDiff) | Scenario Generation |
| [Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning](https://arxiv.org/abs/2503.15558) | 2025-03 | arXiv | [GitHub](https://github.com/nvidia-cosmos/cosmos-reason1) | Scenario Generation |
| [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/abs/2503.20523) | 2025-03 | arXiv | - | Scenario Generation |
| [Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control](https://arxiv.org/abs/2503.14492) | 2025-04 | arXiv | [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1) | Scenario Generation |
| [OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving](https://arxiv.org/abs/2405.20337) | 2025-05 | arXiv | - | Scenario Generation |

</details>

### World Models for Articulated Robots

The following table compares researches for World Models in Robotics in terms of model input, architecture, experiment platform, and code availability.

<p align="center">
<img src="assets/6_Researches_of_WM_for_robotics.png" width="80%" height="auto"/>
</p>

<details>
<summary> Neural Simulators</summary>

| Paper | Date | Venue | Code |
|:------|:-----|:------|:-----|
| [Whale: Towards generalizable and scalable world models for embodied decision-making](https://arxiv.org/abs/2411.05619) | 2024-08 | arXiv | - |
| [RoboDreamer: Learning Compositional World Models for Robot Imagination](https://robovideo.github.io/) | 2024-08 | ICML 2024 | [Github](https://github.com/rainbow979/robodreamer) |
| [Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination](https://dreamtomanipulate.github.io/) |2024-11 | ICLR 2025 | [GitHub](https://github.com/leobarcellona/drema_code) |
|[EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation](https://sites.google.com/view/enerverse) | 2025-01 | arXiv |[Github](https://github.com/OpenDriveLab/Agibot-World)|
| [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575) | 2025-03 | arXiv | [Github](https://github.com/nvidia-cosmos/cosmos-predict1)|
| [WorldEval: World Model as Real-World Robot Policies Evaluator](https://worldeval.github.io/) | 2025-05 | arXiv | [Github](https://github.com/liyaxuanliyaxuan/Worldeval)|
| [DreamGen: Unlocking Generalization in Robot Learning through Neural Trajectories](https://research.nvidia.com/labs/gear/dreamgen/) | 2025-05 | arXiv | [Github](https://github.com/nvidia/GR00T-dreams)|

</detailsk>
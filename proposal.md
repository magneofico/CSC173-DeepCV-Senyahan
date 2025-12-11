# CSC173 Deep Computer Vision Project Proposal
**Student:** KRISTOFFER NEO V. SENYAHAN, 2022-4762   
**Date:** December 11, 2025  

## Project Title
**Smart Aesthetic Image Cropper for People-Centered Photographs using Tiny Vision Transformers and Reinforcement Learning**

## Problem Statement
Modern photojournalism workflows often involve manually cropping photos to emphasize a human subject, improve composition, and produce visually appealing thumbnails. Traditional automated croppers rely solely on object detection or fixed rules, resulting in awkward framing or inconsistent aesthetics. There is a need for an intelligent, data-driven cropping system capable of understanding human-centered composition—particularly headroom, centering, and framing styles used by professional photographers or photojournalists. This project aims to address this gap by developing a lightweight, deep-learning–based cropper that combines transformers and reinforcement learning to generate editorial-quality crops automatically.

## Objectives
- Develop a Tiny Vision Transformer (TinyViT) model that predicts an initial aesthetic crop focusing on human subjects.
- Implement a Mini Reinforcement Learning (RL) agent that refines the crop via pan/zoom actions based on learned aesthetic rewards.
- Produce a complete dataset pipeline including person filtering, bounding box extraction, and aesthetic ground-truth generation.
- Evaluate model outputs against professional photojournalists’ crops through quantitative metrics (IoU, center distance) and qualitative feedback.

## Dataset Plan
- Source: Open Images Dataset V7 (Person class) via OIDv6 downloader
- Expected Size: ~120–200 usable images (filtered to “single person per image”)
- Annotations:
    - Bounding boxes from `oidv6-train-annotations-bbox.csv`
    - Class metadata from `class-descriptions-boxable.csv`
- Processing Steps:
    - Keep only images with exactly one Person bounding box
    - Generate aesthetic ground-truth crops by expanding bounding boxes + rule-of-thirds adjustment
- Why appropriate: Person-centered images match real campus publication workflows and fit the aesthetic cropping goal.


## Technical Approach
### Architecture Overview
1. Supervised Stage (TinyViT):
    - **Input:** full image
    - **Output:** (x_center, y_center, width, height)
    - **Components:**
        - Patch embedding
        - 2–4 transformer encoder blocks
        - MLP regression head
    - **Loss:** Smooth L1 + IoU loss

2. RL Refinement Stage:
    - **Environment:** cropping window with allowed actions
    - **Actions:** pan-left, pan-right, pan-up, pan-down, zoom-in, zoom-out, stop
    - **State:** current crop (96×96)
    - **Reward:** IoU improvement, headroom bonus, centering bonus
    - **Algorithm:** DQN or REINFORCE (lightweight for student-level training)

### Framework
    - PyTorch
    - Jupyter Notebook inside VSCode
    - Python 3.10 (venv environment)

### Hardware
    - Local machine CPU
    - Optional: lightweight GPU runtime on Colab for TinyViT training if needed


## Training Configuration
| Challenges | Mitigation | 
|-----------|--------| 
| Small dataset | Use augmentation (resize, crop, flip); TinyViT is lightweight and works well with small data |
| Highly variable bounding box sizes | Normalize bbox + expand with aesthetic margins |
| Reinforcement Learning might be unstable | Use simple action space + reward shaping + short training episodes |
| Limited Computation | Use TinyViT (small model), reduce resolution, and limit RL training steps |
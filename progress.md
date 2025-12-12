# CSC173 Deep Computer Vision Project Progress Report
**Student:** KRISTOFFER NEO V. SENYAHAN, 2022-4762 
**Date:** December 11, 2025  
**Repository:** https://github.com/magneofico/CSC173-DeepCV-Senyahan.git  
**Commits Since Proposal:** [1 commit/s] | **Last Commit:** December 11, 2025  

## Project Title
**Smart Aesthetic Image Cropper for People-Centered Photographs using Tiny Vision Transformers and Reinforcement Learning**


## Project Description

This project aims to build an intelligent cropping system specifically for **images of people**, targeting use cases in **photojournalism and publication workflows**.

> The model will:
> 1. Subject detection and cropping via supervised TinyViT.
> 2. Aesthetic refinement via a Mini RL cropping agent.
> 3. Human-aligned evaluation against professional photojournalist crops.

> This repository will contain:
> - Dataset preprocessing scripts
> - Custom TinyViT model for initial person cropping
> - RL environment and policy network
> - Training/evaluation notebooks
> - Human-photojournalist comparison tests
> - Final report + presentation materials


## Current Status
| Milestone                           | Status      | Notes |
|-------------------------------------|-------------|-------|
| Project Setup                       | Completed   | Repo created, structure finalized |
| Class Selection (Person)            | Completed   | Using Open Images V7 — Person class only using OIDv6 |
| Dataset Preparation                 | Completed   | Filtering person images; designing aesthetic crop ground truth |
| Supervised Model (TinyVit) Design   | In Progress | Architecture Design Drafting |
| Supervised Training                 | Not Started | Planned after dataset finalization |
| RL Agent (Mini RL) Design           | Not Started | To follow after baseline cropper |
| Photographer Crop Collection        | In Progress | Scheduled for mid-project evaluation |
| Final Testing                       | Not Started | Pending trained cropper + RL refinement |


## 1. Dataset Progress
### 1.1 Dataset Source
- Open Images V7 (Person class only)
- Bounding box annotations included
- Only images with exactly 1 person (simple baseline)

### 1.2 Aesthetic Crop Ground Truth
- Expanding bounding box by 10–20% margin
- Optional rule-of-thirds adjustments
- Normalized coordinates `(x_center, y_center, width, height)`

### 1.3 Current Stats
- **Downloaded images:** 5000
- **Filtered valid person images:** 1992
- **Train/Val/Test split:** Planned 70/15/15

Planned preview placeholder:

```bash
images/sample_person_dataset_preview.png
```


## 2. Model Development Progress
### 2.1 Supervised Cropper (TinyVit)
- Architecture:
    - Patch embedding
    - 2-4 Transformer encoder blocks
    - [CLS] token pooling
    - MLP head → `(x_center, y_center, width, height)`
- Status: **DRAFTED**, coding text

### 2.2 RL Refinement Agent
- Mini RL Setup:
    - Actions: pan left/right/up/down, zoom in/out, stop
    - State: current crop (96×96) + optional crop coords
    - Reward: IoU improvement + aesthetic bonuses
- Status: **NOT YET started**, (dependent on baseline cropper)


## 3. Planned Evaluation Methods
### 3.1 Quantitative
- IoU between:
    - Baseline crop vs GT crop
    - RL crop vs GT crop
    - Model crop vs photographer crop
- Center-distance score
- Aspect ratio similarity

### 3.2 Qualitative
- Photojournalist crop 20-25 images
- Side-by-side comparisons:
    - Human crop
    - TinyVit crop
    - RL-refined crop
- Ratings:
    - Aesthetic quality (1-5)


## 4. Progress Achieved Checklists
- [x] Filter Open Images V7 dataset to person-only images
- [x] Generate ground-truth crop targets from bounding boxes
- [x] Integrate TinyViT as the vision backbone for feature extraction
- [x] Implement bounding-box crop regression head
- [x] Train supervised crop regression model end-to-end
- [x] Fine-tune the model and achieve stable convergence
- [x] Log training and validation metrics in a structured format
- [ ] Design reinforcement learning environment for crop refinement
- [ ] Implement lightweight RL policy network (pan/zoom/stop actions)
- [ ] Train RL agent for aesthetic crop refinement (fast training setup)
- [ ] Conduct qualitative evaluation with photojournalists
- [ ] Finalize quantitative and qualitative performance metrics
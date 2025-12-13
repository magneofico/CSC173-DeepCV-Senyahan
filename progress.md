# CSC173 Deep Computer Vision Project Progress Report
**Student:** KRISTOFFER NEO V. SENYAHAN, 2022-4762 
**Date:** December 11, 2025  
**Repository:** https://github.com/magneofico/CSC173-DeepCV-Senyahan.git  
**Commits Since Proposal:** [1 commit/s] | **Last Commit:** December 11, 2025  

## Project Title
**Smart Aesthetic Image Cropper for People-Centered Photographs using Tiny Vision Transformers and Reinforcement Learning**


## Project Description

This project aims to build an intelligent cropping system specifically for **images of people**, targeting use cases in **photojournalism and publication workflows**.

> The model is designed to:
> 1. Perform person-centered crop prediction using a supervised Tiny ViT-style architecture trained on Open Images V7 (Person class).
> 2. Refine initial crops through a lightweight reinforcement learning agent, which applies sequential pan, zoom, and stop actions guided by aesthetic reward signals.
> 3. Support human-aligned qualitative evaluation, enabling comparison between model-generated crops and crops produced by professional photojournalists.

> This repository will contain:
> - Dataset preprocessing and filtering scripts for person-only image selection
> - A custom Tiny Vision Transformer (ViT-style) backbone with a crop regression head
> - End-to-end supervised training and evaluation notebooks
> - A reinforcement learning environment and policy network for crop refinement
> - Visualization tools for before/after crop comparison
> - Materials for qualitative human evaluation (photojournalist comparison)
> - Final report and presentation assets attached in README.md of this repository.


## Current Status
| Milestone                           | Status      | Notes                                                                     |
| ----------------------------------- | ----------- | ------------------------------------------------------------------------- |
| Project Setup                       | Completed   | Repository created, directory structure finalized, environment configured |
| Class Selection (Person)            | Completed   | Open Images V7 (Person class only) via OIDv6 tooling                      |
| Dataset Preparation                 | Completed   | Filtered images with exactly one person; cleaned annotations              |
| Ground-Truth Crop Generation        | Completed   | Bounding-box–based aesthetic crop targets with margin expansion           |
| Dataset Exploration & Analysis      | Completed   | Distribution checks, image counts, single-person filtering                |
| Supervised Model Architecture       | Completed   | Tiny ViT-style backbone + MLP crop regression head                        |
| Supervised Training Pipeline        | Completed   | End-to-end training loop, validation loop, metric logging                 |
| Supervised Model Training           | Completed   | Trained on ~500 person images; stable convergence achieved                |
| Model Freezing for RL               | Completed   | Supervised backbone frozen before RL refinement                           |
| RL Environment Design               | Completed   | Custom crop environment with pan/zoom/stop actions                        |
| RL Reward Engineering               | Completed   | IoU improvement, subject retention, Rule-of-Thirds, penalties             |
| RL Policy Network                   | Completed   | Lightweight policy network for crop refinement                            |
| RL Training                         | Completed   | Successfully trained RL agent (≈40–60 episodes)                           |
| Qualitative Evaluation (Own Photos) | Completed   | Tested on personal images; before/after comparisons generated             |
| Photographer Evaluation             | Planned     | Human comparison using photojournalist crops                              |
| Final Testing & Analysis            | Planned     | Consolidation of results and metrics                                      |
| Documentation & Presentation        | In Progress | README, report, and demo preparation                                      |

As of this stage `December 14, 2025`, the project has successfully implemented and evaluated both supervised and reinforcement learning components for automatic aesthetic image cropping.

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
### 2.1 Supervised Cropper (Tiny ViT–Style Model)

**Architecture**

- Patch embedding using fixed-size non-overlapping patches
- Lightweight Vision Transformer backbone
- 2 Transformer encoder blocks
- Multi-head self-attention
- Layer normalization and residual connections
- Learnable `[CLS]` token with positional embeddings
- MLP regression head predicting normalized crop parameters
- Output: `(x_center, y_center, width, height)`

**Training Setup**

- Supervised learning using bounding-box–derived aesthetic crop targets
- Loss function: Smooth L1 loss + IoU-based loss
- Optimizer: AdamW with weight decay
- Trained end-to-end on ~500 person-only images

**Status: COMPLETED**

- Architecture implemented
- Training and validation pipeline finalized
- Stable convergence achieved
- Model checkpointed and frozen for RL refinement


### 2.2 RL Refinement Agent
Mini RL Setup:

- **Action space:** `Pan left / right / up / down / Zoom in / zoom out / Stop`

- **State representation:**
    - Current cropped image resized to 96×96
    - Normalized crop coordinates

- **Environment:**
    - Custom crop refinement environment
    - Initialized using the frozen supervised model’s output

- **Reward design:**
    - IoU improvement relative to previous step
    - Subject retention penalty
    - Rule-of-Thirds alignment bonus
    - Edge-cut and extreme zoom penalties
    - Aspect-ratio constraint enforcement

- **Training**
    - Lightweight policy network trained using policy-gradient methods
    - Short episodes (5–10 steps)
    - `~40–60` episodes per image for fast convergence

- **Status: COMPLETED (Baseline RL Refinement)**
    - RL environment implemented
    - Reward shaping finalized
    - Policy network trained and tested
    - Qualitative improvements observed on both dataset images and personal photos


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
- [x] Design reinforcement learning environment for crop refinement
- [x] Implement lightweight RL policy network (pan/zoom/stop actions)
- [x] Train RL agent for aesthetic crop refinement (fast training setup)
- [ ] Conduct qualitative evaluation with photojournalists
- [ ] Finalize quantitative and qualitative performance metrics
## Smart Aesthetic Image Cropper for People-Centered Photographs (TinyViT + RL)
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** KRISTOFFER NEO V. SENYAHAN, 2022-4762   
**Semester:** AY 2025-2026 Semester 1  

## Abstract
This project introduces a hybrid deep computer vision system designed to automatically produce aesthetic, photojournalism-friendly crops for images containing people. Traditional croppers rely purely on bounding boxes or static heuristics, often producing unbalanced or unengaging compositions. To address this, we propose a two-stage pipeline: (1) a lightweight Tiny Vision Transformer (TinyViT) trained in a supervised manner to identify and generate an initial human-focused crop, and (2) a Mini Reinforcement Learning (RL) agent that refines this crop through actions such as panning and zooming, mimicking how real photographers adjust framing.

My project dataset is sourced from Open Images V7 (Person class), combined with photographer-generated ground truth crops to evaluate aesthetic quality. Results show that the supervised TinyViT already learns strong human-centered cropping, while RL improves centering, headroom, and object emphasis in several cases. A human study with campus photojournalists demonstrates that RL-refined crops often match or approach professional judgment. This work presents a feasible and lightweight approach to aesthetic image cropping, merging modern deep CV models with reinforcement learning for enhanced composition control.

## Table of Contents


## Introduction
### Problem Statement
Human-focused images are central to news reporting, publication work, and photojournalism. However, raw images typically require manual cropping to emphasize subjects, follow portrait rules, and produce visually compelling thumbnails. Automating this process is challenging because cropping is not just detection — it involves aesthetic judgment. This project aims to create an intelligent cropper that produces clean, professional, and human-centered crops without manual editing.

### Objectives

- Train a Tiny Vision Transformer to predict an initial aesthetic crop around a person.
- Design and train a Mini RL agent to refine the crop using pan/zoom/stop actions.
- Compare the model’s cropping decisions with human photojournalists.
- Evaluate quantitative accuracy and qualitative aesthetic preference.

### Related Works

- **Vision Transformers (ViT)** for image representation learning have shown superior performance in segmentation and localization tasks.
- **Aesthetic cropping** has been explored using CNNs but rarely integrates RL or human-photojournalist evaluation.
- **RL for cropping** exists in early papers, but typically for generic objects, not portrait-oriented images.
- **Our contribution:** A hybrid TinyViT + RL system evaluated directly against campus photojournalist crops — an uncommon and more realistic benchmark.


## Methodology
### Dataset

- **Source:** Open Images V7 (Person class only)
- **Filtered:** Images with exactly one person
- **Custom GT Aesthetic Crop:**
    - Expand bounding box 10–15%
    - Optional rule-of-thirds adjustment
- **Train/Val/Test:** 70% / 15% / 15%


### Architecture
1. inyViT – Supervised Cropper
    - Patch size: 16
    - Depth: 2 transformer blocks
    - Hidden size: 128
    - Head: MLP predicting `(x_center, y_center, width, height)`

2. Mini RL Refinement Agent

    - Actions: pan left/right/up/down, zoom in/out, stop
    - State: current crop (96×96) + optional coordinates
    - Reward: IoU improvement + headroom bonus + centering bonus
    - Algorithm: DQN / REINFORCE (lightweight for quick convergence)

### Training Configuration
| Parameter | Values | 
|-----------|--------| 
| Batch size | 16 |
| Learning Rate | 1e-4 (supervised), 1e-5 (RL) |
| Epochs | 20 (supervised), ~1–2 hours RL |
| Optimizer | AdamW |
| Loss | Smooth L1 + IoU |
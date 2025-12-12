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
- **Filtering Criteria:**
    - Images containing exactly one annotated person
    - Removes ambiguity and multi-subject framing
- **Ground-Truth Crop Generation:**
    - Person bounding boxes converted to crop targets
    - Bounding boxes optionally expanded by ~10–15% margin to include contextual headroom
    - All crop coordinates normalized to `[0,1]`
- **Dataset Split:**
    - Training / Validation / Test = 70% / 15% / 15%
    - Fixed random seed used for reproducibility

## Architecture
### Supervised Architecture: Smart Image Cropper

The supervised component of the system is designed to predict a person-centered crop directly from an input photograph. This model serves as a strong baseline and provides the initial crop for later reinforcement learning refinement.   

![Supervised Architecture pipeline diagram](images/supervised_architecture.png)  

#### 1. Input Representation and Preprocessing
Each input image is resized to a fixed resolution of 224 × 224 pixels and normalized using standard ImageNet statistics.

$$
x \in \mathbb{R}^{B \times 3 \times 224 \times 224}
$$

Where:
- 𝐵 is the batch size,
- 3 represents the RGB color channels.

This standardization ensures stable training and enables batch processing.

---

#### 2. Patch Embedding Layer

The input image is divided into non-overlapping **16 × 16** patches, resulting in:

$$
N = \left(\frac{224}{16}\right)^2 = 196 \text{ patches}
$$

Each patch is flattened and projected into a latent embedding space using a linear layer:

$$
\mathbf{p}_i = W \mathbf{x}_i + \mathbf{b}, \qquad \mathbf{p}_i \in \mathbb{R}^D
$$

where:

- **\(\mathbf{x}_i\)** is the flattened patch,  
- **\(D = 128\)** is the hidden dimension.

A learnable **[CLS] token** is prepended to the patch sequence to capture global image information.

---

#### 3. Tiny Vision Transformer (TinyViT) Backbone

The patch embeddings are processed by **two transformer encoder blocks**, each composed of:

- Multi-Head Self-Attention (MHSA)
- Feed-Forward Network (FFN)
- Residual connections and layer normalization

#### Self-Attention Mechanism
For each token, attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{D}} \right) V
$$

This allows the model to learn **global spatial relationships**, such as:

- where the person is located,
- how large the person appears,
- how the subject relates to surrounding context.

---

#### 4. Global Feature Embedding

After the transformer blocks, the embedding corresponding to the **[CLS] token** is extracted:

$$
\mathbf{z} \in \mathbb{R}^{B \times D}
$$

This vector represents a **global summary** of the image, encoding composition, subject location, and scale.

---

#### 5. Crop Regression Head (MLP)

The global embedding **z** is passed into a lightweight **multi-layer perceptron (MLP)** that predicts crop parameters.

$$
\mathbf{h} = \text{GELU}(\mathbf{W_1 z} + \mathbf{b_1})
$$

$$
\hat{\mathbf{b}} = \text{sigmoid}(\mathbf{W_2 h} + \mathbf{b_2})
$$

The output is:

$$
\hat{\mathbf{b}} = (x_c, y_c, w, h)
$$

Where:

- \(( x_c, y_c )\) is the crop center,  
- \(( w, h )\) are the crop width and height,  
- all values are normalized to \([0, 1]\).

The **sigmoid activation** ensures valid crop boundaries.

---

#### 6. Crop Box Parameterization

The predicted crop is converted into corner coordinates:

$$
x_1 = x_c - \frac{w}{2}, \quad y_1 = y_c - \frac{h}{2}
$$

$$
x_2 = x_c + \frac{w}{2}, \quad y_2 = y_c + \frac{h}{2}
$$

These values define the region used to crop the original image.

---

#### 7. Training Objective

The model is trained end-to-end using a **compound loss**.

#### Smooth L1 Loss

$$\mathcal{L}_{\text{SmoothL1}}(\hat{\mathbf{b}}, \mathbf{b})$$

Encourages stable regression of crop parameters.

#### IoU Loss

$$
\mathcal{L}_{\text{IoU}} = 1 - \frac{|\hat{B} \cap B|}{|\hat{B} \cup B|}
$$

Encourages strong spatial overlap between predicted and target crops.

#### Final Loss

$$
\mathcal{L} = \mathcal{L}_{\text{SmoothL1}} + \mathcal{L}_{\text{IoU}}
$$

---
The supervised TinyViT crop regressor learns global image composition using transformer attention and directly predicts an aesthetically meaningful crop around a person, forming a strong foundation for later reinforcement learning refinement.











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
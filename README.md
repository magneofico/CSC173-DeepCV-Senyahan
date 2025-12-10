## Smart Aesthetic Image Cropper for <br>People-Centered Photographs (TinyViT + RL)
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** KRISTOFFER NEO V. SENYAHAN, 2022-4762   
**Semester:** AY 2025-2026 Semester 1  

## Abstract
This project introduces a hybrid deep computer vision system designed to automatically produce aesthetic, photojournalism-friendly crops for images containing people. Traditional croppers rely purely on bounding boxes or static heuristics, often producing unbalanced or unengaging compositions. To address this, we propose a two-stage pipeline: (1) a lightweight Tiny Vision Transformer (TinyViT) trained in a supervised manner to identify and generate an initial human-focused crop, and (2) a Mini Reinforcement Learning (RL) agent that refines this crop through actions such as panning and zooming, mimicking how real photographers adjust framing.

My project dataset is sourced from Open Images V7 (Person class), combined with photographer-generated ground truth crops to evaluate aesthetic quality. Results show that the supervised TinyViT already learns strong human-centered cropping, while RL improves centering, headroom, and object emphasis in several cases. A human study with campus photojournalists demonstrates that RL-refined crops often match or approach professional judgment. This work presents a feasible and lightweight approach to aesthetic image cropping, merging modern deep CV models with reinforcement learning for enhanced composition control.
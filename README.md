# Deep Learning for Risky Activity Detection in RGB Videos

This repository contains the implementation and results of a research project focused on using deep learning techniques to classify risky activities in RGB videos. The project leverages a relabeled subset of the NTU RGB+D 120 dataset to evaluate the performance of various models in identifying risky versus non-risky activities. Demo video can be found [here](https://youtu.be/rs6FhPj3d8g). 

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models and Architectures](#models-and-architectures)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributions and Future Work](#contributions-and-future-work)
- [References](#references)

---

## Introduction
Human Activity Recognition (HAR) is a critical field in artificial intelligence, with applications in surveillance, security, and public safety. This project addresses the specific challenge of detecting risky activities using only RGB video data, without relying on depth or skeletal information.

### Key Research Questions:
1. Can deep learning models effectively detect risky activities using RGB video data alone?
2. What is the impact of input resolution on the accuracy and efficiency of these models?
3. How do the models perform under challenging real-world conditions like occlusions and complex interactions?

## Dataset
The project uses a relabeled subset of the NTU RGB+D 120 dataset, focusing on binary classification of activities as "risky" or "non-risky." The relabeling process categorized 9 activities as risky (e.g., "kicking someone" or "shooting") and 15 activities as non-risky (e.g., "sitting" or "drinking water"). 

Key features of the dataset include:
- RGB video-only focus to match real-world surveillance systems.
- Balanced representation of classes to prevent data imbalance.
- Custom splits for training (60%), validation (20%), and testing (20%).

## Models and Architectures
The following deep learning architectures were evaluated:
1. **ResNet18 + 3-Layer LSTM**: Combines a CNN for spatial feature extraction with an LSTM for temporal modeling ([Shahroudy et al., 2016](https://arxiv.org/abs/1604.02808)).
2. **MC3_18 3D CNN**: Captures spatiotemporal features using 3D convolutions ([Tran et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)).
3. **Swin3D Transformer**: Utilizes self-attention for long-range dependency modeling ([Liu et al., 2022](https://arxiv.org/abs/2106.13230)).

All models employed transfer learning with pre-trained weights from the Kinetics-400 dataset ([Carreira et al., 2017](https://arxiv.org/abs/1705.07750)) and were fine-tuned for the specific task.

### Training and Pipeline Code Inspiration
The training and inference pipeline was inspired by Sovit Ranjan Rath's article on training video classification models with PyTorch ([Rath, 2023](https://debuggercafe.com/training-a-video-classification-model/)). The approach was adapted to include preprocessing, clip creation, and efficient data handling using PyTorch utilities such as `VideoClips` and `DataLoader`.

## Results
The **Swin3D Transformer** model achieved the best results:
- **Accuracy**: 95.80% at 360p resolution.
- **Efficiency**: High inference speed (72.03 FPS at 112p resolution).

**Strengths:**
- High accuracy, even at lower resolutions.
- Robust performance across different input resolutions.

**Weaknesses:**
- Challenges in handling occlusions, unusual viewpoints, and complex interactions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the relabeled dataset and place it in the `data/` directory.

## Usage
1. **Preprocess Videos**:
   ```bash
   python preprocess.py --input data/raw --output data/processed
   ```
2. **Train Models**:
   ```bash
   python train.py --config configs/swin3d.yaml
   ```
3. **Evaluate Models**:
   ```bash
   python evaluate.py --model outputs/swin3d.pth
   ```
4. **Inference on Custom Videos**:
   ```bash
   python inference.py --video path/to/video.mp4
   ```

## Contributions and Future Work
Future research could address:
- Robustness to occlusions and complex interactions.
- Domain adaptation for real-world surveillance footage.
- Real-time deployment using model compression techniques.

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## References
1. Shahroudy, Amir, et al. *NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis.* [Link](https://arxiv.org/abs/1604.02808)
2. Tran, Du, et al. *A Closer Look at Spatiotemporal Convolutions for Action Recognition.* [Link](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)
3. Liu, Ze, et al. *Video Swin Transformer.* [Link](https://arxiv.org/abs/2106.13230)
4. Carreira, Joao, et al. *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset.* [Link](https://arxiv.org/abs/1705.07750)
5. Rath, Sovit Ranjan. *Training a Video Classification Model from torchvision.* [Link](https://debuggercafe.com/training-a-video-classification-model/)
```
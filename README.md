# Weakly Supervised Monocular Depth Estimation for Spike Camera with Adaptive Self Distillation



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## Table of Contents
- [Overview](#overview)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Demos](#demos)
- [License](#license)

## Overview

![2KhzO.jpeg](https://i.imgs.ovh/2023/10/20/2KhzO.jpeg)

## Background

By mimicking the neural processes of biological organisms, the neuromorphic spike camera captures visual spike streams at high temporal resolution, and has great potential in many visual fields. 
However, existing approaches for visual content understanding are generally designed for frame-based conventional cameras, and are not well suited to the spike camera because of the spatial sparsity of the binary spike data. Since it's challenging to learn dense depth information directly from binary spike streams without depth labels, in this paper, we present R2SDpeth, a novel deep learning framework to perform spike-modality monocular depth estimation by combining frame-based RGB data and stream-based spike data in a weakly-supervised paradigm. Specifically, we first align the RGB frames and spike streams, and carry out a view synthesis method to train the multimodal depth estimation network, of which the spike encoder is weakly supervised by the fusion and alignment of RGB and spike data. As a next step, we introduce a cross-modality self-distillation module to transfer knowledge from the RGB encoder to spike encoder, further enhance the spike encoder of the depth net and finally achieve single-spike modality depth estimation.
Additionally, we generate a dataset named Spike-KITTI by a spike camera simulator, which contains 156 sequences of different scenes.  

### Any optional sections

## Install

```
pip install -r requirements.txt
```

### Any optional sections

## Usage

### Download The Dataset: SpikeKITTI

```
```
### Training R2SDepth on SpikeKITTI

**To be released.**

## Demos 

####  Visualization of Neuromorphic Depth Prediction on Spike KITTI

![](https://i.imgs.ovh/2023/10/20/2s7NA.gif)

####  Visualization of Fusion Depth Prediction on Spike KITTI

![](https://i.imgs.ovh/2023/10/20/2sQp5.gif)



# Senorita-HANDS19-Pose

# Introduction

This repo is our implementation for the [HANDS19 Challenge Task 1 - Depth-Based 3D Hand Pose Estimation](https://competitions.codalab.org/competitions/20913#learn_the_details) entry. A full report is here.

# Environment
- Dell Alienware R4 Laptop w/ one 8GB Geforce GTX 1070.

For PyTorch users, it's quite easy to transform weights across DL frameworks. (e.g. [this repo](https://github.com/xxradon/PytorchToCaffe)) This project applies to both Ubuntu and Windows. Piece of cake!

# Data
Please go to the [official webpage](https://sites.google.com/view/hands2019/challenge) and fill the form to participate. **Please do not ask me for the data. I am not authorized to release the data, neither do I have permission to redistribute it.**

The following steps are required to preprocess the data.
1. Run ```${POSE_ROOT}/src/preprocessing/DemoSeeBBX/DemoSeeBBX.cpp``` to crop depth patch.
2. Run ```${POSE_ROOT}/src/preprocessing/GenHands19H5/GenHands19H5.cpp``` to generate (annotation) HDF5 file for training / testing.

# Work it over
## Train

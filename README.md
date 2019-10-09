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
**[NOTE]** CHANGE THE TRAINING SCHEDULE AND SEE IF THE RESULT GETS IMPROVED. INCLUDING BUT NOT LIMITED TO BASE LEANRING RATE, STEPS TO SLOWER THE LEARNING RATE, LOSS WEIGHT RATIOS.

Now let's get started.

```cd``` enter into any one of the four folders, ```M1``` & ```M2``` & ```M3``` & ```M4```. Let's say ```M1```.
- Train w/ only integral loss 
  ```
  caffe train --solver=solver_aug_lfd.prototxt --weights=../mks.caffemodel --gpu 0
  ```
- Add 3d skeleton volume rendering loss. (Compare with that generated from ground truth 3D pose)
  ```
  caffe train --solver=solver_aug_lfd_3dske.prototxt --snapshot=XXXX.solverstate --gpu 0
  ```
  XXXX stands for the solverstate which terminates the *w/ only integral loss* step.
# Senorita-HANDS19-Pose

# Introduction

This repo is our implementation for the [HANDS19 Challenge Task 1 - Depth-Based 3D Hand Pose Estimation](https://competitions.codalab.org/competitions/20913#learn_the_details) entry. A full report is here.

# Environment
- Dell Alienware R4 Laptop w/ one 8GB Geforce GTX 1070.

For PyTorch users, it's quite easy to transform weights across DL frameworks. (e.g. [this repo](https://github.com/xxradon/PytorchToCaffe)) This project applies to both Ubuntu and Windows. Piece of cake!


# Result
| EXTRAP. | INTERP. | SHAPE. | ARTIC. | VIEWP. |
|:-:|:-:|:-:|:-:|:-:|
| 19.65 | 8.45 | 14.28 | 7.53 | 14.21 |

# Data
Please go to the [official webpage](https://sites.google.com/view/hands2019/challenge) and fill the form to participate. **Please do not ask me for the data. I am not authorized to release the data, neither do I have permission to redistribute it.**

The following steps are required to preprocess the data.
1. Run ```${POSE_ROOT}/src/preprocessing/DemoSeeBBX/DemoSeeBBX.cpp``` to crop depth patch.
2. Run ```${POSE_ROOT}/src/preprocessing/GenHands19H5/GenHands19H5.cpp``` to generate (annotation) HDF5 file for training / testing.

# Trained weights
| Method | Weight Model  |
|:-:|:-:|
| M1     | [M1](https://drive.google.com/file/d/1jH50yED6Jr6uAAlomvn5AGdaDaHGoLhD/view?usp=sharing) |
| M2     | [M2](https://drive.google.com/file/d/1lPXogZKChtGbpjEQQLDYf3493XKyng1V/view?usp=sharing) |
| M3     | [M3](https://drive.google.com/file/d/1ddOyxxqLffyjzeYkm1c9heu4wl_zAEh4/view?usp=sharing) |
| M4     | [M4](https://drive.google.com/file/d/1A9i4bIH53C90EXHIiV5iTOIZdT1VbLoo/view?usp=sharing) |

# Work it over
## Train
**[NOTE]** CHANGE THE TRAINING SCHEDULE AND SEE IF THE RESULT GETS IMPROVED. INCLUDING BUT NOT LIMITED TO BASE LEANRING RATE, STEPS TO SLOWER THE LEARNING RATE, LOSS WEIGHT RATIOS.

Now let's get started.

```cd ${POSE_ROOT}/experiments/train``` enter into any one of the four folders, ```M1``` & ```M2``` & ```M3``` & ```M4```. Let's say ```M1```.
- Train w/ only integral loss 
  ```
  caffe train --solver=solver_aug_lfd.prototxt --weights=../mks.caffemodel --gpu 0
  ```
- Add 3D skeleton volumes rendering loss. (Compare with that generated from ground truth 3D pose)
  ```
  caffe train --solver=solver_aug_lfd_3dske.prototxt --snapshot=XXXX.solverstate --gpu 0
  ```
  XXXX stands for the solverstate which terminates the *w/ only integral loss* step.
- Add the patch-based refinement and jointly train the initial pose estimator and patch-based refinement module.
  ```
  caffe train --solver=solver_aug_lfd_patch.prototxt --weights=YYYY.caffemodel --gpu 0
  ```
  YYYY is the model of last step. Ensure that the initial pose estimator does not dominate the loss, meaning that the MPJPE error of refined pose ought to be smaller than the initial estimate.
  
Switch to [this page](https://github.com/strawberryfg/Senorita-HANDS19-Pose/tree/master/src/network_layers) for the code about rendering and refinement.

## Test

```cd ${POSE_ROOT}/experiments/test``` Download the models and place to corresponding folders. Enter either of the four folders.


- ```M1```

  Run
  
  ```
  caffe test -model test_aug_lfd_patch.prototxt -weights m1.caffemodel --gpu 0 -iterations 20834
  ```  
- ```M2```

  Run 
  
  ```
  caffe test -model test_aug_lfd_proj_single_ld_patchref.prototxt -weights m2.caffemodel --gpu 0 -iterations 20834
  ```
  
- ```M3```

  Run 
  
  ```
  caffe test -model test_aug_lfd_proj_single_dv_patchref.prototxt -weights m3.caffemodel --gpu 0 -iterations 20834
  ```
  
- ```M4```

  Run 
  
  ```
  caffe test -model test_patchref.prototxt -weights m4.caffemodel --gpu 0 -iterations 20834
  ```
  

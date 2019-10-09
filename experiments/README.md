# Experiments

To understand the code structure, generally you only have to follow the order of layers presented in the [code](https://github.com/strawberryfg/Senorita-HANDS19-Pose/tree/master/src/network_layers) page. By then you would get to grips with most of the critical layers used throughout the training.

The work-flow is described below.

- You are given a HDF5 file from which you extract the image index to be trained, the bounding box provided by the organizer, and the 3D ground truth in real-world coordinate.

- You get the adjusted ```bbx```, ```avgX```, ```avgY```, ```avgZ```, ```avgU```, ```avgV```

- Now you are able to crop the depth patch and augment it! In the meantime, the 3D pose ground truth needs augmentation. Also, you may want to consider producing ```3D points projection```, ```multi-layer depth maps```, ```depth voxel```, all of which act as possible system input representation.
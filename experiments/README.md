# Experiments

To understand the code structure, generally you only have to follow the order of layers presented in the [code](https://github.com/strawberryfg/Senorita-HANDS19-Pose/tree/master/src/network_layers) page. By then you would get to grips with most of the critical layers used throughout the training.

The work-flow is described below.

* You are given a ```HDF5 file``` from which you extract the ```image index``` to be trained, the ```bounding box``` provided by the organizer, and the ```3D ground truth``` in real-world coordinate.

* You get the adjusted ```bbx```, ```avgX```, ```avgY```, ```avgZ```, ```avgU```, ```avgV```

* Now you are able to ```crop the depth patch``` and augment it! In the meantime, the ```3D pose ground truth``` needs augmentation. Also, you may want to consider producing ```3D points projection```, ```multi-layer depth maps```, ```depth voxel```, all of which act as possible system input representation.

* You obtain augmented ground truth of ```2D``` & ```3D``` & ```depth```, either in ```local``` coordinate space or ```global```.

* You start to articulate the system input depending on which model you would like to use.

* You run the ```initial pose estimator```, which in this case is the [integral regression](https://github.com/strawberryfg/int-3dhuman-I1). As a result, predicted ```3D``` & ```2D```, either in ```local``` & ```global``` are computed. You observe the ```MPJPE``` error in ```mm```. 

* You may optionally opt to ```overlay predicted 2D``` on the depth image, and output the prediction to store the initial estimate result.

* ```3D skeleton volumes``` renderer takes predicted 2D & depth as input, and synthesize a volume. Which is then compared against ground truth. This *pseudo analysis by synthesis* (**NO, IT'S NOT**) is another explicit loss function merely to accelerate and enhance the training.

* If you happen to be at the stage where ```patch-based refinement``` is working.
  - Simply search ```"GenBonePatch"``` and you are scurrying to the ```refinement``` part. 
  - Patches around each bone (20 in total) are generated, which are essentially in a pyramid-like style.
  - You easily get initial orientation from initial pose estimate.
  - For each bone, you start building a really simple conv + FC network to estimate the residual orientation of that bone.
  - 20 per-bone refinement subnetworks add up to a residual orientation vector, which is added to the initial orientation.
  - Now that you have the refined orientation vector, as well as root location from the initial 3D estimate, you achieve the refined 3D pose.
  - This refined 3D pose is our final output.
# Generate HDF5 File


**Note that you can also use other input format, for instance you may directly read files from SSD.**

Roughly we define the training set as the first **170k** images in training set, setting the remained images as validation set. The H5 files contain **3** fields.

* image_index: image id
* gt_joint_3d_global: **real-world** 3D coordinates
* bbx: bounding box provided by the challenge organizer
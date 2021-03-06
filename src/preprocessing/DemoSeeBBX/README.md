# Data Preprocessing


**Note that you may have to change the directory path accordingly. Easy stuff.**

## Overall Pipeline

``` shell
|-- Main Function Process()
`-- |-- For each sample 
    `-- |-- Save original bounding box provided by the author
        |   |-- ### line 895
    `-- |-- Save 3d joints annotation in real-world coordinate 
        |   |-- ### line 927
    `-- |-- Crop interested depth image patch 
        |   |-- Rough background subtraction
        |   |   |-- ### line 955 Deploy "GetRidOfBackground". Remove palpable pixels which are outliers.
        |   |-- Rough CoM (center of mass; 2.5D) calculation  
        |   |   |-- ### line 965 Deploy "calcBoxCenter()" w/ option = 0
        |   |-- Fine bounding box reset w/ CoM 
        |   |   |-- ### line 981 - 997 new box size: 500.0 / avgZ * 210.0 (smaller avgZ = closer hand = larger crop size)
        |   |-- Fine background subtraction
        |   |   |-- ### line 1002 Deploy "GetRidOfBackground". This time abandons pixel with depth outside the reasonable depth range [avgZ - 100, avgZ + 100]
        |   |-- Fine CoM recalculation
        |   |   |-- ### line 1007 Deploy "calcBoxCenter()" w/ option = 1; Direct averaging 2.5D 
        |   |-- Resize to a 256x256 patch of depth values normalized to [-1, 1]
        |   |   |-- ### line 1015 - 1026
    `-- |-- Finally save the cropped image patch to disk
        |   |   |-- ### line 1096 - 1098 The path "XXXXX_images_crop" is often used in prototxt configuration	    
```
		
## Functions
**GetRidOfBackground()**
``` shell
|-- GetRidOfBackground()
`-- |-- Extend the rectangle to square 
    |   |-- ### line 494 - 521
`-- |-- Copy cropped area
    |   |-- ### line 573 - 583 (to handMat)
`-- |-- (Optional) prune outliers whose depth values fall out of the cropped 3D bounding box
    |   |-- ### line 586 - 599 (z outside the [-100, 100])
`-- |-- Run otsu thresholding to single out the foreground
    |   |-- ### line 625 - 736	
``` 

	
**calcBoxCenter()**
``` shell	
|-- calcBoxCenter(): Acquire centroid of the cropped depth point cloud
`-- |-- Compute the average UVD (2.5D coordinate mean values) of the cropped depth image
    |   |-- ### line 137 - 168 mean(u, v, d); pixel coordinate + depth value 
`-- |-- If option == 0: take the depth of closest non-zero pixel to the cropped image center
        |   |-- Take the depth value of center point (which is approximately middle MCP)
        |   |   |-- ### line 175
        |   |   |   |   |-- If it equals 0.0 BFS to find nearest non-zero pixel starting from the center pixel 
        |   |   |   |   |   |-- ### line 178 - 203 If cannot find forcibly set the depth to 500.0 
`-- |-- Else: 2.5D -> 3D save to avgX, avgY, avgZ
        |   |-- ### line 171
``` 

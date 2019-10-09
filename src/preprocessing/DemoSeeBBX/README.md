# Data Preprocessing

## Overall Pipeline

``` shell
|-- Main Function Process()
`-- |-- For each sample 
    `-- |-- Save original bounding box provided by the author
        |   |-- ### line 895
    `-- |-- Save 3d joints annotation in real-world coordinate 
        |   |-- ### line 927
    `-- |-- Rough background subtraction
        |   |-- ### Deploy "GetRidOfBackground". Remove palpable pixels which are outliers.
    `-- |-- Rough CoM (center of mass; 2.5D) calculation  
        |   |-- ### Deploy "calcBoxCenter()" w/ option = 0
    `-- |-- Fine bounding box reset w/ CoM 
        |   |-- ### line 981 new box size: 500.0 / avgZ * 210.0 (smaller avgZ = closer hand = larger crop size)
	
	

       
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
		
		
		
		
		
		
		
		

``` shell	
`-- |-- src
    `-- |-- caffe
        |   |-- layers
        |   |   |-- DeepHumanModel
        |   |   |   |-- deep_human_model_argmax_2d_hm_layer.cpp 
        |   |   |   |-- ### This takes argmax operation on 2d heatmap 
``` 
		
		


       
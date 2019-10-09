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
	

       
```
		
## Functions
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
	
    `-- |-- save original bounding box provided by the author
        |   |-- ### line 895
    `-- |-- save 3d joints annotation in real-world coordinate 
        |   |-- ### line 927
		
`-- |-- src
    `-- |-- caffe
        |   |-- layers
        |   |   |-- DeepHumanModel
        |   |   |   |-- deep_human_model_argmax_2d_hm_layer.cpp 
        |   |   |   |-- ### This takes argmax operation on 2d heatmap 


		
|-- calcBoxCenter()
`-- |-- Compute the average UVD (2.5D coordinate mean values) of the cropped depth image
    |   |-- ### line 137 - 168 mean(u, v, d); pixel coordinate + depth value 
`-- |-- If option == 0
        |   |-- Take the 2.5D coordinate of center point (which is approximately middle MCP)
        |   |   |-- ### line 175
        |   |   |   |   |-- If it equals 0.0 BFS to find nearest non-zero pixel starting from the center pixel 
		
		


       
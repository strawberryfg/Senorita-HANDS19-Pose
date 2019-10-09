# Data Preprocessing

## Overall Pipeline

``` shell
|-- Main Function Process()
`-- |-- For each sample 
    `-- |-- Save original bounding box provided by the author
        |   |-- ### line 895
    `-- |-- Save 3d joints annotation in real-world coordinate 
        |   |-- ### line 927

       
```
		
## Functions
``` shell
|-- GetRidOfBackground()
`-- |-- Extend the rectangle to square 
    |   |-- ### line 494 - 521
`-- |-- Copy cropped area
    |   |-- ### line 573 - 583 (to handMat)
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
       
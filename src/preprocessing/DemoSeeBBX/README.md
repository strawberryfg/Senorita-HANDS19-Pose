# Data Preprocessing

## Pipeline

``` shell
|-- Main Function Process()
`-- |-- For each sample 
    `-- |-- save original bounding box provided by the author
        |   |-- ### line 895
    `-- |-- save 3d joints annotation in real-world coordinate 
        |   |-- ### line 927
		
		
		
|-- GetRidOfBackground()
`-- |-- For each sample 
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
       
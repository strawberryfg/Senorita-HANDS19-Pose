# Data Preprocessing

## Pipeline

``` shell
|-- Main Function Process()
`-- |-- For each sample 
    `-- |-- read & save original bounding box provided by the author 
        |   |-- deep_human_model_layers.hpp
        |   |   | ### This includes operations about 2d/3d heatmap /integral / augmentation / local <-> global transformation etc.
        |   |-- h36m.h
        |   |   | ### This includes definitions of joint / part / bone (h36m 32 joints / usable 16 joints / c2f 17 joints etc.)
        |   |-- operations.hpp 
        |   |   | ### This includes operations w.r.t scalar / vector / fetch file / output data.
`-- |-- src
    `-- |-- caffe
        |   |-- layers
        |   |   |-- DeepHumanModel
        |   |   |   |-- deep_human_model_argmax_2d_hm_layer.cpp 
        |   |   |   |-- ### This takes argmax operation on 2d heatmap 
        |   |   |   |-- deep_human_model_convert_2d_layer.cpp 
        |   |   |   |-- ### h36m provides full 32 joints, of which we only care 16 joints. Conversion from 16x2 <-> 32x2
        |   |   |   |-- deep_human_model_convert_3d_layer.cpp 
        |   |   |   |-- ### Conversion from 16x3 <-> 32x3
        |   |   |   |-- deep_human_model_convert_depth_layer.cpp 
        |   |   |   |-- ### Conversion from root-relative camera coordinate <-> [-1, 1] normalized depth
        |   |   |   |-- deep_human_model_gen_3d_heatmap_in_more_detail_v3_layer.cpp 
        |   |   |   |-- ### Generate groud truth for 3d heatmap. Closely follows c2f Torch code.
        |   |   |   |-- deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_layer.cpp 
        |   |   |   |-- ### Argmax operation on 3d heatmap
        |   |   |   |-- deep_human_model_h36m_gen_aug_3d_layer.cpp 
        |   |   |   |-- ### Generate augmented 3d ground truth according to augmented 2d gt and 3d gt
        |   |   |   |-- deep_human_model_h36m_gen_pred_mono_3d_layer.cpp 
        |   |   |   |-- ### 2.5D -> 3D camera frame coordinate
        |   |   |   |-- deep_human_model_integral_vector_layer.cpp 
        |   |   |   |-- ### \sum_{i=0}^{D-1} probability * position
        |   |   |   |-- deep_human_model_integral_x_layer.cpp 
        |   |   |   |-- ### Integral along X axis
        |   |   |   |-- deep_human_model_integral_y_layer.cpp 
        |   |   |   |-- ### Integral along Y axis
        |   |   |   |-- deep_human_model_integral_z_layer.cpp 
        |   |   |   |-- ### Integral along Z axis
        |   |   |   |-- deep_human_model_norm_3d_hm_layer.cpp 
        |   |   |   |-- ### Normalize 3D heatmap responses to make them sum up to 1.0
        |   |   |   |-- deep_human_model_normalization_response_v0_layer.cpp 
        |   |   |   |-- ### 2D heatmap normalization
        |   |   |   |-- deep_human_model_numerical_coordinate_regression_layer.cpp 
        |   |   |   |-- ### Integral over normalized 2D heatmap -> (x, y)
        |   |   |   |-- deep_human_model_output_heatmap_sep_channel_layer.cpp 
        |   |   |   |-- ### Output heatmap of different joints to different folders
        |   |   |   |-- deep_human_model_output_joint_on_skeleton_map_h36m_layer.cpp 
        |   |   |   |-- ### Plot predicted joints on raw image
        |   |   |   |-- deep_human_model_softmax_3d_hm_layer.cpp 
        |   |   |   |-- ### Softmax normalization on 3d heatmap
        |   |   |   |-- deep_human_model_softmax_hm_layer.cpp 
        |   |   |   |-- ### Softmax normalization on 2d heatmap
		
		
		
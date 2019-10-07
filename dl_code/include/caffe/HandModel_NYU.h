//num
#define JointNum_NYUNYU 31
#define ParamNum_NYUNYU 47
#define BoneNum_NYUNYU 30
#define Num_Of_Const_Matr_NYUNYU 30
#define num_of_bones_each_finger_NYUNYU 5
#define num_of_dof_each_mcp_NYUNYU 3
#define num_of_dof_each_finger_NYUNYU 4
//end of num



//matr operation
#define rot_x_NYUNYU 0
#define rot_y_NYUNYU 1
#define rot_z_NYUNYU 2
#define trans_x_NYUNYU 3
#define trans_y_NYUNYU 4
#define trans_z_NYUNYU 5
#define Const_Matr_NYUNYU 6
//end of matr operation

//DoF id
#define global_trans_x_id_NYUNYU 0
#define global_trans_y_id_NYUNYU 1
#define global_trans_z_id_NYUNYU 2

#define global_rot_x_id_NYUNYU 3
#define global_rot_y_id_NYUNYU 4
#define global_rot_z_id_NYUNYU 5

#define wrist_left_rot_x_id_NYUNYU 6
#define wrist_left_rot_y_id_NYUNYU 7
#define wrist_left_rot_z_id_NYUNYU 8

#define wrist_middle_rot_x_id_NYUNYU 9
#define wrist_middle_rot_y_id_NYUNYU 10
#define wrist_middle_rot_z_id_NYUNYU 11

#define thumb_mcp_rot_x_id_NYUNYU 12
#define thumb_mcp_rot_y_id_NYUNYU 13
#define thumb_mcp_rot_z_id_NYUNYU 14

#define thumb_pip_rot_y_id_NYUNYU 15
#define thumb_pip_rot_z_id_NYUNYU 16
#define thumb_dip_rot_z_id_NYUNYU 17
#define thumb_tip_rot_z_id_NYUNYU 18

#define finger_mcp_rot_x_start_id_NYUNYU 19
#define finger_mcp_rot_y_start_id_NYUNYU 20
#define finger_mcp_rot_z_start_id_NYUNYU 21

#define finger_base_rot_x_start_id_NYUNYU 31
#define finger_base_rot_z_start_id_NYUNYU 32

#define finger_pip_rot_x_start_id_NYUNYU 33
#define finger_dip_rot_x_start_id_NYUNYU 34
//end of DoF id

//const matrix id

#define wrist_left_id_in_const_NYUNYU 0
#define wrist_middle_id_in_const_NYUNYU 1
#define thumb_mcp_id_in_const_NYUNYU 2
#define thumb_pip_id_in_const_NYUNYU 3
#define thumb_dip_id_in_const_NYUNYU 4
#define thumb_tip_id_in_const_NYUNYU 5
#define finger_mcp_start_id_in_const_NYUNYU 6
#define finger_base_start_id_in_const_NYUNYU 10
#define finger_pip_first_start_id_in_const_NYUNYU 11
#define finger_pip_second_start_id_in_const_NYUNYU 12
#define finger_dip_start_id_in_const_NYUNYU 13
#define finger_tip_start_id_in_const_NYUNYU 14
//end of const matrix id

//bone id
#define bone_wrist_left_NYUNYU 24
#define bone_wrist_middle_NYUNYU 25
#define bone_thumb_mcp_NYUNYU 26
#define bone_thumb_pip_NYUNYU 27
#define bone_thumb_dip_NYUNYU 28
#define bone_thumb_tip_NYUNYU 29
#define bone_mcp_start_NYUNYU 20
#define bone_base_start_NYUNYU 4
#define bone_pip_first_start_NYUNYU 3
#define bone_pip_second_start_NYUNYU 2
#define bone_dip_start_NYUNYU 1
#define bone_tip_start_NYUNYU 0
//end of bone id

//keypoint (some are not real joint) id
#define palm_center_NYUNYU 24
#define wrist_left_NYUNYU 25
#define wrist_middle_NYUNYU 26

#define thumb_mcp_NYUNYU 27
#define thumb_pip_NYUNYU 28
#define thumb_dip_NYUNYU 29
#define thumb_tip_NYUNYU 30

#define finger_mcp_start_NYUNYU 20
#define finger_base_start_NYUNYU 4
#define finger_pip_first_start_NYUNYU 3
#define finger_pip_second_start_NYUNYU 2
#define finger_dip_start_NYUNYU 1
#define finger_tip_start_NYUNYU 0

//End of keypoint

// sequence of forward

//

#define pb push_back
#define mp std::make_pair



const int forward_seq_NYUNYU[31] = { 24, 25, 26, 27, 28, 29, 30, 20, 21, 22, 23, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 14, 13, 12, 11, 10, 19, 18, 17, 16, 15 };
const int prev_seq_NYUNYU[31] = { -1, 24, 24, 24, 27, 28, 29, 24, 24, 24, 24, 20, 4, 3, 2, 1, 21, 9, 8, 7, 6, 22, 14, 13, 12, 11, 23, 19, 18, 17, 16 };
enum {
	JointNumAll_RHD = 42, BoneNumAll_RHD = 40,
	JointNum_RHD = 21, BoneNum_RHD = 20,
	root_RHD = 0,
	ref_bone_RHD = 7
};

enum joint_left_right 
{
	//left: 0-20
	left_wrist_RHD, left_thumb_tip_RHD, left_thumb_dip_RHD, left_thumb_pip_RHD, left_thumb_mcp_RHD,
	left_index_tip_RHD, left_index_dip_RHD, left_index_pip_RHD, left_index_mcp_RHD,
	left_middle_tip_RHD, left_middle_dip_RHD, left_middle_pip_RHD, left_middle_mcp_RHD,
	left_ring_tip_RHD, left_ring_dip_RHD, left_ring_pip_RHD, left_ring_mcp_RHD,
	left_pinky_tip_RHD, left_pinky_dip_RHD, left_pinky_pip_RHD, left_pinky_mcp_RHD,

	//right: 21-41
	right_wrist_RHD, right_thumb_tip_RHD, right_thumb_dip_RHD, right_thumb_pip_RHD, right_thumb_mcp_RHD,
	right_index_tip_RHD, right_index_dip_RHD, right_index_pip_RHD, right_index_mcp_RHD,
	right_middle_tip_RHD, right_middle_dip_RHD, right_middle_pip_RHD, right_middle_mcp_RHD,
	right_ring_tip_RHD, right_ring_dip_RHD, right_ring_pip_RHD, right_ring_mcp_RHD,
	right_pinky_tip_RHD, right_pinky_dip_RHD, right_pinky_pip_RHD, right_pinky_mcp_RHD
};


enum joint_part
{
	//21 joints
	wrist_RHD, thumb_tip_RHD, thumb_dip_RHD, thumb_pip_RHD, thumb_mcp_RHD,
	index_tip_RHD, index_dip_RHD, index_pip_RHD, index_mcp_RHD,
	middle_tip_RHD, middle_dip_RHD, middle_pip_RHD, middle_mcp_RHD,
	ring_tip_RHD, ring_dip_RHD, ring_pip_RHD, ring_mcp_RHD,
	pinky_tip_RHD, pinky_dip_RHD, pinky_pip_RHD, pinky_mcp_RHD
};


enum bone_left_right // 40
{
	//left: 0-19
	bone_left_ttip_tdip_RHD, bone_left_tdip_tpip_RHD, bone_left_tpip_tmcp_RHD, bone_left_tmcp_wrist_RHD,  //thumb
	bone_left_itip_idip_RHD, bone_left_idip_ipip_RHD, bone_left_ipip_imcp_RHD, bone_left_imcp_wrist_RHD,  //index
	bone_left_mtip_mdip_RHD, bone_left_mdip_mpip_RHD, bone_left_mpip_mmcp_RHD, bone_left_mmcp_wrist_RHD,  //middle
	bone_left_rtip_rdip_RHD, bone_left_rdip_rpip_RHD, bone_left_rpip_rmcp_RHD, bone_left_rmcp_wrist_RHD,  //ring
	bone_left_ptip_pdip_RHD, bone_left_pdip_ppip_RHD, bone_left_ppip_pmcp_RHD, bone_left_pmcp_wrist_RHD,  //pinky(little)	

	//right: 20-39
	bone_right_ttip_tdip_RHD, bone_right_tdip_tpip_RHD, bone_right_tpip_tmcp_RHD, bone_right_tmcp_wrist_RHD,  //thumb
	bone_right_itip_idip_RHD, bone_right_idip_ipip_RHD, bone_right_ipip_imcp_RHD, bone_right_imcp_wrist_RHD,  //index
	bone_right_mtip_mdip_RHD, bone_right_mdip_mpip_RHD, bone_right_mpip_mmcp_RHD, bone_right_mmcp_wrist_RHD,  //middle
	bone_right_rtip_rdip_RHD, bone_right_rdip_rpip_RHD, bone_right_rpip_rmcp_RHD, bone_right_rmcp_wrist_RHD,  //ring
	bone_right_ptip_pdip_RHD, bone_right_pdip_ppip_RHD, bone_right_ppip_pmcp_RHD, bone_right_pmcp_wrist_RHD,  //pinky(little)	

};


enum bone_part // 20
{
	bone_ttip_tdip_RHD, bone_tdip_tpip_RHD, bone_tpip_tmcp_RHD, bone_tmcp_wrist_RHD,  //thumb
	bone_itip_idip_RHD, bone_idip_ipip_RHD, bone_ipip_imcp_RHD, bone_imcp_wrist_RHD,  //index
	bone_mtip_mdip_RHD, bone_mdip_mpip_RHD, bone_mpip_mmcp_RHD, bone_mmcp_wrist_RHD,  //middle
	bone_rtip_rdip_RHD, bone_rdip_rpip_RHD, bone_rpip_rmcp_RHD, bone_rmcp_wrist_RHD,  //ring
	bone_ptip_pdip_RHD, bone_pdip_ppip_RHD, bone_ppip_pmcp_RHD, bone_pmcp_wrist_RHD,  //pinky(little)	
};



const int bones_all_RHD[BoneNumAll_RHD][2] =
{
	//left: 0-19
	{ left_thumb_tip_RHD, left_thumb_dip_RHD }, { left_thumb_dip_RHD, left_thumb_pip_RHD }, { left_thumb_pip_RHD, left_thumb_mcp_RHD }, { left_thumb_mcp_RHD, left_wrist_RHD },
	{ left_index_tip_RHD, left_index_dip_RHD }, { left_index_dip_RHD, left_index_pip_RHD }, { left_index_pip_RHD, left_index_mcp_RHD }, { left_index_mcp_RHD, left_wrist_RHD },
	{ left_middle_tip_RHD, left_middle_dip_RHD }, { left_middle_dip_RHD, left_middle_pip_RHD }, { left_middle_pip_RHD, left_middle_mcp_RHD }, { left_middle_mcp_RHD, left_wrist_RHD },
	{ left_ring_tip_RHD, left_ring_dip_RHD }, { left_ring_dip_RHD, left_ring_pip_RHD }, { left_ring_pip_RHD, left_ring_mcp_RHD }, { left_ring_mcp_RHD, left_wrist_RHD },
	{ left_pinky_tip_RHD, left_pinky_dip_RHD }, { left_pinky_dip_RHD, left_pinky_pip_RHD }, { left_pinky_pip_RHD, left_pinky_mcp_RHD }, { left_pinky_mcp_RHD, left_wrist_RHD },

	//right: 20-39
	{ right_thumb_tip_RHD, right_thumb_dip_RHD }, { right_thumb_dip_RHD, right_thumb_pip_RHD }, { right_thumb_pip_RHD, right_thumb_mcp_RHD }, { right_thumb_mcp_RHD, right_wrist_RHD },
	{ right_index_tip_RHD, right_index_dip_RHD }, { right_index_dip_RHD, right_index_pip_RHD }, { right_index_pip_RHD, right_index_mcp_RHD }, { right_index_mcp_RHD, right_wrist_RHD },
	{ right_middle_tip_RHD, right_middle_dip_RHD }, { right_middle_dip_RHD, right_middle_pip_RHD }, { right_middle_pip_RHD, right_middle_mcp_RHD }, { right_middle_mcp_RHD, right_wrist_RHD },
	{ right_ring_tip_RHD, right_ring_dip_RHD }, { right_ring_dip_RHD, right_ring_pip_RHD }, { right_ring_pip_RHD, right_ring_mcp_RHD }, { right_ring_mcp_RHD, right_wrist_RHD },
	{ right_pinky_tip_RHD, right_pinky_dip_RHD }, { right_pinky_dip_RHD, right_pinky_pip_RHD }, { right_pinky_pip_RHD, right_pinky_mcp_RHD }, { right_pinky_mcp_RHD, right_wrist_RHD }
};


const int bones_RHD[BoneNum_RHD][2] =
{
	{ thumb_tip_RHD, thumb_dip_RHD }, { thumb_dip_RHD, thumb_pip_RHD }, { thumb_pip_RHD, thumb_mcp_RHD }, { thumb_mcp_RHD, wrist_RHD },
	{ index_tip_RHD, index_dip_RHD }, { index_dip_RHD, index_pip_RHD }, { index_pip_RHD, index_mcp_RHD }, { index_mcp_RHD, wrist_RHD },
	{ middle_tip_RHD, middle_dip_RHD }, { middle_dip_RHD, middle_pip_RHD }, { middle_pip_RHD, middle_mcp_RHD }, { middle_mcp_RHD, wrist_RHD },
	{ ring_tip_RHD, ring_dip_RHD }, { ring_dip_RHD, ring_pip_RHD }, { ring_pip_RHD, ring_mcp_RHD }, { ring_mcp_RHD, wrist_RHD },
	{ pinky_tip_RHD, pinky_dip_RHD }, { pinky_dip_RHD, pinky_pip_RHD }, { pinky_pip_RHD, pinky_mcp_RHD }, { pinky_mcp_RHD, wrist_RHD },
};


const int color_pred_joint_all_RHD[JointNumAll_RHD][3]=
{
	//left: 0-20
	{0, 0, 0},       //left wrist
	{255, 0, 255},   //left thumb tip
	{255, 0, 255},   //left thumb dip
	{255, 0, 255},   //left thumb pip
	{255, 0, 255},   //left thumb mcp
	{255, 0, 0},     //left index tip
	{255, 0, 0},     //left index dip
	{255, 0, 0},     //left index pip
	{255, 0, 0},     //left index mcp
	{0, 255, 0},     //left middle tip
	{0, 255, 0},     //left middle dip
	{0, 255, 0},     //left middle pip
	{0, 255, 0},     //left middle mcp
	{0, 255, 255},   //left ring tip
	{0, 255, 255},   //left ring dip
	{0, 255, 255},   //left ring pip
	{0, 255, 255},   //left ring mcp
	{0, 0, 255},     //left pinky tip
	{0, 0, 255},     //left pinky dip
	{0, 0, 255},     //left pinky pip
	{0, 0, 255},     //left pinky mcp

	//right: 21-41
	{ 0, 0, 0 },     //right wrist
	{ 255, 0, 255 }, //right thumb tip
	{ 255, 0, 255 }, //right thumb dip
	{ 255, 0, 255 }, //right thumb pip
	{ 255, 0, 255 }, //right thumb mcp
	{ 255, 0, 0 },   //right index tip
	{ 255, 0, 0 },   //right index dip
	{ 255, 0, 0 },   //right index pip
	{ 255, 0, 0 },   //right index mcp
	{ 0, 255, 0 },   //right middle tip
	{ 0, 255, 0 },   //right middle dip
	{ 0, 255, 0 },   //right middle pip
	{ 0, 255, 0 },   //right middle mcp
	{ 0, 255, 255 }, //right ring tip
	{ 0, 255, 255 }, //right ring dip
	{ 0, 255, 255 }, //right ring pip
	{ 0, 255, 255 }, //right ring mcp
	{ 0, 0, 255 },   //right pinky tip
	{ 0, 0, 255 },   //right pinky dip
	{ 0, 0, 255 },   //right pinky pip
	{ 0, 0, 255 },   //right pinky mcp
};



const int color_pred_joint_RHD[JointNum_RHD][3] =
{
	{ 0, 0, 0 },       //wrist
	{ 255, 0, 255 },   //thumb tip
	{ 255, 0, 255 },   //thumb dip
	{ 255, 0, 255 },   //thumb pip
	{ 255, 0, 255 },   //thumb mcp
	{ 255, 0, 0 },     //index tip
	{ 255, 0, 0 },     //index dip
	{ 255, 0, 0 },     //index pip
	{ 255, 0, 0 },     //index mcp
	{ 0, 255, 0 },     //middle tip
	{ 0, 255, 0 },     //middle dip
	{ 0, 255, 0 },     //middle pip
	{ 0, 255, 0 },     //middle mcp
	{ 0, 255, 255 },   //ring tip
	{ 0, 255, 255 },   //ring dip
	{ 0, 255, 255 },   //ring pip
	{ 0, 255, 255 },   //ring mcp
	{ 0, 0, 255 },     //pinky tip
	{ 0, 0, 255 },     //pinky dip
	{ 0, 0, 255 },     //pinky pip
	{ 0, 0, 255 },     //pinky mcp

};


const int color_pred_bone_all_RHD[BoneNumAll_RHD][3] = {
	//left: 0-19
	{ 255, 0, 255 },  //ttip -> tdip
	{ 255, 0, 255 },  //tdip -> tpip
	{ 255, 0, 255 },  //tpip -> tmcp
	{ 255, 0, 255 },  //tmcp -> wrist

	{ 255, 0, 0 },  //itip -> idip
	{ 255, 0, 0 },   //idip -> ipip
	{ 255, 0, 0 },  //ipip -> imcp
	{ 255, 0, 0 },  //imcp -> wrist

	{ 0, 255, 0 },     //mtip -> mdip
	{ 0, 255, 0 },     //mdip -> mpip
	{ 0, 255, 0 },     //mpip -> mmcp
	{ 0, 255, 0 },     //mmcp -> wrist

	{ 0, 255, 255 },   //rtip -> rdip 
	{ 0, 255, 255 },   //rdip -> rpip
	{ 0, 255, 255 },   //rpip -> rmcp
	{ 0, 255, 255 },   //rmcp -> wrist

	{ 0, 0, 255 },     //ptip -> pdip 
	{ 0, 0, 255 },     //pdip -> ppip
	{ 0, 0, 255 },     //ppip -> pmcp
	{ 0, 0, 255 },     //pmcp -> wrist

	//right: 20-39
	{ 255, 0, 255 },  //ttip -> tdip
	{ 255, 0, 255 },  //tdip -> tpip
	{ 255, 0, 255 },  //tpip -> tmcp
	{ 255, 0, 255 },  //tmcp -> wrist

	{ 255, 0, 0 },  //itip -> idip
	{ 255, 0, 0 },   //idip -> ipip
	{ 255, 0, 0 },  //ipip -> imcp
	{ 255, 0, 0 },  //imcp -> wrist

	{ 0, 255, 0 },     //mtip -> mdip
	{ 0, 255, 0 },     //mdip -> mpip
	{ 0, 255, 0 },     //mpip -> mmcp
	{ 0, 255, 0 },     //mmcp -> wrist

	{ 0, 255, 255 },   //rtip -> rdip 
	{ 0, 255, 255 },   //rdip -> rpip
	{ 0, 255, 255 },   //rpip -> rmcp
	{ 0, 255, 255 },   //rmcp -> wrist

	{ 0, 0, 255 },     //ptip -> pdip 
	{ 0, 0, 255 },     //pdip -> ppip
	{ 0, 0, 255 },     //ppip -> pmcp
	{ 0, 0, 255 },     //pmcp -> wrist
};


const int color_pred_bone_RHD[BoneNum_RHD][3] = {
	//left: 0-19
	{ 255, 0, 255 },  //ttip -> tdip
	{ 255, 0, 255 },  //tdip -> tpip
	{ 255, 0, 255 },  //tpip -> tmcp
	{ 255, 0, 255 },  //tmcp -> wrist

	{ 255, 0, 0 },  //itip -> idip
	{ 255, 0, 0 },   //idip -> ipip
	{ 255, 0, 0 },  //ipip -> imcp
	{ 255, 0, 0 },  //imcp -> wrist

	{ 0, 255, 0 },     //mtip -> mdip
	{ 0, 255, 0 },     //mdip -> mpip
	{ 0, 255, 0 },     //mpip -> mmcp
	{ 0, 255, 0 },     //mmcp -> wrist

	{ 0, 255, 255 },   //rtip -> rdip 
	{ 0, 255, 255 },   //rdip -> rpip
	{ 0, 255, 255 },   //rpip -> rmcp
	{ 0, 255, 255 },   //rmcp -> wrist

	{ 0, 0, 255 },     //ptip -> pdip 
	{ 0, 0, 255 },     //pdip -> ppip
	{ 0, 0, 255 },     //ppip -> pmcp
	{ 0, 0, 255 },     //pmcp -> wrist

};

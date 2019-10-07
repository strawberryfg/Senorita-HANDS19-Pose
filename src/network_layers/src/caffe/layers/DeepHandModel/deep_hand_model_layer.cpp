#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"

namespace caffe {
	//set up constant translation matrices
	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::SetupConstantMatrices() {
		//T thumb
		ConstMatr[tmcp] = numeric::Matrix4d(trans_x, BoneLen[bone_tmcp_wrist], false);
		ConstMatr[tpip] = numeric::Matrix4d(trans_x, BoneLen[bone_tpip_tmcp], false);
		ConstMatr[tdip] = numeric::Matrix4d(trans_x, BoneLen[bone_tdip_tpip], false);
		ConstMatr[ttip] = numeric::Matrix4d(trans_x, BoneLen[bone_ttip_tdip], false);

		//I index
		ConstMatr[imcp] = numeric::Matrix4d(trans_y, BoneLen[bone_imcp_wrist], false);
		ConstMatr[ipip] = numeric::Matrix4d(trans_y, BoneLen[bone_ipip_imcp], false);
		ConstMatr[idip] = numeric::Matrix4d(trans_y, BoneLen[bone_idip_ipip], false);
		ConstMatr[itip] = numeric::Matrix4d(trans_y, BoneLen[bone_itip_idip], false);

		//M middle
		ConstMatr[mmcp] = numeric::Matrix4d(trans_y, BoneLen[bone_mmcp_wrist], false);
		ConstMatr[mpip] = numeric::Matrix4d(trans_y, BoneLen[bone_mpip_mmcp], false);
		ConstMatr[mdip] = numeric::Matrix4d(trans_y, BoneLen[bone_mdip_mpip], false);
		ConstMatr[mtip] = numeric::Matrix4d(trans_y, BoneLen[bone_mtip_mdip], false);

		//R ring
		ConstMatr[rmcp] = numeric::Matrix4d(trans_y, BoneLen[bone_rmcp_wrist], false);
		ConstMatr[rpip] = numeric::Matrix4d(trans_y, BoneLen[bone_rpip_rmcp], false);
		ConstMatr[rdip] = numeric::Matrix4d(trans_y, BoneLen[bone_rdip_rpip], false);
		ConstMatr[rtip] = numeric::Matrix4d(trans_y, BoneLen[bone_rtip_rdip], false);

		//P pinky
		ConstMatr[pmcp] = numeric::Matrix4d(trans_y, BoneLen[bone_pmcp_wrist], false);
		ConstMatr[ppip] = numeric::Matrix4d(trans_y, BoneLen[bone_ppip_pmcp], false);
		ConstMatr[pdip] = numeric::Matrix4d(trans_y, BoneLen[bone_pdip_ppip], false);
		ConstMatr[ptip] = numeric::Matrix4d(trans_y, BoneLen[bone_ptip_pdip], false);


		//BONE LENGTH GRADIENT MATRIX FOR EXAMPLE
		//[0 0 0 0]
		//[0 0 0 1]
		//[0 0 0 0]
		//[0 0 0 0]
		//T thumb
		ConstMatr_grad[tmcp] = numeric::Matrix4d(trans_x, BoneLen[bone_tmcp_wrist], true);
		ConstMatr_grad[tpip] = numeric::Matrix4d(trans_x, BoneLen[bone_tpip_tmcp], true);
		ConstMatr_grad[tdip] = numeric::Matrix4d(trans_x, BoneLen[bone_tdip_tpip], true);
		ConstMatr_grad[ttip] = numeric::Matrix4d(trans_x, BoneLen[bone_ttip_tdip], true);

		//I index
		ConstMatr_grad[imcp] = numeric::Matrix4d(trans_y, BoneLen[bone_imcp_wrist], true);
		ConstMatr_grad[ipip] = numeric::Matrix4d(trans_y, BoneLen[bone_ipip_imcp], true);
		ConstMatr_grad[idip] = numeric::Matrix4d(trans_y, BoneLen[bone_idip_ipip], true);
		ConstMatr_grad[itip] = numeric::Matrix4d(trans_y, BoneLen[bone_itip_idip], true);

		//M middle
		ConstMatr_grad[mmcp] = numeric::Matrix4d(trans_y, BoneLen[bone_mmcp_wrist], true);
		ConstMatr_grad[mpip] = numeric::Matrix4d(trans_y, BoneLen[bone_mpip_mmcp], true);
		ConstMatr_grad[mdip] = numeric::Matrix4d(trans_y, BoneLen[bone_mdip_mpip], true);
		ConstMatr_grad[mtip] = numeric::Matrix4d(trans_y, BoneLen[bone_mtip_mdip], true);

		//R ring
		ConstMatr_grad[rmcp] = numeric::Matrix4d(trans_y, BoneLen[bone_rmcp_wrist], true);
		ConstMatr_grad[rpip] = numeric::Matrix4d(trans_y, BoneLen[bone_rpip_rmcp], true);
		ConstMatr_grad[rdip] = numeric::Matrix4d(trans_y, BoneLen[bone_rdip_rpip], true);
		ConstMatr_grad[rtip] = numeric::Matrix4d(trans_y, BoneLen[bone_rtip_rdip], true);

		//P pinky
		ConstMatr_grad[pmcp] = numeric::Matrix4d(trans_y, BoneLen[bone_pmcp_wrist], true);
		ConstMatr_grad[ppip] = numeric::Matrix4d(trans_y, BoneLen[bone_ppip_pmcp], true);
		ConstMatr_grad[pdip] = numeric::Matrix4d(trans_y, BoneLen[bone_pdip_ppip], true);
		ConstMatr_grad[ptip] = numeric::Matrix4d(trans_y, BoneLen[bone_ptip_pdip], true);
	}

	//set up forward kinematics rotation & translation transformations
	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::SetupTransformation() {
		for (int i = 0; i < JointNum; i++) HomoMatr[i].clear();
		HomoMatr[wrist].pb(mp(trans_x, global_trans_x));
		HomoMatr[wrist].pb(mp(trans_y, global_trans_y));
		HomoMatr[wrist].pb(mp(trans_z, global_trans_z));

		HomoMatr[wrist].pb(mp(rot_z, global_rot_z));
		HomoMatr[wrist].pb(mp(rot_x, global_rot_x));
		HomoMatr[wrist].pb(mp(rot_y, global_rot_y));

		//T thumb
		//tmcp
		for (int i = 0; i < HomoMatr[wrist].size(); i++)
			HomoMatr[tmcp].pb(HomoMatr[wrist][i]);
		HomoMatr[tmcp].pb(mp(rot_z, thumb_mcp_const_rot_z));
		HomoMatr[tmcp].pb(mp(rot_x, thumb_mcp_const_rot_x));
		HomoMatr[tmcp].pb(mp(rot_y, thumb_mcp_const_rot_y));
		HomoMatr[tmcp].pb(mp(Const_Matr, tmcp));
		
		//tpip
		for (int i = 0; i < HomoMatr[tmcp].size(); i++) HomoMatr[tpip].pb(HomoMatr[tmcp][i]);
		HomoMatr[tpip].pb(mp(rot_z, thumb_mcp_rot_z));
		HomoMatr[tpip].pb(mp(rot_y, thumb_mcp_rot_y));
		HomoMatr[tpip].pb(mp(Const_Matr, tpip));
		
		//tdip
		for (int i = 0; i < HomoMatr[tpip].size(); i++)
			HomoMatr[tdip].pb(HomoMatr[tpip][i]);
		HomoMatr[tdip].pb(mp(rot_y, thumb_pip_rot_y));
		HomoMatr[tdip].pb(mp(Const_Matr, tdip));
		
		//ttip
		for (int i = 0; i < HomoMatr[tdip].size(); i++)
			HomoMatr[ttip].pb(HomoMatr[tdip][i]);
		HomoMatr[ttip].pb(mp(rot_y, thumb_dip_rot_y));
		HomoMatr[ttip].pb(mp(Const_Matr, ttip));
		
		//I index
		//imcp
		for (int i = 0; i < HomoMatr[wrist].size(); i++)
			HomoMatr[imcp].pb(HomoMatr[wrist][i]);
		HomoMatr[imcp].pb(mp(rot_z, index_mcp_const_rot_z));
		HomoMatr[imcp].pb(mp(rot_x, index_mcp_const_rot_x));
		HomoMatr[imcp].pb(mp(rot_y, index_mcp_const_rot_y));
		HomoMatr[imcp].pb(mp(Const_Matr, imcp));
		
		//ipip
		for (int i = 0; i < HomoMatr[imcp].size(); i++)
			HomoMatr[ipip].pb(HomoMatr[imcp][i]);
		HomoMatr[ipip].pb(mp(rot_z, index_mcp_rot_z));
		HomoMatr[ipip].pb(mp(rot_x, index_mcp_rot_x));
		HomoMatr[ipip].pb(mp(Const_Matr, ipip));
		
		//idip
		for (int i = 0; i < HomoMatr[ipip].size(); i++)
			HomoMatr[idip].pb(HomoMatr[ipip][i]);
		HomoMatr[idip].pb(mp(rot_x, index_pip_rot_x));
		HomoMatr[idip].pb(mp(Const_Matr, idip));
		
		//itip
		for (int i = 0; i < HomoMatr[idip].size(); i++)
			HomoMatr[itip].pb(HomoMatr[idip][i]);
		HomoMatr[itip].pb(mp(rot_x, index_dip_rot_x));
		HomoMatr[itip].pb(mp(Const_Matr, itip));
		
		//M middle
		//mmcp
		for (int i = 0; i < HomoMatr[wrist].size(); i++)
			HomoMatr[mmcp].pb(HomoMatr[wrist][i]);
		HomoMatr[mmcp].pb(mp(rot_z, middle_mcp_const_rot_z));
		HomoMatr[mmcp].pb(mp(rot_x, middle_mcp_const_rot_x));
		HomoMatr[mmcp].pb(mp(rot_y, middle_mcp_const_rot_y));
		HomoMatr[mmcp].pb(mp(Const_Matr, mmcp));
		
		//mpip
		for (int i = 0; i < HomoMatr[mmcp].size(); i++)
			HomoMatr[mpip].pb(HomoMatr[mmcp][i]);
		HomoMatr[mpip].pb(mp(rot_z, middle_mcp_rot_z));
		HomoMatr[mpip].pb(mp(rot_x, middle_mcp_rot_x));
		HomoMatr[mpip].pb(mp(Const_Matr, mpip));
		
		//mdip
		for (int i = 0; i < HomoMatr[mpip].size(); i++)
			HomoMatr[mdip].pb(HomoMatr[mpip][i]);
		HomoMatr[mdip].pb(mp(rot_x, middle_pip_rot_x));
		HomoMatr[mdip].pb(mp(Const_Matr, mdip));
		
		//mtip
		for (int i = 0; i < HomoMatr[mdip].size(); i++)
			HomoMatr[mtip].pb(HomoMatr[mdip][i]);
		HomoMatr[mtip].pb(mp(rot_x, middle_dip_rot_x));
		HomoMatr[mtip].pb(mp(Const_Matr, mtip));
		
		//R ring
		//rmcp
		for (int i = 0; i < HomoMatr[wrist].size(); i++)
			HomoMatr[rmcp].pb(HomoMatr[wrist][i]);
		HomoMatr[rmcp].pb(mp(rot_z, ring_mcp_const_rot_z));
		HomoMatr[rmcp].pb(mp(rot_x, ring_mcp_const_rot_x));
		HomoMatr[rmcp].pb(mp(rot_y, ring_mcp_const_rot_y));
		HomoMatr[rmcp].pb(mp(Const_Matr, rmcp));
		
		//rpip
		for (int i = 0; i < HomoMatr[rmcp].size(); i++)
			HomoMatr[rpip].pb(HomoMatr[rmcp][i]);
		HomoMatr[rpip].pb(mp(rot_z, ring_mcp_rot_z));
		HomoMatr[rpip].pb(mp(rot_x, ring_mcp_rot_x));
		HomoMatr[rpip].pb(mp(Const_Matr, rpip));
		
		//rdip
		for (int i = 0; i < HomoMatr[rpip].size(); i++)
			HomoMatr[rdip].pb(HomoMatr[rpip][i]);
		HomoMatr[rdip].pb(mp(rot_x, ring_pip_rot_x));
		HomoMatr[rdip].pb(mp(Const_Matr, rdip));
		
		//rtip
		for (int i = 0; i < HomoMatr[rdip].size(); i++)
			HomoMatr[rtip].pb(HomoMatr[rdip][i]);
		HomoMatr[rtip].pb(mp(rot_x, ring_dip_rot_x));
		HomoMatr[rtip].pb(mp(Const_Matr, rtip));
		
		//P pinky
		//pmcp
		for (int i = 0; i < HomoMatr[wrist].size(); i++)
			HomoMatr[pmcp].pb(HomoMatr[wrist][i]);
		HomoMatr[pmcp].pb(mp(rot_z, pinky_mcp_const_rot_z));
		HomoMatr[pmcp].pb(mp(rot_x, pinky_mcp_const_rot_x));
		HomoMatr[pmcp].pb(mp(rot_y, pinky_mcp_const_rot_y));
		HomoMatr[pmcp].pb(mp(Const_Matr, pmcp));
		
		//ppip
		for (int i = 0; i < HomoMatr[pmcp].size(); i++)
			HomoMatr[ppip].pb(HomoMatr[pmcp][i]);
		HomoMatr[ppip].pb(mp(rot_z, pinky_mcp_rot_z));
		HomoMatr[ppip].pb(mp(rot_x, pinky_mcp_rot_x));
		HomoMatr[ppip].pb(mp(Const_Matr, ppip));
		
		//pdip
		for (int i = 0; i < HomoMatr[ppip].size(); i++)
			HomoMatr[pdip].pb(HomoMatr[ppip][i]);
		HomoMatr[pdip].pb(mp(rot_x, pinky_pip_rot_x));
		HomoMatr[pdip].pb(mp(Const_Matr, pdip));
		
		//ptip
		for (int i = 0; i < HomoMatr[pdip].size(); i++)
			HomoMatr[ptip].pb(HomoMatr[pdip][i]);
		HomoMatr[ptip].pb(mp(rot_x, pinky_dip_rot_x));
		HomoMatr[ptip].pb(mp(Const_Matr, ptip));
	}

	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int n;
		FILE *fin = fopen("../configuration/DofLimitId.in", "r");
		fscanf(fin, "%d", &n);
		for (int i = 0; i < ParamNum + 5; i++) isFixed[i] = false;
		for (int i = 0; i < n; i++) { int id; fscanf(fin, "%d", &id); isFixed[id] = true; }
		fclose(fin);

		//load initial bone length(fixed number)
		fin = fopen("../configuration/BoneLength.in", "r");
		for (int i = 0; i < BoneNum; i++) {
			fscanf(fin, "%*s %lf", &saveBoneLen[i]);
		}
		fclose(fin);

		//set initial parameters
		fin = fopen("../configuration/InitParam.in", "r");
		double t_param[ParamNum];
		for (int i = 0; i < ParamNum; i++) fscanf(fin, "%lf", &t_param[i]);
		for (int i = 0; i < MaxBatch; i++) {
			for (int j = 0; j < ParamNum; j++) InitParam[i][j] = t_param[j];
			for (int j = 0; j < 5; j++) InitParam[i][j + ParamNum] = 0.0;
		}
		SetupConstantMatrices();
		SetupTransformation();

		use_training_bone_len_stats_ = this->layer_param_.deep_hand_model_layer_param().use_training_bone_len_stats();
		//Whether to use constant bone length from pred bone len rather than learnable scale + base_scale (=1.0)
		use_constant_scale_ = this->layer_param_.deep_hand_model_layer_param().use_constant_scale();
	}

	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = JointNum * 3;
		top[0]->Reshape(top_shape);
	}

	//get matrix w & wo gradient (forward & backward)
	template <typename Dtype>
	numeric::Matrix4d DeepHandModelLayer<Dtype>::GetMatrix(int image_id, int Bid, matrix_operation opt, int id, bool is_gradient, const Dtype *bottom_data) {
		return opt == Const_Matr ? (use_constant_scale_ ? ConstMatr[id] : (is_gradient ? ConstMatr_grad[id] : ConstMatr[id])) : Matrix4d(opt, isFixed[id] ? InitParam[image_id][id] : bottom_data[Bid + id] + InitParam[image_id][id], is_gradient);
	}

	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::Forward(int image_id, Matrix4d mat, int i, int Bid, int prev_size, const Dtype *bottom_data) {
		for (int r = prev_size; r < HomoMatr[i].size(); r++)	mat = mat * GetMatrix(image_id, Bid, HomoMatr[i][r].first, HomoMatr[i][r].second, false, bottom_data);
		PrevMatr[i] = mat;
		temp_Joint[i] = PrevMatr[i] * Vector4d(0.0, 0.0, 0.0, 1.0);
	}

	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bone_3d_data; //joint 3d local [-1, 1] -> Joint2Bone
		const Dtype* bone_scale_data; //scale for each finger bone
		int id = 1;
		if (!use_training_bone_len_stats_) //NOT READ FROM FILE BEFOREHAND
		{
			bone_3d_data = bottom[id++]->cpu_data();
		}

		if (!use_constant_scale_) //NOT FIXED BONE LENGTH SCALE 
		{
			bone_scale_data = bottom[id++]->cpu_data();
		}

		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];

		for (int t = 0; t < batSize; t++) 
		{
			int Bid = t * ParamNum;
			int Tid = t * JointNum * 3;
			//Reset bone length according to the calculated bone length of current 3d pose prediction (estimation)
			int Boneid = t * BoneNum * 3;
			int Sid = t * 5; //5 scales

			//USE SAVED BACKUP BONELEN
			for (int i = 0; i < BoneNum; i++) BoneLen[i] = saveBoneLen[i];

			if (!use_training_bone_len_stats_)
			{
				for (int i = 0; i < BoneNum; i++)
				{
					//CALC L2 NORM OF THIS BONE
					BoneLen[i] = 0.0;
					for (int j = 0; j < 3; j++)
					{
						BoneLen[i] += pow(bone_3d_data[Boneid + i * 3 + j], 2);
					}
					BoneLen[i] = sqrt(BoneLen[i]);
				}
			}

			//base_scale 1.0 + each_finger_bone_scale
			if (!use_constant_scale_)
			{
				for (int i = 0; i < 5; i++) //T I M R P
				{
					for (int j = 0; j < 4; j++)
					{
						BoneLen[i * 4 + j] *= (1.0 + bone_scale_data[Sid + i]); //base_scale = 1.0
					}
				}
			}

			//SET UP CONSTANT MATRIX AGAIN
			SetupConstantMatrices();

			for (int i = 0; i < JointNum; i++) 
			{
				int id = forward_seq[i];
				Matrix4d mat;
				if (prev_seq[i] != -1) mat = PrevMatr[prev_seq[i]];
				Forward(t, mat, id, Bid, prev_seq[i] == -1 ? 0 : HomoMatr[prev_seq[i]].size(), bottom_data);
			}
			for (int i = 0; i < JointNum; i++) {
				for (int j = 0; j < 3; j++) top_data[Tid + i * 3 + j] = temp_Joint[i][j];
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::Backward(int image_id, int Bid, int i, const Dtype *bottom_data) {
		std::vector<std::pair<matrix_operation, int> > mat = HomoMatr[i];
		Matrix4d m_left[ParamNum * 3];
		Vector4d v_right[ParamNum * 3];
		v_right[mat.size() - 1] = Vector4d(0.0, 0.0, 0.0, 1.0);
		for (int r = mat.size() - 2; r >= 0; r--) v_right[r] = GetMatrix(image_id, Bid, mat[r + 1].first, mat[r + 1].second, false, bottom_data) * v_right[r + 1];
		m_left[0] = Matrix4d(); //Identity matrix
		for (int r = 1; r < mat.size(); r++) m_left[r] = m_left[r - 1] * GetMatrix(image_id, Bid, mat[r - 1].first, mat[r - 1].second, false, bottom_data);
		for (int r = 0; r < mat.size(); r++)
		{
			//IF W.R.T. FK ROTATION ANGLE PARAMS
			if (mat[r].first != Const_Matr) Jacobian[i][mat[r].second] = isFixed[mat[r].second] ? Vector4d(0.0, 0.0, 0.0, 1.0) : m_left[r] * GetMatrix(image_id, Bid, mat[r].first, mat[r].second, true, bottom_data) * v_right[r];
			else if (!use_constant_scale_) Jacobian[i][joint_belongs_to_TIMRP[mat[r].second] + ParamNum] = m_left[r] * GetMatrix(image_id, Bid, mat[r].first, mat[r].second, true, bottom_data) * v_right[r];
			// == Const_Matr (trans_x, trans_y, trans_z) && !use_constant_scale_
			//MUST HAVE GRADIENT SO IGNORE isFixed
		}
	}
	
	//LEARNABLE BONE LENGTH SCALE mat[r].second = ParamNum + k
	//where k is 0 - 4 (for 5 scales; on for each bone)

	//[ d Loss / d Joint.x * d Joint.x / d Scale_param
	//  d Loss / d Joint.y * d Joint.y / d Scale_param
	//  d Loss / d Joint.z * d Joint.z / d Scale_param
	//  0 ]
	//v[(opt - trans_x) * 4 + 3] = (!is_gradient) ? value : value_type(1);


	template <typename Dtype>
	void DeepHandModelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* bone_3d_data; //joint 3d local [-1, 1] -> Joint2Bone NO GRADIENT (CONSTANT)
			const Dtype* bone_scale_data; //scale for each finger bone
			Dtype* bone_scale_diff;
			int id = 1;
			if (!use_training_bone_len_stats_) //NOT READ FROM FILE BEFOREHAND
			{
				bone_3d_data = bottom[id++]->cpu_data();
			}

			if (!use_constant_scale_) //NOT FIXED BONE LENGTH SCALE 
			{
				bone_scale_data = bottom[id]->cpu_data();
				bone_scale_diff = bottom[id++]->mutable_cpu_diff();
			}
			

			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const int batSize = (bottom[0]->shape())[0];
			for (int t = 0; t < batSize; t++) {
				int Bid = t * ParamNum;

				int Boneid = t * BoneNum * 3;
				int Sid = t * 5; //5 scales

				//USE SAVED BACKUP BONELEN
				for (int i = 0; i < BoneNum; i++) BoneLen[i] = saveBoneLen[i];

				if (!use_training_bone_len_stats_)
				{
					//use bone len calculated from predicted 3d
					for (int i = 0; i < BoneNum; i++)
					{
						//CALC L2 NORM OF THIS BONE
						BoneLen[i] = 0.0;
						for (int j = 0; j < 3; j++)
						{
							BoneLen[i] += pow(bone_3d_data[Boneid + i * 3 + j], 2);
						}
						BoneLen[i] = sqrt(BoneLen[i]);
					}
				}

				//base_scale 1.0 + each_finger_bone_scale
				if (!use_constant_scale_)
				{
					for (int i = 0; i < 5; i++) //T I M R P
					{
						for (int j = 0; j < 4; j++)
						{
							BoneLen[i * 4 + j] *= (1.0 + bone_scale_data[Sid + i]); //base_scale = 1.0
						}
					}
				}

				//SET UP CONSTANT MATRIX AGAIN
				SetupConstantMatrices();

				for (int i = 0; i < JointNum; i++) 
				{
					for (int j = 0; j < (use_constant_scale_ ? ParamNum: ParamNum + 5); j++) 
					{
						//d Joint.x (y, z) / d Param (Scale_param)
						Jacobian[i][j].x[0] = Jacobian[i][j].x[1] = Jacobian[i][j].x[2] = Jacobian[i][j].x[3] = 0.0; //crucial
					}
					Backward(t, Bid, i, bottom_data);
				}
				//BACK PROP FORWARD KINEMATICS MODEL PARAM
				for (int j = 0; j < ParamNum; j++) 
				{
					bottom_diff[Bid + j] = 0;
					for (int i = 0; i < JointNum; i++) 
					{
						int Tid = t * JointNum * 3 + i * 3;
						for (int k = 0; k < 3; k++) bottom_diff[Bid + j] += Jacobian[i][j][k] * top_diff[Tid + k];
					}
				}

				if (!use_constant_scale_)
				{
					//5 scales
					for (int j = 0; j < 5; j++)
					{
						bone_scale_diff[Sid + j] = 0.0;
						//d Joint.x / d Scale_param * d Scale_param / d cnn_fc_output_scale_correction (w.r.t 1.0 base_scale)
						for (int i = 0; i < JointNum; i++)
						{
							int Tid = t * JointNum * 3 + i * 3;
							//Attached to the end of all FK model params 1.0 + cnn_fc_output_scale_correction
							for (int k = 0; k < 3; k++) bone_scale_diff[Sid + j] += Jacobian[i][j + ParamNum][k] * top_diff[Tid + k];
						}
					}
				}
			}
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelLayer);
	REGISTER_LAYER_CLASS(DeepHandModel);

}
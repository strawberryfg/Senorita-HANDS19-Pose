
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

#define PI 3.14159265359
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGen3DSegMapPerChannelLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_dims_ = this->layer_param_.gen_3d_skeleton_map_param().depth_dims();
		map_size_ = this->layer_param_.gen_3d_skeleton_map_param().map_size();



		gamma_ = this->layer_param_.gen_3d_skeleton_map_param().gamma();

		//whether to back propogate the gradient flow to the level of joint coordinates
		perform_backprop_ = this->layer_param_.gen_3d_skeleton_map_param().perform_backprop();

		depth_size_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().depth_size();

		focusx_ = this->layer_param_.pinhole_camera_origin_param().focusx();
		focusy_ = this->layer_param_.pinhole_camera_origin_param().focusy();
		u0offset_ = this->layer_param_.pinhole_camera_origin_param().u0offset();
		v0offset_ = this->layer_param_.pinhole_camera_origin_param().v0offset();
	}
	template <typename Dtype>
	void DeepHandModelGen3DSegMapPerChannelLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(BoneNum * depth_dims_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGen3DSegMapPerChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_3d_global_data = bottom[0]->cpu_data(); //aug gt 3d global
		const Dtype* bbx_x1_data = bottom[1]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[2]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[3]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[4]->cpu_data(); //bbx_y2

		const Dtype* x_lb_data = bottom[5]->cpu_data(); //min_x of cube
		const Dtype* x_ub_data = bottom[6]->cpu_data(); //max_x of cube
		const Dtype* y_lb_data = bottom[7]->cpu_data(); //min_y of cube
		const Dtype* y_ub_data = bottom[8]->cpu_data(); //max_y of cube
		const Dtype* z_lb_data = bottom[9]->cpu_data(); //min_z of cube
		const Dtype* z_ub_data = bottom[10]->cpu_data(); //max_z of cube

														 //aug_depth_image to filter out black pixels
		const Dtype* depth_img_data = bottom[11]->cpu_data(); //depth image
		const Dtype* avgz_data = bottom[12]->cpu_data(); //z of centroid


		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int Jid = t * JointNum * 2;
			int Gid = t * JointNum * 3;
			int Tid = t * BoneNum * depth_dims_ * map_size_ * map_size_;
			//for (int j = 0; j < BoneNum; j++) 
			//clear
			for (int j = 0; j < BoneNum * depth_dims_ * map_size_ * map_size_; j++) top_data[Tid + j] = 0.0;

			double bbx_x1 = bbx_x1_data[t];
			double bbx_y1 = bbx_y1_data[t];
			double bbx_x2 = bbx_x2_data[t];
			double bbx_y2 = bbx_y2_data[t];
			double x_lb = x_lb_data[t];
			double x_ub = x_ub_data[t];
			double y_lb = y_lb_data[t];
			double y_ub = y_ub_data[t];
			double z_lb = z_lb_data[t];
			double z_ub = z_ub_data[t];
			double avg_d = avgz_data[t];

			//for each voxel find the closest bone
			for (int k = 0; k < depth_dims_; k++)
			{
				for (int row = 0; row < map_size_; row++)
				{
					for (int col = 0; col < map_size_; col++)
					{
						//True 3D bounding box
						double global_x = double(col) / double(map_size_) * (x_ub - x_lb) + x_lb;
						double global_y = double(row) / double(map_size_) * (y_ub - y_lb) + y_lb;
						double global_z = double(k) / double(depth_dims_) * (z_ub - z_lb) + z_lb;

						//Project to 2D
						double global_u = focusx_ * global_x / global_z + u0offset_;
						double global_v = focusy_ * global_y / global_z + v0offset_;

						//Local [0, 1] projection * depth_size_
						double local_u = (global_u - bbx_x1) / (bbx_x2 - bbx_x1) * depth_size_;
						double local_v = (global_v - bbx_y1) / (bbx_y2 - bbx_y1) * depth_size_;

						int t_col = min(max(0, (int)local_u), depth_size_ - 1);
						int t_row = min(max(0, (int)local_v), depth_size_ - 1);

						//int t_row = int(double(row) / double(map_size_) * depth_size_);				
						//int t_col = int(double(col) / double(map_size_) * depth_size_);
						int Did = t * 3 * depth_size_ * depth_size_;
						double d_in_depth = depth_img_data[Did + t_row * depth_size_ + t_col];
						if (d_in_depth > 1e-6)
						{
							//not black pixel
							int min_bone_id = 0;
							double min_bone_dist = 1000000000;
							double min_bone_prob = 0.0;
							//Find closest point -> bone distance
							for (int j = 0; j < BoneNum; j++)
							{
								//====two endpoints 				
								int u = bones[j][0];
								int v = bones[j][1];
								double x_mid = (gt_joint_3d_global_data[Gid + u * 3] + gt_joint_3d_global_data[Gid + v * 3]) * 0.5;
								double y_mid = (gt_joint_3d_global_data[Gid + u * 3 + 1] + gt_joint_3d_global_data[Gid + v * 3 + 1]) * 0.5;
								double z_mid = (gt_joint_3d_global_data[Gid + u * 3 + 2] + gt_joint_3d_global_data[Gid + v * 3 + 2]) * 0.5;
							    double dist_2_bone = sqrt(pow(global_x - x_mid, 2) + pow(global_y - y_mid, 2) + pow(global_z - z_mid, 2));
								if (min_bone_dist - dist_2_bone > 1e-6)
								{
									min_bone_dist = dist_2_bone;
									min_bone_id = j;
								}
							}
							top_data[Tid + min_bone_id * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = exp(-gamma_ * min_bone_dist);// 1.0;
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelGen3DSegMapPerChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_3d_global_data = bottom[0]->cpu_data(); //aug gt 3d global
		//aug_depth_image to filter out black pixels
		const Dtype* bbx_x1_data = bottom[1]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[2]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[3]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[4]->cpu_data(); //bbx_y2

		const Dtype* x_lb_data = bottom[5]->cpu_data(); //min_x of cube
		const Dtype* x_ub_data = bottom[6]->cpu_data(); //max_x of cube
		const Dtype* y_lb_data = bottom[7]->cpu_data(); //min_y of cube
		const Dtype* y_ub_data = bottom[8]->cpu_data(); //max_y of cube
		const Dtype* z_lb_data = bottom[9]->cpu_data(); //min_z of cube
		const Dtype* z_ub_data = bottom[10]->cpu_data(); //max_z of cube

														 //aug_depth_image to filter out black pixels
		const Dtype* depth_img_data = bottom[11]->cpu_data(); //depth image
		const Dtype* avgz_data = bottom[12]->cpu_data(); //z of centroid
		
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_3d_diff = bottom[0]->mutable_cpu_diff();
		
		if (perform_backprop_)
		{

			for (int t = 0; t < batSize; t++)
			{
				int Jid = t * JointNum * 2;
				int Gid = t * JointNum * 3;
				int Tid = t * BoneNum * depth_dims_ * map_size_ * map_size_;
				//for (int j = 0; j < BoneNum; j++) 
				//clear
				for (int j = 0; j < JointNum * 3; j++) bottom_3d_diff[Gid + j] = 0.0;
				
				double bbx_x1 = bbx_x1_data[t];
				double bbx_y1 = bbx_y1_data[t];
				double bbx_x2 = bbx_x2_data[t];
				double bbx_y2 = bbx_y2_data[t];
				double x_lb = x_lb_data[t];
				double x_ub = x_ub_data[t];
				double y_lb = y_lb_data[t];
				double y_ub = y_ub_data[t];
				double z_lb = z_lb_data[t];
				double z_ub = z_ub_data[t];
				double avg_d = avgz_data[t];

				//for each voxel find the closest bone
				for (int k = 0; k < depth_dims_; k++)
				{
					for (int row = 0; row < map_size_; row++)
					{
						for (int col = 0; col < map_size_; col++)
						{
							//True 3D bounding box
							double global_x = double(col) / double(map_size_) * (x_ub - x_lb) + x_lb;
							double global_y = double(row) / double(map_size_) * (y_ub - y_lb) + y_lb;
							double global_z = double(k) / double(depth_dims_) * (z_ub - z_lb) + z_lb;

							//Project to 2D
							double global_u = focusx_ * global_x / global_z + u0offset_;
							double global_v = focusy_ * global_y / global_z + v0offset_;

							//Local [0, 1] projection * depth_size_
							double local_u = (global_u - bbx_x1) / (bbx_x2 - bbx_x1) * depth_size_;
							double local_v = (global_v - bbx_y1) / (bbx_y2 - bbx_y1) * depth_size_;

							int t_col = min(max(0, (int)local_u), depth_size_ - 1);
							int t_row = min(max(0, (int)local_v), depth_size_ - 1);

							//int t_row = int(double(row) / double(map_size_) * depth_size_);				
							//int t_col = int(double(col) / double(map_size_) * depth_size_);
							int Did = t * 3 * depth_size_ * depth_size_;
							double d_in_depth = depth_img_data[Did + t_row * depth_size_ + t_col];
							if (d_in_depth > 1e-6)
							{
								//not black pixel
								int min_bone_id = 0;
								double min_bone_dist = 1000000000;
								double min_bone_prob = 0.0;
							    double min_dldxu = 0.0, min_dldxv = 0.0, min_dldyu = 0.0, min_dldyv = 0.0, min_dldzu = 0.0, min_dldzv = 0.0;
								int min_u = 0, min_v = 0;
								//Find closest point -> bone distance
								for (int j = 0; j < BoneNum; j++)
								{
									//====two endpoints 				
									int u = bones[j][0];
									int v = bones[j][1];
									double x_mid = (gt_joint_3d_global_data[Gid + u * 3] + gt_joint_3d_global_data[Gid + v * 3]) * 0.5;
									double y_mid = (gt_joint_3d_global_data[Gid + u * 3 + 1] + gt_joint_3d_global_data[Gid + v * 3 + 1]) * 0.5;
									double z_mid = (gt_joint_3d_global_data[Gid + u * 3 + 2] + gt_joint_3d_global_data[Gid + v * 3 + 2]) * 0.5;
									double P = pow(global_x - x_mid, 2) + pow(global_y - y_mid, 2) + pow(global_z - z_mid, 2);
									double dist_2_bone = sqrt(P);
									double prob = exp(-gamma_ * dist_2_bone);
									double dPdxu = 2 * (global_x - x_mid) * -1.0 / 2.0;
									double dPdxv = 2 * (global_x - x_mid) * -1.0 / 2.0;
									double dPdyu = 2 * (global_y - y_mid) * -1.0 / 2.0;
									double dPdyv = 2 * (global_y - y_mid) * -1.0 / 2.0;
									double dPdzu = 2 * (global_z - z_mid) * -1.0 / 2.0;
									double dPdzv = 2 * (global_z - z_mid) * -1.0 / 2.0;
									double ddistdP = 1.0 / (2.0 * sqrt(P));
									double dprobddist = prob * -gamma_;
									double dldprob = top_diff[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col];
									double dldxu = dldprob * dprobddist * ddistdP * dPdxu;
									double dldxv = dldprob * dprobddist * ddistdP * dPdxv;
									double dldyu = dldprob * dprobddist * ddistdP * dPdyu;
									double dldyv = dldprob * dprobddist * ddistdP * dPdyv;
									double dldzu = dldprob * dprobddist * ddistdP * dPdzu;
									double dldzv = dldprob * dprobddist * ddistdP * dPdzv;

									if (min_bone_dist - dist_2_bone > 1e-6)
									{
										min_bone_dist = dist_2_bone;
										min_bone_id = j;
										min_dldxu = dldxu;
										min_dldxv = dldxv;
										min_dldyu = dldyu;
										min_dldyv = dldyv;
										min_dldzu = dldzu;
										min_dldzv = dldzv;
									}
								}

								bottom_3d_diff[Gid + min_u * 3] = min_dldxu;
								bottom_3d_diff[Gid + min_u * 3 + 1] = min_dldyu;
								bottom_3d_diff[Gid + min_u * 3 + 2] = min_dldzu;
								bottom_3d_diff[Gid + min_v * 3] = min_dldxv;
								bottom_3d_diff[Gid + min_v * 3 + 1] = min_dldyv;
								bottom_3d_diff[Gid + min_v * 3 + 2] = min_dldzv;
								//top_data[Tid + min_bone_id * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = exp(-gamma_ * min_bone_dist);// 1.0;
							}
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGen3DSegMapPerChannelLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGen3DSegMapPerChannelLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGen3DSegMapPerChannel);
}

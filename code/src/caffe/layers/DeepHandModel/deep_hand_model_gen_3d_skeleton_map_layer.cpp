
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

#define PI 3.14159265359
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_dims_ = this->layer_param_.gen_3d_skeleton_map_param().depth_dims();
		map_size_ = this->layer_param_.gen_3d_skeleton_map_param().map_size();
		
		line_width_ = this->layer_param_.gen_3d_skeleton_map_param().line_width();
		
		color_label_option_ = this->layer_param_.gen_3d_skeleton_map_param().color_label_option();


		x_lb_ = this->layer_param_.gen_3d_skeleton_map_param().x_lb();
		x_ub_ = this->layer_param_.gen_3d_skeleton_map_param().x_ub();

		y_lb_ = this->layer_param_.gen_3d_skeleton_map_param().y_lb();
		y_ub_ = this->layer_param_.gen_3d_skeleton_map_param().y_ub();

		z_lb_ = this->layer_param_.gen_3d_skeleton_map_param().z_lb();
		z_ub_ = this->layer_param_.gen_3d_skeleton_map_param().z_ub();
		endpoint_dist_threshold_ = this->layer_param_.gen_3d_skeleton_map_param().endpoint_dist_threshold();


	}
	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(3 * depth_dims_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_2d_data = bottom[0]->cpu_data(); //gt joint 2d [0,  1]
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int Jid = t * JointNum * 2;
			int Did = t * JointNum;
			int Tid = t * 3 * depth_dims_ * map_size_ * map_size_;
			//for (int j = 0; j < BoneNum; j++) 
			//clear
			for (int k = 0; k < depth_dims_; k++)
			{
				for (int row = 0; row < map_size_; row++)
				{
					for (int col = 0; col < map_size_; col++)
					{
						for (int c = 0; c < 3; c++)
						{
							top_data[Tid + c * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = 0.0;
						}						
					}
				}
			}

			//for each voxel find the closest bone
			for (int k = 0; k < depth_dims_; k++)
			{
				for (int row = 0; row < map_size_; row++)
				{
					for (int col = 0; col < map_size_; col++)
					{
						double x0 = double(col) / double(map_size_) * (x_ub_ - x_lb_) + x_lb_;
						double y0 = double(row) / double(map_size_) * (y_ub_ - y_lb_) + y_lb_;
						double z0 = double(k) / double(depth_dims_) * (z_ub_ - z_lb_) + z_lb_;
						int min_dist_bone_id = -1;
						double min_dist_2_bone = 1e30;
						for (int j = 0; j < BoneNum; j++)
						{
							//====two endpoints 
							int u = bones[j][0];
							int v = bones[j][1];
							double x1 = gt_joint_2d_data[Jid + u * 2];
							double x2 = gt_joint_2d_data[Jid + v * 2];
							double y1 = gt_joint_2d_data[Jid + u * 2 + 1];
							double y2 = gt_joint_2d_data[Jid + v * 2 + 1];
							double z1 = gt_depth_data[Did + u];
							double z2 = gt_depth_data[Did + v];
							//=========solve distance of point 2 line (3D)
							//====see details here: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
							double fz = (x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1);
							double fm = pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2) + 1e-8;
							double t = -fz / fm;
							double d = sqrt(pow((x1 - x0) + (x2 - x1) * t, 2) + pow((y1 - y0) + (y2 - y1) * t, 2) + pow((z1 - z0) + (z2 - z1) * t, 2));
							//========distance from point X0 to line X1->X2 X0 = [x0, y0, z0]^T, X1 = [x1, y1, z1]^T, X2 = [x2, y2, z2]^T
							bool flag_x_a;
							bool flag_x_b;
							//x2 > x0 > x1 or x1 > x0 > x2 x0 = x1 (or x2) endpoint 
							if (x0 - x1 > 1e-6 || fabs(x0 - x1) < 1e-6) flag_x_a = true; else flag_x_a = false;
							if (x2 - x0 > 1e-6 || fabs(x2 - x0) < 1e-6) flag_x_b = true; else flag_x_b = false;

							bool flag_y_a;
							bool flag_y_b;
							//y2 > y0 > y1 or y1 > y0 > y2 or endpoint
							if (y0 - y1 > 1e-6 || fabs(y0 - y1) < 1e-6) flag_y_a = true; else flag_y_a = false;
							if (y2 - y0 > 1e-6 || fabs(y2 - y0) < 1e-6) flag_y_b = true; else flag_y_b = false;

							bool flag_z_a;
							bool flag_z_b;
							//z2 > z0 > z1 or z1 > z0 > z2 or endpoint
							if (z0 - z1 > 1e-6 || fabs(z0 - z1) < 1e-6) flag_z_a = true; else flag_z_a = false;
							if (z2 - z0 > 1e-6 || fabs(z2 - z0) < 1e-6) flag_z_b = true; else flag_z_b = false;

							//brute force set if the x, y, or z turns out to be the same
							if (fabs(x1 - x2) < endpoint_dist_threshold_)
							{
								flag_x_a = true;
								flag_x_b = true;
							}
							if (fabs(y1 - y2) < endpoint_dist_threshold_)
							{
								flag_y_a = true;
								flag_y_b = true;
							}
							if (fabs(z1 - z2) < endpoint_dist_threshold_)
							{
								flag_z_a = true;
								flag_z_b = true;
							}


							if (line_width_ - d > 1e-6) //distance from point to line within threshold
														//next step: consider occlusion
							{
								if (flag_x_a == flag_x_b && flag_y_a == flag_y_b && flag_z_a == flag_z_b)
								{
									if (min_dist_2_bone - d > 1e-6)
									{
										min_dist_2_bone = d;
										min_dist_bone_id = j;
									}
								}
							}
						}
						
						//found some bone which is within a given threshold line_width
						if (min_dist_bone_id != -1)
						{
							for (int c = 0; c < 3; c++)
							{
								if (color_label_option_ == 0)
								{
									top_data[Tid + c * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = skeleton_color_bone_gt[min_dist_bone_id][c] / 256.0;
								}
							}
						}
					}
				}
			}
			
			/*for (int j = 0; j < 4; j++) 
			{
				//====two endpoints 
				int u = bones[j][0];
				int v = bones[j][1];
				double x1 = gt_joint_2d_data[Jid + u * 2];
				double x2 = gt_joint_2d_data[Jid + v * 2];
				double y1 = gt_joint_2d_data[Jid + u * 2 + 1];
				double y2 = gt_joint_2d_data[Jid + v * 2 + 1];
				double z1 = gt_depth_data[Did + u];
				double z2 = gt_depth_data[Did + v];
				//=========solve distance of point 2 line (3D)
				//====see details here: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
				
				for (int k = 0; k < depth_dims_; k++) 
				{
					for (int row = 0; row < map_size_; row++) {
						for (int col = 0; col < map_size_; col++) {
							double x0 = double(col) / double(map_size_) * (x_ub_ - x_lb_) + x_lb_;
							double y0 = double(row) / double(map_size_) * (y_ub_ - y_lb_) + y_lb_;
							double z0 = double(k) / double(depth_dims_) * (z_ub_ - z_lb_) + z_lb_;
							double fz = (x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1);
							double fm = pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2) + 1e-8;
							double t = -fz / fm;
							double d = sqrt(pow((x1 - x0) + (x2 - x1) * t, 2) + pow((y1 - y0) + (y2 - y1) * t, 2) + pow((z1 - z0) + (z2 - z1) * t, 2));
							//========distance from point X0 to line X1->X2 X0 = [x0, y0, z0]^T, X1 = [x1, y1, z1]^T, X2 = [x2, y2, z2]^T

							bool flag_x_a;
							bool flag_x_b;
							//x2 > x0 > x1 or x1 > x0 > x2 x0 = x1 (or x2) endpoint 
							if (x0 - x1 > 1e-6 || fabs(x0 - x1) < 1e-6) flag_x_a = true; else flag_x_a = false;
							if (x2 - x0 > 1e-6 || fabs(x2 - x0) < 1e-6) flag_x_b = true; else flag_x_b = false;

							bool flag_y_a;
							bool flag_y_b;
							//y2 > y0 > y1 or y1 > y0 > y2 or endpoint
							if (y0 - y1 > 1e-6 || fabs(y0 - y1) < 1e-6) flag_y_a = true; else flag_y_a = false;
							if (y2 - y0 > 1e-6 || fabs(y2 - y0) < 1e-6) flag_y_b = true; else flag_y_b = false;

							bool flag_z_a;
							bool flag_z_b;
							//z2 > z0 > z1 or z1 > z0 > z2 or endpoint
							if (z0 - z1 > 1e-6 || fabs(z0 - z1) < 1e-6) flag_z_a = true; else flag_z_a = false;
							if (z2 - z0 > 1e-6 || fabs(z2 - z0) < 1e-6) flag_z_b = true; else flag_z_b = false;

							//brute force set if the x, y, or z turns out to be the same
							if (fabs(x1 - x2) < 1e-6)
							{
								flag_x_a = true;
								flag_x_b = true;
							}
							if (fabs(y1 - y2) < 1e-6)
							{
								flag_y_a = true;
								flag_y_b = true;
							}
							if (fabs(z1 - z2) < 1e-6)
							{
								flag_z_a = true;
								flag_z_b = true;
							}


							if (line_width_ - d > 1e-6) //distance from point to line within threshold
								//next step: consider occlusion
							{
								if (flag_x_a == flag_x_b && flag_y_a == flag_y_b && flag_z_a == flag_z_b)
								{
									for (int c = 0; c < 3; c++)
									{
										if (color_label_option_ == 0)
										{
											top_data[Tid + c * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = skeleton_color_bone_gt[j][c] / 256.0;
										}
									}
								}
								
							}
							
						}
					}
				}
			}*/
		}
	}

	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGen3DSkeletonMapLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGen3DSkeletonMapLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGen3DSkeletonMap);
}

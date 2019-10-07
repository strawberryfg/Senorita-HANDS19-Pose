
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

#define PI 3.14159265359
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGen3DHeatmapInMoreDetailLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_dims_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_param().depth_dims();
		map_size_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_param().map_size();
		sigma_x_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_param().sigma_x();
		sigma_y_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_param().sigma_y();
		sigma_z_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_param().sigma_z();
		joint_num_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_param().joint_num();

		x_lower_bound_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_v2_param().x_lower_bound();
		x_upper_bound_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_v2_param().x_upper_bound();
		y_lower_bound_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_v2_param().y_lower_bound();
		y_upper_bound_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_v2_param().y_upper_bound();
		z_lower_bound_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_v2_param().z_lower_bound();
		z_upper_bound_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_in_more_detail_v2_param().z_upper_bound();

		perform_backprop_ = this->layer_param_.gen_3d_skeleton_map_param().perform_backprop();
	}
	template <typename Dtype>
	void DeepHandModelGen3DHeatmapInMoreDetailLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_ * depth_dims_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGen3DHeatmapInMoreDetailLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_2d_data = bottom[0]->cpu_data(); //gt joint 2d [0,  1]
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int Jid = t * joint_num_ * 2;
			int Did = t * joint_num_;
			int Tid = t * joint_num_ * depth_dims_ * map_size_ * map_size_;
			for (int j = 0; j < joint_num_ * depth_dims_ * map_size_ * map_size_; j++) top_data[Tid + j] = 0.0;

			for (int j = 0; j < joint_num_; j++) {

				for (int k = 0; k < depth_dims_; k++) {

					for (int row = 0; row < map_size_; row++) {
						for (int col = 0; col < map_size_; col++) {
							double x = double(col) / double(map_size_);
							double y = double(row) / double(map_size_);
							//double z = 2.0 * (1.0 / double(depth_dims_ - 1.0) * k) - 1.0;
							double z = (1.0 / double(depth_dims_ - 1.0) * k); // in range [0, 1]

							double gt_x = (gt_joint_2d_data[Jid + j * 2] - x_lower_bound_) / (x_upper_bound_ - x_lower_bound_);
							double gt_y = (gt_joint_2d_data[Jid + j * 2 + 1] - y_lower_bound_) / (y_upper_bound_ - y_lower_bound_);
							double gt_z = (gt_depth_data[Did + j] - z_lower_bound_) / (z_upper_bound_ - z_lower_bound_);
							//multiplication of three gaussians in X, Y, Z direction respectively (is there any problem?)

							//most importantly here is removes the normalization denominator (2.0 * pi * sigma * sigma), which in this case seems redundant
							double dist_x = exp(-1.0 / (2.0 * sigma_x_ * sigma_x_) * (pow(x - gt_x, 2)));
							double dist_y = exp(-1.0 / (2.0 * sigma_y_ * sigma_y_) * (pow(y - gt_y, 2)));
							double dist_z = exp(-1.0 / (2.0 * sigma_z_ * sigma_z_) * (pow(z - gt_z, 2)));


							double dist = dist_x * dist_y * dist_z;

							if (dist - 0.3 > 1e-6)
							{
								top_data[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = dist;
							}
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelGen3DHeatmapInMoreDetailLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_2d_data = bottom[0]->cpu_data(); //gt joint 2d [0,  1]
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_2d_diff = bottom[0]->mutable_cpu_diff();
		Dtype* bottom_depth_diff = bottom[1]->mutable_cpu_diff();
		if (!perform_backprop_) return;
		for (int t = 0; t < batSize; t++) 
		{
			int Jid = t * joint_num_ * 2;
			int Did = t * joint_num_;
			int Tid = t * joint_num_ * depth_dims_ * map_size_ * map_size_;
			for (int j = 0; j < joint_num_ * 2; j++) bottom_2d_diff[Jid + j] = 0.0;
			for (int j = 0; j < joint_num_; j++) bottom_depth_diff[Did + j] = 0.0;

			for (int j = 0; j < joint_num_; j++) 
			{
				for (int k = 0; k < depth_dims_; k++) 
				{
					for (int row = 0; row < map_size_; row++) 
					{
						for (int col = 0; col < map_size_; col++) 
						{
							double x = double(col) / double(map_size_);
							double y = double(row) / double(map_size_);
							//double z = 2.0 * (1.0 / double(depth_dims_ - 1.0) * k) - 1.0;
						    double z = (1.0 / double(depth_dims_ - 1.0) * k); // in range [0, 1]

							double gt_x = (gt_joint_2d_data[Jid + j * 2] - x_lower_bound_) / (x_upper_bound_ - x_lower_bound_);
							double gt_y = (gt_joint_2d_data[Jid + j * 2 + 1] - y_lower_bound_) / (y_upper_bound_ - y_lower_bound_);
							double gt_z = (gt_depth_data[Did + j] - z_lower_bound_) / (z_upper_bound_ - z_lower_bound_);
							//multiplication of three gaussians in X, Y, Z direction respectively (is there any problem?)

							//most importantly here is removes the normalization denominator (2.0 * pi * sigma * sigma), which in this case seems redundant
							double dist_x = exp(-1.0 / (2.0 * sigma_x_ * sigma_x_) * (pow(x - gt_x, 2)));
							double dist_y = exp(-1.0 / (2.0 * sigma_y_ * sigma_y_) * (pow(y - gt_y, 2)));
							double dist_z = exp(-1.0 / (2.0 * sigma_z_ * sigma_z_) * (pow(z - gt_z, 2)));


							double dist = dist_x * dist_y * dist_z;

							if (dist - 0.3 > 1e-6)
							{
								double ddist_xdgt_x = dist_x * -1.0 / (2.0 * sigma_x_ * sigma_x_) * 2 * (x - gt_x) * -1.0;
								double ddist_ydgt_y = dist_y * -1.0 / (2.0 * sigma_y_ * sigma_y_) * 2 * (y - gt_y) * -1.0;
								double ddist_zdgt_z = dist_z * -1.0 / (2.0 * sigma_z_ * sigma_z_) * 2 * (z - gt_z) * -1.0;

								double dgt_xdx = 1.0 / (x_upper_bound_ - x_lower_bound_);
								double dgt_ydy = 1.0 / (y_upper_bound_ - y_lower_bound_);
								double dgt_zdz = 1.0 / (z_upper_bound_ - z_lower_bound_);

								double ddistddist_x = 1.0 * dist_y * dist_z;
								double ddistddist_y = dist_x * 1.0 * dist_z;
								double ddistddist_z = dist_x * dist_y * 1.0;

								double ddistdx = ddistddist_x * ddist_xdgt_x * dgt_xdx;
								double ddistdy = ddistddist_y * ddist_ydgt_y * dgt_ydy;
								double ddistdz = ddistddist_z * ddist_zdgt_z * dgt_zdz;

								//top_data[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = dist;
								bottom_2d_diff[Jid + j * 2] += top_diff[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] * ddistdx;
								bottom_2d_diff[Jid + j * 2 + 1] += top_diff[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] * ddistdy;
								bottom_depth_diff[Did + j] += top_diff[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] * ddistdz;
							}
						}
					}
				}
		}
	}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGen3DHeatmapInMoreDetailLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGen3DHeatmapInMoreDetailLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGen3DHeatmapInMoreDetail);
}

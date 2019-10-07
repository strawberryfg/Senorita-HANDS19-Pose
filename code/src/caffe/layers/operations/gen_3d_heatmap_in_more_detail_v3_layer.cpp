
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"

#define PI 3.14159265359
namespace caffe {

	template <typename Dtype>
	void Gen3DHeatmapInMoreDetailV3Layer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_dims_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().depth_dims();
		map_size_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().map_size();
		crop_size_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().crop_size();
		render_sigma_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().render_sigma();
		stride_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().stride();

		joint_num_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().joint_num();

		x_lower_bound_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().x_lower_bound();
		x_upper_bound_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().x_upper_bound();
		y_lower_bound_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().y_lower_bound();
		y_upper_bound_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().y_upper_bound();
		z_lower_bound_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().z_lower_bound();
		z_upper_bound_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().z_upper_bound();

		output_res_ = this->layer_param_.gen_3d_heatmap_in_more_detail_v3_param().output_res();
	}
	template <typename Dtype>
	void Gen3DHeatmapInMoreDetailV3Layer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_ * depth_dims_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void Gen3DHeatmapInMoreDetailV3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_3d_data = bottom[0]->cpu_data(); //directly using 3D gt as input
		const Dtype* gt_joint_2d_data = bottom[1]->cpu_data(); //gt joint 2d in bbx
		int depth_stride_ = (bottom[0]->shape())[1] / joint_num_;
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int Jid = t * joint_num_ * depth_stride_;
			int Pid = t * joint_num_ * 2;
			//int Did = t * joint_num_;
			int Tid = t * joint_num_ * depth_dims_ * map_size_ * map_size_;
			//clear 3d hm gt
			for (int j = 0; j < joint_num_ * depth_dims_ * map_size_ * map_size_; j++) top_data[Tid + j] = 0.0;

			for (int j = 0; j < joint_num_; j++)
			{
				//2d heatmap section begins
				int tmp_size = render_sigma_ * 3;
				int size = 2 * tmp_size + 1;
				double c_x = (gt_joint_2d_data[Pid + j * 2] - x_lower_bound_) / (x_upper_bound_ - x_lower_bound_);
				double c_y = (gt_joint_2d_data[Pid + j * 2 + 1] - y_lower_bound_) / (y_upper_bound_ - y_lower_bound_);
				int mu_x = int(c_x * crop_size_ / stride_ + 0.5);
				int mu_y = int(c_y * crop_size_ / stride_ + 0.5);
				int x_lb = mu_x - tmp_size;
				int x_ub = mu_x + tmp_size;
				int y_lb = mu_y - tmp_size;
				int y_ub = mu_y + tmp_size;

				int size_z = 2 * floor((6 * render_sigma_ * double(depth_dims_) / double(output_res_) + 1) / 2.0) + 1;
				int half_size_z = floor(size_z / 2.0);
				for (int row = y_lb; row <= y_ub; row++)
				{
					for (int col = x_lb; col <= x_ub; col++)
					{
						if (row >= 0 && row < map_size_ && col >= 0 && col < map_size_)
						{
							int cur_x = col - mu_x;
							int cur_y = row - mu_y;
							double h_v = exp(-1.0 / (2.0 * render_sigma_ * render_sigma_) * (pow(cur_x, 2) + pow(cur_y, 2)));
							//z is 1-64 scale
							//find the correct bin which depth lies in 
							int z = 1;
							for (int d = 1; d <= output_res_; d++)
							{
								double z_lb = double(d - 1) / output_res_;
								double z_ub = double(d) / output_res_;
								double gt_z = gt_joint_3d_data[Jid + j * depth_stride_ + depth_stride_ - 1];
								if ((gt_z - z_lower_bound_) / (z_upper_bound_ - z_lower_bound_) >= z_lb && (gt_z - z_lower_bound_) / (z_upper_bound_ - z_lower_bound_) <= z_ub)
								{
									z = d;
									break;
								}
							}
							z = ceil(double(z) * double(depth_dims_) / double(output_res_));
							for (int d = int(z - half_size_z); d < int(z + half_size_z + 1); d++)
							{
								int cur_z = d - z;
								double new_sigma_ = double(size_z) / 6.0;
								double hm_z = exp(-1.0 / (2.0 * new_sigma_ * new_sigma_) * (cur_z * cur_z));
								double hm_value = h_v * hm_z;
								if (d >= 1 && d <= depth_dims_)
								{
									top_data[Tid + j * depth_dims_ * map_size_ * map_size_ + (d - 1) * map_size_ * map_size_ + row * map_size_ + col] = hm_value;
								}
							}
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void Gen3DHeatmapInMoreDetailV3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {

		}
	}

#ifdef CPU_ONLY
	STUB_GPU(Gen3DHeatmapInMoreDetailV3Layer);
#endif

	INSTANTIATE_CLASS(Gen3DHeatmapInMoreDetailV3Layer);
	REGISTER_LAYER_CLASS(Gen3DHeatmapInMoreDetailV3);
}

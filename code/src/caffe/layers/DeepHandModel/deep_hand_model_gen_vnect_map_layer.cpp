
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"


namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenVNectMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		map_size_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().map_size();

		joint_num_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().joint_num();

		x_lower_bound_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().x_lower_bound();
		x_upper_bound_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().x_upper_bound();

		y_lower_bound_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().y_lower_bound();
		y_upper_bound_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().y_upper_bound();

		depth_lower_bound_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().depth_lower_bound();
		depth_upper_bound_ = this->layer_param_.deep_hand_model_gen_vnect_map_param().depth_upper_bound();
	}
	template <typename Dtype>
	void DeepHandModelGenVNectMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_ * 3);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGenVNectMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_hm_data = bottom[0]->cpu_data(); //gt heatmap
		const Dtype* gt_3d_data = bottom[1]->cpu_data(); //gt depth       x_i, y_i, z_i

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int Bid = t * joint_num_ * map_size_ * map_size_;
			int Did = t * 3 * joint_num_;
			int Tid = t * 3 * joint_num_ * map_size_ * map_size_; // X, Y, Z
			for (int d = 0; d < 3; d++) //d stands for dimension x, y, z
			{
				for (int j = 0; j < joint_num_; j++)
				{
					double lb_, ub_;
					if (d == 0)
					{
						lb_ = x_lower_bound_;
						ub_ = x_upper_bound_;
					}
					else if (d == 1)
					{
						lb_ = y_lower_bound_;
						ub_ = y_upper_bound_;
					}
					else
					{
						lb_ = depth_lower_bound_;
						ub_ = depth_upper_bound_;
					}

					double gt_v = (gt_3d_data[Did + d * joint_num_ + j] - lb_) / (ub_ - lb_);

					//important restriction
					gt_v = max(0.0, gt_v);
					gt_v = min(1.0, gt_v);

					for (int row = 0; row < map_size_; row++)
					{
						for (int col = 0; col < map_size_; col++)
						{
							double hm_value = gt_hm_data[Bid + j * map_size_ * map_size_ + row * map_size_ + col];
							top_data[Tid + d * joint_num_ * map_size_ * map_size_ + j * map_size_ * map_size_ + row * map_size_ + col] = hm_value * gt_v;
						}
					}
				}
			}

			
		}
	}

	template <typename Dtype>
	void DeepHandModelGenVNectMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {

		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenVNectMapLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenVNectMapLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenVNectMap);
}

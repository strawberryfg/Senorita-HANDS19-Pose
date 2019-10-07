
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"


namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenDepthMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		map_size_ = this->layer_param_.deep_hand_model_gen_depth_map_param().map_size();

		depth_lower_bound_ = this->layer_param_.deep_hand_model_gen_depth_map_param().depth_lower_bound();
		depth_upper_bound_ = this->layer_param_.deep_hand_model_gen_depth_map_param().depth_upper_bound();
	}
	template <typename Dtype>
	void DeepHandModelGenDepthMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		joint_num_ = (bottom[1]->shape())[1];
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGenDepthMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_hm_data = bottom[0]->cpu_data(); //gt heatmap
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int Bid = t * joint_num_ * map_size_ * map_size_;
			int Did = t * joint_num_;
			int Tid = t * joint_num_ * map_size_ * map_size_;
			for (int j = 0; j < joint_num_; j++)
			{
				double gt_depth = (gt_depth_data[Did + j] - depth_lower_bound_) / (depth_upper_bound_ - depth_lower_bound_);

				//important restriction
				gt_depth = max(0.0, gt_depth);
				gt_depth = min(1.0, gt_depth);

				for (int row = 0; row < map_size_; row++)
				{
					for (int col = 0; col < map_size_; col++)
					{
						double hm_value = gt_hm_data[Bid + j * map_size_ * map_size_ + row * map_size_ + col];
						top_data[Tid + j * map_size_ * map_size_ + row * map_size_ + col] = hm_value * gt_depth;
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelGenDepthMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_hm_data = bottom[0]->cpu_data(); //gt heatmap
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]
		
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_hm_diff = bottom[0]->mutable_cpu_diff();
		Dtype* bottom_depth_diff = bottom[1]->mutable_cpu_diff();

		for (int t = 0; t < batSize; t++)
		{
			int Bid = t * joint_num_ * map_size_ * map_size_;
			int Did = t * joint_num_;
			int Tid = t * joint_num_ * map_size_ * map_size_;
			for (int j = 0; j < joint_num_ * map_size_ * map_size_; j++) bottom_hm_diff[Bid + j] = 0.0;
			for (int j = 0; j < joint_num_; j++) bottom_depth_diff[Did + j] = 0.0;
			for (int j = 0; j < joint_num_; j++)
			{
				double gt_depth = (gt_depth_data[Did + j] - depth_lower_bound_) / (depth_upper_bound_ - depth_lower_bound_);

				//important restriction
				gt_depth = max(0.0, gt_depth);
				gt_depth = min(1.0, gt_depth);

				for (int row = 0; row < map_size_; row++)
				{
					for (int col = 0; col < map_size_; col++)
					{
						double hm_value = gt_hm_data[Bid + j * map_size_ * map_size_ + row * map_size_ + col];
						//top_data[Tid + j * map_size_ * map_size_ + row * map_size_ + col] = hm_value * gt_depth;
						bottom_hm_diff[Bid + j * map_size_ * map_size_ + row * map_size_ + col] += top_diff[Tid + j * map_size_ * map_size_ + row * map_size_ + col] * (gt_depth);
						bottom_depth_diff[Did + j] += top_diff[Tid + j * map_size_ * map_size_ + row * map_size_ + col] * hm_value * 1.0 / (depth_upper_bound_ - depth_lower_bound_);
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenDepthMapLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenDepthMapLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenDepthMap);
}

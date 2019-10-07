
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenHeatmapAllChannelsLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		gen_size_ = this->layer_param_.deep_hand_model_gen_heatmap_all_channels_param().gen_size();

		render_sigma_ = this->layer_param_.deep_hand_model_gen_heatmap_all_channels_param().render_sigma();

	}
	template <typename Dtype>
	void DeepHandModelGenHeatmapAllChannelsLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		joint_num_ = (bottom[0]->shape())[1] / 2;
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_);
		top_shape.push_back(gen_size_);
		top_shape.push_back(gen_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGenHeatmapAllChannelsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //2d gt [0,1]

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) 
		{
			int Bid = t * joint_num_ * 2;
			int Tid = t * joint_num_ * gen_size_ * gen_size_;
			for (int row = 0; row < gen_size_; row++) 
			{
				for (int col = 0; col < gen_size_; col++) 
				{
					for (int channel = 0; channel < joint_num_; channel++) 
					{
						float gt_x = bottom_data[Bid + channel * 2], gt_y = bottom_data[Bid + channel * 2 + 1];
						float t = exp(-1.0 / (2.0 * (render_sigma_ * render_sigma_)) * (pow(col / float(gen_size_) - gt_x, 2) + pow(row / float(gen_size_) - gt_y, 2)));
						top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = t;						
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelGenHeatmapAllChannelsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		for (int t = 0; t < batSize; t++)
		{
			int Bid = t * joint_num_ * 2;
			int Tid = t * joint_num_ * gen_size_ * gen_size_;
			for (int j = 0; j < joint_num_ * 2; j++) bottom_diff[Bid + j] = 0.0;
			for (int row = 0; row < gen_size_; row++)
			{
				for (int col = 0; col < gen_size_; col++)
				{
					for (int channel = 0; channel < joint_num_; channel++)
					{
						float gt_x = bottom_data[Bid + channel * 2], gt_y = bottom_data[Bid + channel * 2 + 1];
						float t = exp(-1.0 / (2.0 * (render_sigma_ * render_sigma_)) * (pow(col / float(gen_size_) - gt_x, 2) + pow(row / float(gen_size_) - gt_y, 2)));
						//top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = t;
						bottom_diff[Bid + channel * 2] += top_diff[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] * t * -1.0 / (2.0 * (render_sigma_ * render_sigma_)) * 2 * (col / float(gen_size_) - gt_x) * -1.0;
						bottom_diff[Bid + channel * 2 + 1] += top_diff[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] * t * -1.0 / (2.0 * (render_sigma_ * render_sigma_)) * 2 * (row / float(gen_size_) - gt_y) * -1.0; 
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenHeatmapAllChannelsLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenHeatmapAllChannelsLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenHeatmapAllChannels);
}

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"

//#define _DEBUG
namespace caffe {

	template <typename Dtype>
	void EltAddChannelsLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		dim_lb_ = this->layer_param_.elt_add_channels_param().dim_lb();
		dim_ub_ = this->layer_param_.elt_add_channels_param().dim_ub();
		
	}
	template <typename Dtype>
	void EltAddChannelsLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1); //grayscale
		top_shape.push_back((bottom[0]->shape())[2]);
		top_shape.push_back((bottom[0]->shape())[3]);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void EltAddChannelsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();  //feature map patch
		
		Dtype* top_data = top[0]->mutable_cpu_data();
		int channels = (bottom[0]->shape())[1];
		int h = (bottom[0]->shape())[2];
		int w = (bottom[0]->shape())[3];


		for (int t = 0; t < batSize; t++) 
		{
			
			int Tid = t * 1 * h * w;
			int Bid = t * channels * h * w;
			//clear top data
			for (int row = 0; row < h; row++)
			{
				for (int col = 0; col < w;col++)
				{
					top_data[Tid + row * w + col] = 0.0;
				}
			}

			for (int c = 0; c < channels; c++) 
			{
				if (c >= dim_lb_ && c <= dim_ub_) //within interested dimension range
				{
					for (int row = 0; row < h; row++)
					{
						for (int col = 0; col < w; col++)
						{
							top_data[Tid + row * w + col] += bottom_data[Bid + c * h * w + row * w + col];
						}
					}
				}				
			}
		}
	}

	template <typename Dtype>
	void EltAddChannelsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {

			
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const int batSize = (bottom[0]->shape())[0];

			int channels = (bottom[0]->shape())[1];
			int h = (bottom[0]->shape())[2];
			int w = (bottom[0]->shape())[3];

			for (int t = 0; t < batSize; t++) 
			{
				int Tid = t * 1 * h * w;
				int Bid = t * channels * h * w;
				for (int c = 0; c < channels; c++)
				{
					for (int row = 0; row < h; row++) 
					{
						for (int col = 0; col < w; col++) 
						{
							bottom_diff[Bid + c * h * w + row * w + col] = 0.0;
						}
					}
				}
				


				for (int c = 0; c < channels; c++) 
				{
					if (c >= dim_lb_ && c <= dim_ub_) //within interested dimension range
					{
						for (int row = 0; row < h; row++) 
						{
							for (int col = 0; col < w; col++) 
							{
								bottom_diff[Bid + c * h * w + row * w + col] = top_diff[Tid + row * w + col] * 1.0;
							}
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(EltAddChannelsLayer);
#endif

	INSTANTIATE_CLASS(EltAddChannelsLayer);
	REGISTER_LAYER_CLASS(EltAddChannels);
}

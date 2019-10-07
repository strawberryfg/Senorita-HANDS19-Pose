#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


using namespace cv;
//#define _DEBUG
namespace caffe {

	template <typename Dtype>
	void MaskFeatureMapByFeatureMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		skeleton_threshold_ = this->layer_param_.skeleton_map_param().skeleton_threshold();
	}
	template <typename Dtype>
	void MaskFeatureMapByFeatureMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top_shape.push_back((bottom[0]->shape())[2]);
		top_shape.push_back((bottom[0]->shape())[3]);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void MaskFeatureMapByFeatureMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* featuremap_data = bottom[0]->cpu_data();  //feature map patch
		const Dtype* mask_data = bottom[1]->cpu_data(); //IN THIS CASE MASK IS USUALLY THE GROUND TRUTH MAP
		Dtype* top_data = top[0]->mutable_cpu_data();
		int channels = (bottom[0]->shape())[1];
		int h = (bottom[0]->shape())[2];
		int w = (bottom[0]->shape())[3];

		
		for (int t = 0; t < batSize; t++)
		{
			for (int c = 0; c < channels; c++)
			{
				int Tid = t * channels * h * w;
				int Did = Tid;
				for (int row = 0; row < h; row++)
				{
					for (int col = 0; col < w; col++)
					{
						int ske = mask_data[Did + c * h * w + row * w + col] * 255;
						
						if (ske > skeleton_threshold_)
							//the mask is one (mask map contains value)
						{
							top_data[Tid + c * h * w + row * w + col] = featuremap_data[Did + c * h * w + row * w + col];
						}
						else
						{
							top_data[Tid + c * h * w + row * w + col] = 0;
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void MaskFeatureMapByFeatureMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* mask_data = bottom[1]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int batSize = (bottom[0]->shape())[0];

		int channels = (bottom[0]->shape())[1];
		int h = (bottom[0]->shape())[2];
		int w = (bottom[0]->shape())[3];

		for (int t = 0; t < batSize; t++)
		{
			for (int c = 0; c < channels; c++)
			{
				int Bid = t * channels * h * w;
				int Tid = Bid;
				for (int row = 0; row < h; row++)
				{
					for (int col = 0; col < w; col++)
					{
						int ske = mask_data[Bid + c * h * w + row * w + col] * 255.0;
						if (ske > skeleton_threshold_)
							//the mask is one (mask map contains value)
						{
							bottom_diff[Bid + c * h * w + row * w + col] = 1.0 * top_diff[Tid + c * h * w + row * w + col];
						}
						else
						{
							bottom_diff[Bid + c * h * w + row * w + col] = 0.0;
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MaskFeatureMapByFeatureMapLayer);
#endif

	INSTANTIATE_CLASS(MaskFeatureMapByFeatureMapLayer);
	REGISTER_LAYER_CLASS(MaskFeatureMapByFeatureMap);
}

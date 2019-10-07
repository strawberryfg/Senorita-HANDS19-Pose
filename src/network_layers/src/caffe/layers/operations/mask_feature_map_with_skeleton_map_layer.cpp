#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


using namespace cv;
//#define _DEBUG
namespace caffe {

	template <typename Dtype>
	void MaskFeatureMapBySkeletonLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		skeleton_threshold_ = this->layer_param_.skeleton_map_param().skeleton_threshold();
	}
	template <typename Dtype>
	void MaskFeatureMapBySkeletonLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top_shape.push_back((bottom[0]->shape())[2]);
		top_shape.push_back((bottom[0]->shape())[3]);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void MaskFeatureMapBySkeletonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* featuremap_data = bottom[0]->cpu_data();  //feature map patch
		const Dtype* skeleton_data = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int channels = (bottom[0]->shape())[1];
		int h = (bottom[0]->shape())[2];
		int w = (bottom[0]->shape())[3];

		//rows and cols of skeleton data
		int ske_h = (bottom[1]->shape())[2];
		int ske_w = (bottom[1]->shape())[3];

		for (int t = 0; t < batSize; t++) 
		{
			Mat raw_ske = Mat::zeros(Size(ske_w, ske_h), CV_8UC3);
			for (int c = 0; c < 3; c++)
			{
				int ske_id = t * 3 * ske_h * ske_w;
				for (int row = 0; row < ske_h; row++)
				{
					for (int col = 0; col < ske_w; col++)
					{
						raw_ske.at<Vec3b>(row, col)[c] = min(255.0, skeleton_data[ske_id + c * ske_h * ske_w + row * ske_w + col] * 1.0);
					}
				}
			}
			resize(raw_ske, raw_ske, Size(w, h));

			for (int c = 0; c < channels; c++)
			{
				int Tid = t * channels * h * w;
				int Did = Tid;
				for (int row = 0; row < h; row++) 
				{
					for (int col = 0; col < w; col++) 
					{						
						int skeleton_b = raw_ske.at<Vec3b>(row, col)[0];
						int skeleton_g = raw_ske.at<Vec3b>(row, col)[1];
						int skeleton_r = raw_ske.at<Vec3b>(row, col)[2];
						if (skeleton_b + skeleton_g + skeleton_r > skeleton_threshold_)
							//the mask is one (skeleton map contains value)
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
	void MaskFeatureMapBySkeletonLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			
			const Dtype* skeleton_data = bottom[1]->cpu_data();

			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const int batSize = (bottom[0]->shape())[0];

			int channels = (bottom[0]->shape())[1];
			int h = (bottom[0]->shape())[2];
			int w = (bottom[0]->shape())[3];

			//rows and cols of skeleton data
			int ske_h = (bottom[1]->shape())[2];
			int ske_w = (bottom[1]->shape())[3];
			for (int t = 0; t < batSize; t++) 
			{
				Mat raw_ske = Mat::zeros(Size(ske_w, ske_h), CV_8UC3);
				for (int c = 0; c < 3; c++) 
				{
					int ske_id = t * 3 * ske_h * ske_w;
					for (int row = 0; row < ske_h; row++) 
					{
						for (int col = 0; col < ske_w; col++) 
						{
							raw_ske.at<Vec3b>(row, col)[c] = min(255.0, skeleton_data[ske_id + c * ske_h * ske_w + row * ske_w + col] * 1.0);
						}
					}
				}
				resize(raw_ske, raw_ske, Size(w, h));

				for (int c = 0; c < channels; c++) 
				{
					int Bid = t * channels * h * w;
					int Tid = Bid;
					for (int row = 0; row < h; row++) 
					{
						for (int col = 0; col < w; col++) 
						{
							int skeleton_b = raw_ske.at<Vec3b>(row, col)[0];
							int skeleton_g = raw_ske.at<Vec3b>(row, col)[1];
							int skeleton_r = raw_ske.at<Vec3b>(row, col)[2];
							if (skeleton_b + skeleton_g + skeleton_r > skeleton_threshold_)
								//the mask is one (skeleton map contains value)
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
	}

#ifdef CPU_ONLY
	STUB_GPU(MaskFeatureMapBySkeletonLayer);
#endif

	INSTANTIATE_CLASS(MaskFeatureMapBySkeletonLayer);
	REGISTER_LAYER_CLASS(MaskFeatureMapBySkeleton);
}

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


using namespace cv;
//#define _DEBUG
namespace caffe {

	template <typename Dtype>
	void MaskFeatureMapOutByMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		skeleton_threshold_ = this->layer_param_.skeleton_map_param().skeleton_threshold();
	}
	template <typename Dtype>
	void MaskFeatureMapOutByMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top_shape.push_back((bottom[0]->shape())[2]);
		top_shape.push_back((bottom[0]->shape())[3]);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void MaskFeatureMapOutByMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* featuremap_data = bottom[0]->cpu_data();  //feature map patch
		const Dtype* skeleton_data = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int channels = (bottom[0]->shape())[1];
		int h = (bottom[0]->shape())[2];
		int w = (bottom[0]->shape())[3];

		int ske_c = (bottom[1]->shape())[1];
		
		//rows and cols of skeleton data
		int ske_h = (bottom[1]->shape())[2];
		int ske_w = (bottom[1]->shape())[3];

		for (int t = 0; t < batSize; t++) 
		{
			Mat raw_ske;
			if (ske_c == 3) raw_ske = Mat::zeros(Size(ske_w, ske_h), CV_8UC3);
			else raw_ske = Mat::zeros(Size(ske_w, ske_h), CV_8UC1);
		
			for (int c = 0; c < ske_c; c++)
			{
				int ske_id = ske_c == 3 ? t * 3 * ske_h * ske_w : t * 1 * ske_h * ske_w;
				for (int row = 0; row < ske_h; row++)
				{
					for (int col = 0; col < ske_w; col++)
					{
						if (ske_c == 3) raw_ske.at<Vec3b>(row, col)[c] = skeleton_data[ske_id + c * ske_h * ske_w + row * ske_w + col];
						else raw_ske.at<uchar>(row, col) = skeleton_data[ske_id + row * ske_w + col];
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
						int ske_value = 0;
						if (ske_c == 3)
						{
							ske_value = raw_ske.at<Vec3b>(row, col)[0] + raw_ske.at<Vec3b>(row, col)[1] + raw_ske.at<Vec3b>(row, col)[2];
						}
						else 
						{
							ske_value = raw_ske.at<uchar>(row, col);
						}
												
						if (ske_value > skeleton_threshold_)
							//the mask is one (skeleton map contains value)
							//mask out 
						{													
							top_data[Tid + c * h * w + row * w + col] = 0;							
						} 
						else 
						{
							top_data[Tid + c * h * w + row * w + col] = featuremap_data[Did + c * h * w + row * w + col];
						}
					}
				}
			}			
		}
	}

	template <typename Dtype>
	void MaskFeatureMapOutByMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			
			const Dtype* skeleton_data = bottom[1]->cpu_data();

			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const int batSize = (bottom[0]->shape())[0];

			int channels = (bottom[0]->shape())[1];
			int h = (bottom[0]->shape())[2];
			int w = (bottom[0]->shape())[3];

			int ske_c = (bottom[1]->shape())[1];
			
			//rows and cols of skeleton data
			int ske_h = (bottom[1]->shape())[2];
			int ske_w = (bottom[1]->shape())[3];
			for (int t = 0; t < batSize; t++) 
			{
				Mat raw_ske;
				if (ske_c == 3) raw_ske = Mat::zeros(Size(ske_w, ske_h), CV_8UC3);
				else raw_ske = Mat::zeros(Size(ske_w, ske_h), CV_8UC1);
				for (int c = 0; c < ske_c; c++)
				{
					int ske_id = ske_c == 3 ? t * 3 * ske_h * ske_w : t * 1 * ske_h * ske_w;
					for (int row = 0; row < ske_h; row++)
					{
						for (int col = 0; col < ske_w; col++)
						{
							if (ske_c == 3) raw_ske.at<Vec3b>(row, col)[c] = skeleton_data[ske_id + c * ske_h * ske_w + row * ske_w + col];
							else raw_ske.at<uchar>(row, col) = skeleton_data[ske_id + row * ske_w + col];
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
							int ske_value = 0;
							if (ske_c == 3)
							{
								ske_value = raw_ske.at<Vec3b>(row, col)[0] + raw_ske.at<Vec3b>(row, col)[1] + raw_ske.at<Vec3b>(row, col)[2];
							}
							else 
							{
								ske_value = raw_ske.at<uchar>(row, col);
							}
												
							if (ske_value > skeleton_threshold_)
							//the mask is one (skeleton map contains value)
							//mask out 
							{
						
								bottom_diff[Bid + c * h * w + row * w + col] = 0.0;								
							} 
							else 
							{
								bottom_diff[Bid + c * h * w + row * w + col] = 1.0 * top_diff[Tid + c * h * w + row * w + col];
							}
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MaskFeatureMapOutByMapLayer);
#endif

	INSTANTIATE_CLASS(MaskFeatureMapOutByMapLayer);
	REGISTER_LAYER_CLASS(MaskFeatureMapOutByMap);
}

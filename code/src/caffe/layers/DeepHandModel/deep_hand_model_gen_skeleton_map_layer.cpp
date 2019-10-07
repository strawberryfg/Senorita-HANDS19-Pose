#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"
using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenSkeletonMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		S = this->layer_param_.gen_skeleton_param().gen_size();
		line_width = this->layer_param_.gen_skeleton_param().line_width();

		
	}
	template <typename Dtype>
	void DeepHandModelGenSkeletonMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(3);
		top_shape.push_back(S);
		top_shape.push_back(S);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGenSkeletonMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int Tid = t * 3 * S * S;
			Mat img = Mat::zeros(Size(S, S), CV_8UC3);
			int Bid = t * JointNum * 2;

			for (int j = 0; j < BoneNum; j++) 
			{
				line(img, Point2d((bottom_data[Bid + bones[j][0] * 2] * S),
					(bottom_data[Bid + bones[j][0] * 2 + 1] * S)),
					Point2d((bottom_data[Bid + bones[j][1] * 2] * S),
					(bottom_data[Bid + bones[j][1] * 2 + 1] * S)), Scalar(skeleton_color_bone_gt[j][0], skeleton_color_bone_gt[j][1], skeleton_color_bone_gt[j][2]), line_width);
			}

			for (int row = 0; row < S; row++)
			{
				for (int col = 0; col < S; col++)
				{
					for (int c = 0; c < 3; c++)
					{
						top_data[Tid + c * S * S + row * S + col] = img.at<Vec3b>(row, col)[c] / 256.0;
					}
				}
			}
		}
	}


	template <typename Dtype>
	void DeepHandModelGenSkeletonMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenSkeletonMapLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenSkeletonMapLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenSkeletonMap);
}

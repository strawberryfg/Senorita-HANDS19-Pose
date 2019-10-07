#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DeepHandModelInterpolateKeypointsLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		interpolate_num_ = this->layer_param_.deep_hand_model_hands19_param().interpolate_num();
		interpolate_id_ = this->layer_param_.deep_hand_model_hands19_param().interpolate_id();

	}

	template <typename Dtype>
	void DeepHandModelInterpolateKeypointsLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = JointNum * 3;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelInterpolateKeypointsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) 
		{
			int Tid = t * JointNum * 3;
			int Bid = t * JointNum * 3;
			//Set root: copy wrist directly
			for (int j = 0; j < 3; j++) top_data[Tid + wrist * 3 + j] = bottom_data[Bid + wrist * 3 + j];
			for (int k = 0; k < BoneNum; k++) 
			{
				int u = bones[k][0], v = bones[k][1];
				int Uid = Bid + u * 3, Vid = Bid + v * 3;
				for (int l = 0; l < 3; l++) 
				{
					double VminusU = bottom_data[Vid + l] - bottom_data[Uid + l];
					top_data[Tid + u * 3 + l] = bottom_data[Uid + l] + interpolate_id_ / (interpolate_num_ + 1) * VminusU;
					//U + interpolate_id / (interpolate_num + 1) * (V - U)
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelInterpolateKeypointsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelInterpolateKeypointsLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelInterpolateKeypointsLayer);
	REGISTER_LAYER_CLASS(DeepHandModelInterpolateKeypoints);

}  // namespace caffe

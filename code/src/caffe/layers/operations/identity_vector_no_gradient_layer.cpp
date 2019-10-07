#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"
namespace caffe {

	template <typename Dtype>
	void IdentityVectorNoGradientLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	}

	template <typename Dtype>
	void IdentityVectorNoGradientLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top[0]->Reshape(top_shape);

		//Or
		//const int axis = bottom[0]->CanonicalAxisIndex(
		//	this->layer_param_.inner_product_param().axis());
		//vector<int> top_shape = bottom[0]->shape();
		//top_shape.resize(axis + 1);
		//top_shape[axis] = (bottom[0]->shape())[1];
		//top[0]->Reshape(top_shape);
	}


	template <typename Dtype>
	void IdentityVectorNoGradientLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		const int dimSize = (bottom[0]->shape())[1]; //2 dimensions
		for (int t = 0; t < batSize; t++) 
		{
			for (int i = 0; i < dimSize; i++) 
			{
				int Tid = t * dimSize, Bid = t * dimSize;				
				top_data[Tid + i] = bottom_data[Bid + i];
			}
		}
	}


	template <typename Dtype>
	void IdentityVectorNoGradientLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const int batSize = (bottom[0]->shape())[0];
		const int dimSize = (bottom[0]->shape())[1]; //2 dimensions
		for (int t = 0; t < batSize; t++) 
		{
			for (int i = 0; i < dimSize; i++) 
			{
				int Tid = t * dimSize, Bid = t * dimSize;				
				bottom_diff[Bid + i] = 0.0; //no gradient
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(IdentityVectorNoGradientLayer);
#endif

	INSTANTIATE_CLASS(IdentityVectorNoGradientLayer);
	REGISTER_LAYER_CLASS(IdentityVectorNoGradient);
}  // namespace caffe

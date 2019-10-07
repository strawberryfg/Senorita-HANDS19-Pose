#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"
namespace caffe {

	template <typename Dtype>
	void TwoVectorSubtractLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		take_abs_ = this->layer_param_.universal_param().take_abs();

	}

	template <typename Dtype>
	void TwoVectorSubtractLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top[0]->Reshape(top_shape);
		dim_size_ = (bottom[0]->shape())[1];
	}


	template <typename Dtype>
	void TwoVectorSubtractLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* a_data = bottom[0]->cpu_data();
		const Dtype* b_data = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++) {
			for (int i = 0; i < dim_size_; i++) {
				int Tid = t * dim_size_, Bid = t * dim_size_;
				top_data[Tid + i] = a_data[Bid + i] - b_data[Bid + i];
				if (take_abs_)
				{
					top_data[Tid + i] = abs(top_data[Tid + i]);
				}
			}
		}
	}


	template <typename Dtype>
	void TwoVectorSubtractLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* a_data = bottom[0]->cpu_data();
		const Dtype* b_data = bottom[1]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* a_diff = bottom[0]->mutable_cpu_diff();
		Dtype* b_diff = bottom[1]->mutable_cpu_diff();
		const int batSize = (bottom[0]->shape())[0];

		for (int t = 0; t < batSize; t++) {
			for (int i = 0; i < dim_size_; i++) {
				int Tid = t * dim_size_, Bid = t * dim_size_;
				a_diff[Bid + i] = top_diff[Tid + i] * 1.0;
				b_diff[Bid + i] = top_diff[Tid + i] * -1.0;
				if (take_abs_)
				{
					if (a_data[Bid + i] < b_data[Bid + i])
					{
						a_diff[Bid + i] = top_diff[Tid + i] * -1.0;
						b_diff[Bid + i] = top_diff[Tid + i] * 1.0;
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(TwoVectorSubtractLayer);
#endif

	INSTANTIATE_CLASS(TwoVectorSubtractLayer);
	REGISTER_LAYER_CLASS(TwoVectorSubtract);
}  // namespace caffe

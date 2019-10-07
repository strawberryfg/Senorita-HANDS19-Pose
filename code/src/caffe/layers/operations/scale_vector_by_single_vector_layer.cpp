#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"
namespace caffe {

	template <typename Dtype>
	void ScaleVectorBySingleVectorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	}

	template <typename Dtype>
	void ScaleVectorBySingleVectorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top[0]->Reshape(top_shape);
		dim_size_ = (bottom[0]->shape())[1];
	}


	template <typename Dtype>
	void ScaleVectorBySingleVectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* scale_data = bottom[1]->cpu_data(); //1-d only
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++) {
			for (int i = 0; i < dim_size_; i++) {
				int Tid = t * dim_size_, Bid = t * dim_size_, Sid = t;
				top_data[Tid + i] = scale_data[Sid] * bottom_data[Bid + i];
			}
		}
	}


	template <typename Dtype>
	void ScaleVectorBySingleVectorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* scale_data = bottom[1]->cpu_data(); //1-d only
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype* scale_diff = bottom[1]->mutable_cpu_diff();
		const int batSize = (bottom[0]->shape())[0];

		for (int t = 0; t < batSize; t++) {
			int Sid = t;
			scale_diff[Sid] = 0.0;
			for (int i = 0; i < dim_size_; i++) {
				int Tid = t * dim_size_, Bid = t * dim_size_;
				bottom_diff[Bid + i] = 1.0 * scale_data[Sid] * top_diff[Tid + i];
				scale_diff[Sid + i] += 1.0 * bottom_data[Bid + i] * top_diff[Tid + i];
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(ScaleVectorBySingleVectorLayer);
#endif

	INSTANTIATE_CLASS(ScaleVectorBySingleVectorLayer);
	REGISTER_LAYER_CLASS(ScaleVectorBySingleVector);
}  // namespace caffe

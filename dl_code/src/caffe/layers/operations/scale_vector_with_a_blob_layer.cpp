#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"
namespace caffe {

    template <typename Dtype>
    void ScaleVectorWithABlobLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        base_scale_ = this->layer_param_.scale_vector_with_a_blob_param().base_scale();
    }

    template <typename Dtype>
    void ScaleVectorWithABlobLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back((bottom[0]->shape())[0]);
        top_shape.push_back((bottom[0]->shape())[1]);
        top[0]->Reshape(top_shape);
        dim_size_ = (bottom[0]->shape())[1];
    }


    template <typename Dtype>
    void ScaleVectorWithABlobLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* scale_data = bottom[1]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int batSize = (bottom[0]->shape())[0];
        for (int t = 0; t < batSize; t++) {
            for (int i = 0; i < dim_size_; i++) {
                int Tid = t * dim_size_, Bid = t * dim_size_;
                top_data[Tid + i] = bottom_data[Bid + i] * (base_scale_ + scale_data[t]);
            }
        }
    }


    template <typename Dtype>
    void ScaleVectorWithABlobLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {

        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* scale_data = bottom[1]->cpu_data();

        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        Dtype* scale_diff = bottom[1]->mutable_cpu_diff();
        const int batSize = (bottom[0]->shape())[0];

        for (int t = 0; t < batSize; t++) {
            scale_diff[t] = 0.0;
            for (int i = 0; i < dim_size_; i++) {
                int Tid = t * dim_size_, Bid = t * dim_size_;
                bottom_diff[Bid + i] = top_diff[Tid + i] * (base_scale_ + scale_data[t]);
                scale_diff[t] += top_diff[Tid + i] * bottom_data[Bid + i];
            }
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(ScaleVectorWithABlobLayer);
#endif

    INSTANTIATE_CLASS(ScaleVectorWithABlobLayer);
    REGISTER_LAYER_CLASS(ScaleVectorWithABlob);
}  // namespace caffe

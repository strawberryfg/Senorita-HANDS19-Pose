#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"


using namespace cv;

namespace caffe {

    template <typename Dtype>
    void DeepHandModelPinholeCameraOriginLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        focusx_ = this->layer_param_.pinhole_camera_origin_param().focusx(); 
        focusy_ = this->layer_param_.pinhole_camera_origin_param().focusy();
        u0offset_ = this->layer_param_.pinhole_camera_origin_param().u0offset();
        v0offset_ = this->layer_param_.pinhole_camera_origin_param().v0offset();
    }
    template <typename Dtype>
    void DeepHandModelPinholeCameraOriginLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        vector<int> top_shape;
        top_shape.push_back((bottom[0]->shape())[0]);
        top_shape.push_back(JointNum * 2);
        
        top[0]->Reshape(top_shape);
    }

    
    template <typename Dtype>
    void DeepHandModelPinholeCameraOriginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        int batSize = (bottom[0]->shape())[0];
        const Dtype* joint3d_data = bottom[0]->cpu_data(); //3d gt for decision of front/back orientation
        Dtype* top_data = top[0]->mutable_cpu_data();
        for (int t = 0; t < batSize; t++) 
        {
            int Bid = t * JointNum * 3;
            int Tid = t * JointNum * 2;
            for (int i = 0; i < JointNum; i++)
            {
                double x = joint3d_data[Bid + i * 3], y = joint3d_data[Bid + i * 3 + 1], z = joint3d_data[Bid + i * 3 + 2];
                double u = focusx_* x / z + u0offset_;
                double v = focusy_ * y / z + v0offset_;
                top_data[Tid + i * 2] = u;
                top_data[Tid + i * 2 + 1] = v;
            }
        }
    }


    template <typename Dtype>
    void DeepHandModelPinholeCameraOriginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }

#ifdef CPU_ONLY
    STUB_GPU(DeepHandModelPinholeCameraOriginLayer);
#endif

    INSTANTIATE_CLASS(DeepHandModelPinholeCameraOriginLayer);
    REGISTER_LAYER_CLASS(DeepHandModelPinholeCameraOrigin);
}

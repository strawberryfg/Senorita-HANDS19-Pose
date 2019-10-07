#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"
namespace caffe {

    template <typename Dtype>
    void DeepHandModelNormalize3DIntoCubiodV2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
		perform_back_prop_ = this->layer_param_.universal_param().perform_back_prop();
	}

    template <typename Dtype>
    void DeepHandModelNormalize3DIntoCubiodV2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const int axis = bottom[0]->CanonicalAxisIndex(
            this->layer_param_.inner_product_param().axis());
        vector<int> top_shape = bottom[0]->shape();
        top_shape.resize(axis + 1);
        top_shape[axis] = JointNum * 3;
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void DeepHandModelNormalize3DIntoCubiodV2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const Dtype* joint_3d_data = bottom[0]->cpu_data();
        const Dtype* x_lb_data = bottom[1]->cpu_data();
        const Dtype* x_ub_data = bottom[2]->cpu_data();
        const Dtype* y_lb_data = bottom[3]->cpu_data();
        const Dtype* y_ub_data = bottom[4]->cpu_data();
        const Dtype* z_lb_data = bottom[5]->cpu_data();
        const Dtype* z_ub_data = bottom[6]->cpu_data();

        Dtype* top_data = top[0]->mutable_cpu_data();
        const int batSize = (bottom[0]->shape())[0];
        for (int t = 0; t < batSize; t++) {
            double x_lb = x_lb_data[t];
            double x_ub = x_ub_data[t];
            double y_lb = y_lb_data[t];
            double y_ub = y_ub_data[t];
            double z_lb = z_lb_data[t];
            double z_ub = z_ub_data[t];


            for (int i = 0; i < JointNum; i++) {
                int Bid = t * JointNum * 3;
                double normalized_x = (joint_3d_data[Bid + i * 3] - x_lb) / (x_ub - x_lb) * 2.0 + (-1);
                double normalized_y = -((joint_3d_data[Bid + i * 3 + 1] - y_lb) / (y_ub - y_lb) * 2.0 + (-1));
                double normalized_z = -((joint_3d_data[Bid + i * 3 + 2] - z_lb) / (z_ub - z_lb) * 2.0 + (-1));
                //NOTE The negative sign
                int Tid = t * JointNum * 3;
                top_data[Tid + i * 3] = normalized_x;
                top_data[Tid + i * 3 + 1] = normalized_y;
                top_data[Tid + i * 3 + 2] = normalized_z;
            }

        }
    }


    template <typename Dtype>
    void DeepHandModelNormalize3DIntoCubiodV2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
		if (perform_back_prop_)
		{
			const Dtype* joint_3d_data = bottom[0]->cpu_data();
			const Dtype* x_lb_data = bottom[1]->cpu_data();
			const Dtype* x_ub_data = bottom[2]->cpu_data();
			const Dtype* y_lb_data = bottom[3]->cpu_data();
			const Dtype* y_ub_data = bottom[4]->cpu_data();
			const Dtype* z_lb_data = bottom[5]->cpu_data();
			const Dtype* z_ub_data = bottom[6]->cpu_data();

			const int batSize = (bottom[0]->shape())[0];
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

			for (int t = 0; t < batSize; t++) 
			{
				double x_lb = x_lb_data[t];
				double x_ub = x_ub_data[t];
				double y_lb = y_lb_data[t];
				double y_ub = y_ub_data[t];
				double z_lb = z_lb_data[t];
				double z_ub = z_ub_data[t];


				for (int i = 0; i < JointNum; i++) 
				{
					int Bid = t * JointNum * 3;
					double normalized_x = (joint_3d_data[Bid + i * 3] - x_lb) / (x_ub - x_lb) * 2.0 + (-1);
					double normalized_y = -((joint_3d_data[Bid + i * 3 + 1] - y_lb) / (y_ub - y_lb) * 2.0 + (-1));
					double normalized_z = -((joint_3d_data[Bid + i * 3 + 2] - z_lb) / (z_ub - z_lb) * 2.0 + (-1));
					//NOTE The negative sign
					int Tid = t * JointNum * 3;
					//top_data[Tid + i * 3] = normalized_x;
					//top_data[Tid + i * 3 + 1] = normalized_y;
					//top_data[Tid + i * 3 + 2] = normalized_z;
					bottom_diff[Bid + i * 3] = top_diff[Tid + i * 3] * 1.0 / (x_ub - x_lb) * 2.0;
					bottom_diff[Bid + i * 3 + 1] = top_diff[Tid + i * 3 + 1] * -((1.0 / (y_ub - y_lb) * 2.0));
					bottom_diff[Bid + i * 3 + 2] = top_diff[Tid + i * 3 + 2] * -((1.0 / (z_ub - z_lb) * 2.0));
				}

			}
		}

    }

#ifdef CPU_ONLY
    STUB_GPU(DeepHandModelNormalize3DIntoCubiodV2Layer);
#endif

    INSTANTIATE_CLASS(DeepHandModelNormalize3DIntoCubiodV2Layer);
    REGISTER_LAYER_CLASS(DeepHandModelNormalize3DIntoCubiodV2);
}  // namespace caffe

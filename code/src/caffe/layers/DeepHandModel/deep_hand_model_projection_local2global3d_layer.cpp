#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"
namespace caffe {

	template <typename Dtype>
	void DeepHandModelProjectionLocal2Global3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		focusx_ = this->layer_param_.deep_hand_model_projection_local2global3d_param().focusx();
		focusy_ = this->layer_param_.deep_hand_model_projection_local2global3d_param().focusy();
		u0offset_ = this->layer_param_.deep_hand_model_projection_local2global3d_param().u0offset();
		v0offset_ = this->layer_param_.deep_hand_model_projection_local2global3d_param().v0offset();

		perform_back_prop_ = this->layer_param_.universal_param().perform_back_prop();
	}

	template <typename Dtype>
	void DeepHandModelProjectionLocal2Global3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		top_shape[axis] = JointNum * 3;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelProjectionLocal2Global3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* joint_2d_data = bottom[0]->cpu_data();
		const Dtype* depth_data = bottom[1]->cpu_data();
		const Dtype* u_lb_data = bottom[2]->cpu_data();
		const Dtype* v_lb_data = bottom[3]->cpu_data();
		const Dtype* z_lb_data = bottom[4]->cpu_data();
		const Dtype* u_ub_data = bottom[5]->cpu_data();
		const Dtype* v_ub_data = bottom[6]->cpu_data();
		const Dtype* z_ub_data = bottom[7]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++) {
			double u_lb = u_lb_data[t];
			double v_lb = v_lb_data[t];
			double z_lb = z_lb_data[t];
			double u_ub = u_ub_data[t];
			double v_ub = v_ub_data[t];
			double z_ub = z_ub_data[t];
			for (int i = 0; i < JointNum; i++) {
				int Bid = t * JointNum * 2;
				double u = joint_2d_data[Bid + i * 2] * (u_ub - u_lb) + u_lb;
				double v = joint_2d_data[Bid + i * 2 + 1] * (v_ub - v_lb) + v_lb;
				int Did = t * JointNum;
				double z = (-depth_data[Did + i] + 1.0) / 2.0 * (z_ub - z_lb) + z_lb;
				//NOTE The negative sign
				int Tid = t * JointNum * 3;
				top_data[Tid + i * 3] = (u - u0offset_) * z / focusx_;
				top_data[Tid + i * 3 + 1] = (v - v0offset_) * z / focusy_;
				top_data[Tid + i * 3 + 2] = z;
			}

		}
	}


	template <typename Dtype>
	void DeepHandModelProjectionLocal2Global3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_2d_diff = bottom[0]->mutable_cpu_diff();
		Dtype* bottom_depth_diff = bottom[1]->mutable_cpu_diff();

		const int batSize = (bottom[0]->shape())[0];

		const Dtype* joint_2d_data = bottom[0]->cpu_data();
		const Dtype* depth_data = bottom[1]->cpu_data();
		const Dtype* u_lb_data = bottom[2]->cpu_data();
		const Dtype* v_lb_data = bottom[3]->cpu_data();
		const Dtype* z_lb_data = bottom[4]->cpu_data();
		const Dtype* u_ub_data = bottom[5]->cpu_data();
		const Dtype* v_ub_data = bottom[6]->cpu_data();
		const Dtype* z_ub_data = bottom[7]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		if (!perform_back_prop_) return;
		for (int t = 0; t < batSize; t++) 
		{
			double u_lb = u_lb_data[t];
			double v_lb = v_lb_data[t];
			double z_lb = z_lb_data[t];
			double u_ub = u_ub_data[t];
			double v_ub = v_ub_data[t];
			double z_ub = z_ub_data[t];
			for (int i = 0; i < JointNum; i++) 
			{
				int Bid = t * JointNum * 2;
				double u = joint_2d_data[Bid + i * 2] * (u_ub - u_lb) + u_lb;
				double v = joint_2d_data[Bid + i * 2 + 1] * (v_ub - v_lb) + v_lb;
				int Did = t * JointNum;
				double z = (-depth_data[Did + i] + 1.0) / 2.0 * (z_ub - z_lb) + z_lb;
				//NOTE The negative sign
				int Tid = t * JointNum * 3;
				//top_data[Tid + i * 3] = (u - u0offset_) * z / focusx_;
				//top_data[Tid + i * 3 + 1] = (v - v0offset_) * z / focusy_;
				//top_data[Tid + i * 3 + 2] = z;
				bottom_2d_diff[Bid + i * 2] = top_diff[Tid + i * 3] * z / focusx_ * 1.0 * (u_ub - u_lb);
				bottom_2d_diff[Bid + i * 2 + 1] = top_diff[Tid + i * 3 + 1] * z / focusy_ * 1.0 * (v_ub - v_lb);
				bottom_depth_diff[Did + i] = top_diff[Tid + i * 3 + 2] * 1.0 / 2.0 * (z_ub - z_lb) * -1.0;
			}

		}

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelProjectionLocal2Global3DLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelProjectionLocal2Global3DLayer);
	REGISTER_LAYER_CLASS(DeepHandModelProjectionLocal2Global3D);
}  // namespace caffe

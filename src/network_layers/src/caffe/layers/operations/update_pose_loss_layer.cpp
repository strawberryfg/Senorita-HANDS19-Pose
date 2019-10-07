#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"

namespace caffe {

	template <typename Dtype>
	void UpdatePoseLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
		}
		lambda_ = this->layer_param_.update_pose_param().lambda();
	}

	template <typename Dtype>
	void UpdatePoseLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);
		top[0]->Reshape(loss_shape);
		joint_num_ = (bottom[0]->shape())[1] / 3;
	}


	template <typename Dtype>
	void UpdatePoseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* gt_data = bottom[0]->cpu_data();
		const Dtype* pred_data = bottom[1]->cpu_data();
		const Dtype* delta_data = bottom[2]->cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		
		Dtype loss = 0;
		for (int t = 0; t < batSize; t++)
		{
			//For each joint each dimension
			int Jid = t * joint_num_ * 3;
			for (int i = 0; i < joint_num_ * 3; i++)
			{
				double q = 0.0;
				//(gt_i - (pref + delta)_i) ^ 2 - lambda * (gt_i - pref_i) ^ 2
				q = pow(gt_data[Jid + i] - (pred_data[Jid + i] + delta_data[Jid + i]), 2) - lambda_ * pow(gt_data[Jid + i] - pred_data[Jid + i], 2);
				q = max(0.0, q);
				loss += q;
			}
		}

		top[0]->mutable_cpu_data()[0] = loss / batSize / joint_num_ / 3;
		
	}


	template <typename Dtype>
	void UpdatePoseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const int batSize = (bottom[0]->shape())[0];

		const Dtype* gt_data = bottom[0]->cpu_data();
		const Dtype* pred_data = bottom[1]->cpu_data();
		const Dtype* delta_data = bottom[2]->cpu_data();
		Dtype top_diff = top[0]->cpu_diff()[0] / batSize / joint_num_ / 3;
		Dtype* delta_diff = bottom[2]->mutable_cpu_diff();
 		

		for (int t = 0; t < batSize; t++)
		{
			//For each joint each dimension
			int Jid = t * joint_num_ * 3;
			int Bid = t * joint_num_ * 3;
			for (int i = 0; i < joint_num_ * 3; i++)
			{
				double q = 0.0;
				//(gt_i - (pref + delta)_i) ^ 2 - lambda * (gt_i - pref_i) ^ 2
				q = pow(gt_data[Jid + i] - (pred_data[Jid + i] + delta_data[Jid + i]), 2) - lambda_ * pow(gt_data[Jid + i] - pred_data[Jid + i], 2);
				q = max(0.0, q);
				if (q == 0)
				{
					delta_diff[Bid + i] = 0.0;
				}
				else
				{
					delta_diff[Bid + i] = top_diff * 2 * (gt_data[Jid + i] - (pred_data[Jid + i] + delta_data[Jid + i])) * -1.0;
				}
				
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(UpdatePoseLossLayer);
#endif

	INSTANTIATE_CLASS(UpdatePoseLossLayer);
	REGISTER_LAYER_CLASS(UpdatePoseLoss);
}  // namespace caffe

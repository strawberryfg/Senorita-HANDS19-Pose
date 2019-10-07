#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"

namespace caffe {

	template <typename Dtype>
	void JointAbsLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
		}

	}


	template <typename Dtype>
	void JointAbsLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);
		top[0]->Reshape(loss_shape);

		joint_num_ = (bottom[0]->shape())[1] / 3;
	}

	template <typename Dtype>
	void JointAbsLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		
		Dtype loss = 0;

		for (int t = 0; t < batSize; t++)
		{
			for (int i = 0; i < joint_num_; i++)
			{
				int Bid = t * joint_num_ * 3 + i * 3;
				Dtype cur_loss = 0.0;
				for (int j = 0; j < 2; j++) cur_loss += bottom_data[Bid + j];
				cur_loss += bottom_data[Bid + 2];
				cur_loss /= 3.0;
				loss += cur_loss;
			}
		}
		top[0]->mutable_cpu_data()[0] = loss / double(batSize) / joint_num_;

	}


	template <typename Dtype>
	void JointAbsLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		
		Dtype top_diff = top[0]->cpu_diff()[0] / batSize / joint_num_;
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) {
				int Bid = t * joint_num_ * 3;
				for (int i = 0; i < joint_num_ * 3; i++) bottom_diff[Bid + i] = 0;
				for (int i = 0; i < joint_num_; i++) {
					for (int j = 0; j < 2; j++)
					{
						Bid = t * joint_num_ * 3 + i * 3;
						bottom_diff[Bid + j] = top_diff * 1.0 / 3.0;
					}

					Bid = t * joint_num_ * 3 + i * 3;
					bottom_diff[Bid + 2] = top_diff * 1.0 / 3.0;
					
				}
			}
		}

	}

#ifdef CPU_ONLY
	STUB_GPU(JointAbsLossLayer);
#endif

	INSTANTIATE_CLASS(JointAbsLossLayer);
	REGISTER_LAYER_CLASS(JointAbsLoss);

}  // namespace caffe

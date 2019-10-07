#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"

namespace caffe {

	template <typename Dtype>
	void BoundPoseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
		upper_bound_ = this->layer_param_.bound_pose_param().upper_bound();
	}

	template <typename Dtype>
	void BoundPoseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top[0]->Reshape(top_shape);

		joint_num_ = (bottom[0]->shape())[1] / 3;
	}


	template <typename Dtype>
	void BoundPoseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* pose_data = bottom[0]->cpu_data();
		
		const int batSize = (bottom[0]->shape())[0];
		Dtype* top_data = top[0]->mutable_cpu_data();

		for (int t = 0; t < batSize; t++)
		{
			//For each joint 
			int Jid = t * joint_num_ * 3;
			int Tid = t * joint_num_ * 3;
			for (int j = 0; j < joint_num_; j++)
			{
				double delta_norm = sqrt(pow(pose_data[Jid + j * 3], 2) + pow(pose_data[Jid + j * 3 + 1], 2) + pow(pose_data[Jid + j * 3 + 2], 2));
				//min(K, ||delta||) * delta / ||delta||
				if (upper_bound_ - delta_norm > 1e-6) // ||delta|| < upper_bound_
				{
					for (int l = 0; l < 3; l++)
					{
						top_data[Tid + j * 3 + l] = pose_data[Jid + j * 3 + l];
					}
				}
				else // ||delta|| > upper_bound_
				{
					for (int l = 0; l < 3; l++)
					{
						//Move along this direction upper_bound_ unit vector
						top_data[Tid + j * 3 + l] = upper_bound_ * pose_data[Jid + j * 3 + l] / delta_norm;
					}
				}
			}
		}

		
	}


	template <typename Dtype>
	void BoundPoseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const int batSize = (bottom[0]->shape())[0];

		const Dtype* pose_data = bottom[0]->cpu_data();
		
		const Dtype* top_diff = top[0]->cpu_diff();
		
		Dtype* delta_diff = bottom[0]->mutable_cpu_diff();


		for (int t = 0; t < batSize; t++)
		{
			//For each joint 
			int Jid = t * joint_num_ * 3;
			int Tid = t * joint_num_ * 3;
			for (int j = 0; j < joint_num_; j++)
			{
				double delta_norm = sqrt(pow(pose_data[Jid + j * 3], 2) + pow(pose_data[Jid + j * 3 + 1], 2) + pow(pose_data[Jid + j * 3 + 2], 2));
				//min(K, ||delta||) * delta / ||delta||
				if (upper_bound_ - delta_norm > 1e-6) // ||delta|| < upper_bound_
				{
					for (int l = 0; l < 3; l++)
					{
						delta_diff[Jid + j * 3 + l] = top_diff[Tid + j * 3 + l] * 1.0;
					}
				}
				else // ||delta|| > upper_bound_
				{
					for (int l = 0; l < 3; l++)
					{
						//Move along this direction upper_bound_ unit vector
						delta_diff[Jid + j * 3 + l] = top_diff[Tid + j * 3 + l] * upper_bound_ * 1.0 / pow(delta_norm, 2) * (1.0 * delta_norm - pose_data[Jid + j * 3 + l] * 1.0 / (2.0 * delta_norm) * 2 * pose_data[Jid + j * 3 + l]);

						//top_data[Tid + j * 3 + l] = upper_bound_ * pose_data[Jid + j * 3 + l] / delta_norm;
					}
			}
		}
	}
	}

#ifdef CPU_ONLY
	STUB_GPU(BoundPoseLayer);
#endif

	INSTANTIATE_CLASS(BoundPoseLayer);
	REGISTER_LAYER_CLASS(BoundPose);
}  // namespace caffe

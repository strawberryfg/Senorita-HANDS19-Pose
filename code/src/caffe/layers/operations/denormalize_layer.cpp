#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"
namespace caffe {

	template <typename Dtype>
	void DenormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		mul_hyp_ = this->layer_param_.universal_param().mul_hyp();
		sub_root_ = this->layer_param_.universal_param().sub_root();
	}

	template <typename Dtype>
	void DenormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top[0]->Reshape(top_shape);
	}


	template <typename Dtype>
	void DenormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* std_data = bottom[1]->cpu_data(); //standard deviation
		const Dtype* avg_data = bottom[2]->cpu_data(); //mean value average
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		const int dimSize = (bottom[0]->shape())[1]; //2 dimensions
		int ndim_stats = (bottom[1]->shape())[1]; //std avg stats dimension size
		int loop_num = dimSize / ndim_stats;
		for (int t = 0; t < batSize; t++)
		{
			for (int j = 0; j < loop_num; j++)
			{

				double root[111];
				//root is 0 0 * 3 + k (3 dimensions)
				int Tid = t * dimSize, Bid = t * dimSize, Sid = t * ndim_stats;

				for (int k = 0; k < 3; k++) root[k] = bottom_data[Bid + j * ndim_stats + k];

				for (int k = 0; k < ndim_stats; k++)
				{
					double std_value = std_data[Sid + k];
					double avg_value = avg_data[Sid + k];
					double v;
					if (sub_root_)
					{
						//sub 0, 1, 2
						v = bottom_data[Bid + j * ndim_stats + k] - root[k % 3]; // bottom_data[Bid + j * ndim_stats + (k % 3)]);
					}
					else
					{
						v = bottom_data[Bid + j * ndim_stats + k];
					}
					if (mul_hyp_ && ((k / 3 == 0) || (k / 3 == 7)))
					{
						//printf("%d ", k);
						//root 0 spine 7
						//printf("%d %d %6.2f\n", j, k, 0.0);
						top_data[Tid + j * ndim_stats + k] = 0.0;
					}
					else
					{
						if (std_value == 0.0) top_data[Tid + j * ndim_stats + k] = 0.0;
						//printf("%d %d %6.2f\n", j, k, (v - avg_value) / std_value);
						else top_data[Tid + j * ndim_stats + k] = (v - avg_value) / std_value; //multiplied by the std of that dimension and plus the average mean value				
					}
				}
			}
		}
	}


	template <typename Dtype>
	void DenormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* std_data = bottom[1]->cpu_data(); //standard deviation
		const Dtype* avg_data = bottom[2]->cpu_data(); //mean value average
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const int batSize = (bottom[0]->shape())[0];
		const int dimSize = (bottom[0]->shape())[1]; //2 dimensions
		int ndim_stats = (bottom[1]->shape())[1]; //std avg stats dimension size
		int loop_num = dimSize / ndim_stats;
		for (int t = 0; t < batSize; t++)
		{
			for (int j = 0; j < loop_num; j++)
			{
				for (int k = 0; k < ndim_stats; k++)
				{
					int Tid = t * dimSize, Bid = t * dimSize, Sid = t * ndim_stats;
					double std_value = std_data[Sid + k];
					double avg_value = avg_data[Sid + k];
					if (mul_hyp_ && ((k / 3 == 0) || (k / 3 == 7)))
					{
						//root 0 spine 7 no norm or de norm
						bottom_diff[Bid + j * ndim_stats + k] = 0.0;
					}
					else
					{
						if (std_value == 0.0) bottom_diff[Bid + j * ndim_stats + k] = 0.0;
						else bottom_diff[Bid + j * ndim_stats + k] = top_diff[Tid + j * ndim_stats + k] / std_value;
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DenormalizeLayer);
#endif

	INSTANTIATE_CLASS(DenormalizeLayer);
	REGISTER_LAYER_CLASS(Denormalize);
}  // namespace caffe


#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenProjImgPlaneMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//depth_dims_ = this->layer_param_.gen_3d_skeleton_map_param().depth_dims();
		//map_size_ = this->layer_param_.gen_3d_skeleton_map_param().map_size();
		perform_backprop_ = this->layer_param_.gen_3d_skeleton_map_param().perform_backprop();

		squeeze_axis_ = this->layer_param_.gen_3d_skeleton_map_param().squeeze_axis();


		dim_lb_ = this->layer_param_.deep_hand_model_integral_vector_param().dim_lb();
		dim_ub_ = this->layer_param_.deep_hand_model_integral_vector_param().dim_ub();
	}
	template <typename Dtype>
	void DeepHandModelGenProjImgPlaneMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top_shape.push_back((bottom[0]->shape())[2]);
		top_shape.push_back((bottom[0]->shape())[3]);
		//shape[1] should be = [2] = [3]
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGenProjImgPlaneMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		int batSize = (bottom[0]->shape())[0];
		int C = (bottom[0]->shape())[1];
		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //bottom is a 3D cube

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {

			int Bid = t * C * H * W;
			int Tid;

			if (squeeze_axis_ == 1) //along Z XY
			{
				Tid = t * H * W;
				for (int y = 0; y < H; y++)
				{
					for (int x = 0; x < W; x++)
					{
						top_data[Tid + y * W + x] = 0.0;
						for (int z = 0; z < C; z++)
						{
							double position = (dim_ub_ - dim_lb_) / (C) * (z + 0.5) + dim_lb_;
							double prob = bottom_data[Bid + z * H * W + y * W + x];
							top_data[Tid + y * W + x] += prob * position;
						}
					}
				}
			} 
			else if (squeeze_axis_ == 2) //along Y ZX
			{
				Tid = t * W * C;
				for (int x = 0; x < W; x++)
				{
					for (int z = 0; z < C; z++)
					{
						top_data[Tid + x * C + z] = 0.0;
						for (int y = 0; y < H; y++)
						{
							double position = (dim_ub_ - dim_lb_) / (H) * (y + 0.5) + dim_lb_;
							double prob = bottom_data[Bid + z * H * W + y * W + x];
							top_data[Tid + x * C + z] += prob * position;
						}
					}
				}
			}
			else if (squeeze_axis_ == 3) //along X ZY
			{
				Tid = t * H * C;
				for (int y = 0; y < H; y++)
				{
					for (int z = 0; z < C; z++)
					{
						top_data[Tid + y * C + z] = 0.0;
						for (int x = 0; x < W; x++)
						{
							double position = (dim_ub_ - dim_lb_) / (W) * (x + 0.5) + dim_lb_;
							double prob = bottom_data[Bid + z * H * W + y * W + x];
							top_data[Tid + y * C + z] += prob * position;
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelGenProjImgPlaneMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int batSize = (bottom[0]->shape())[0];
		int C = (bottom[0]->shape())[1];
		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];

		if (!perform_backprop_) return;
		for (int t = 0; t < batSize; t++) 
		{

			int Bid = t * C * H * W;
			int Tid;
			for (int j = 0; j < C * H * W; j++) bottom_diff[Bid + j] = 0.0;
			if (squeeze_axis_ == 1) //along Z XY
			{
				Tid = t * H * W;
				for (int y = 0; y < H; y++)
				{
					for (int x = 0; x < W; x++)
					{
						//top_data[Tid + y * W + x] = 0.0;
						for (int z = 0; z < C; z++)
						{ 
							double position = (dim_ub_ - dim_lb_) / (C) * (z + 0.5) + dim_lb_;
							double prob = bottom_data[Bid + z * H * W + y * W + x];
							//top_data[Tid + y * W + x] += prob * position;
							bottom_diff[Bid + z * H * W + y * W + x] += top_diff[Tid + y * W + x] * position;
						}
					}
				}
			}
			else if (squeeze_axis_ == 2) //along Y ZX
			{
				Tid = t * W * C;
				for (int x = 0; x < W; x++)
				{
					for (int z = 0; z < C; z++)
					{
						//top_data[Tid + x * C + z] = 0.0;
						for (int y = 0; y < H; y++)
						{
							double position = (dim_ub_ - dim_lb_) / (H) * (y + 0.5) + dim_lb_;
							double prob = bottom_data[Bid + z * H * W + y * W + x];
							//top_data[Tid + x * C + z] += prob * position;
							bottom_diff[Bid + z * H * W + y * W + x] += top_diff[Tid + x * C + z] * position;
						}
					}
				}
			}
			else if (squeeze_axis_ == 3) //along X ZY
			{
				Tid = t * H * C;
				for (int y = 0; y < H; y++)
				{
					for (int z = 0; z < C; z++)
					{
						//top_data[Tid + y * C + z] = 0.0;
						for (int x = 0; x < W; x++)
						{
							double position = (dim_ub_ - dim_lb_) / (W) * (x + 0.5) + dim_lb_;
							double prob = bottom_data[Bid + z * H * W + y * W + x];
							//top_data[Tid + y * C + z] += prob * position;
							bottom_diff[Bid + z * H * W + y * W + x] += top_diff[Tid + y * C + z] * position;
						}
					}
				}
			}
		}

		
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenProjImgPlaneMapLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenProjImgPlaneMapLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenProjImgPlaneMap);
}

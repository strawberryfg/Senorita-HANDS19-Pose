
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"
#include <cmath>
namespace caffe {

	template <typename Dtype>
	void DeepHandModelRotPointCloudLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//depth_dims_ = this->layer_param_.gen_3d_skeleton_map_param().depth_dims();
		//map_size_ = this->layer_param_.gen_3d_skeleton_map_param().map_size();
		perform_backprop_ = this->layer_param_.gen_3d_skeleton_map_param().perform_backprop();

		rot_axis_ = this->layer_param_.gen_3d_skeleton_map_param().rot_axis();
		rot_degree_ = this->layer_param_.gen_3d_skeleton_map_param().rot_degree();
		ctheta_ = cos(rot_degree_ / 180.0 * PI);
		stheta_ = sin(rot_degree_ / 180.0 * PI);
	}
	template <typename Dtype>
	void DeepHandModelRotPointCloudLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		bone_num_ = (bottom[0]->shape())[1] / (bottom[0]->shape())[2];

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top_shape.push_back((bottom[0]->shape())[2]);
		top_shape.push_back((bottom[0]->shape())[3]);
		//shape[2] should be = [3]
		
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelRotPointCloudLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		int batSize = (bottom[0]->shape())[0];
		int C = (bottom[0]->shape())[1] / bone_num_; //=(bottom[0]->shape())[2]
		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //bottom is a 3D cube

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) 
		{
			int Bid = t * bone_num_ * C * H * W;
			int Tid = t * bone_num_ * C * H * W;
			for (int j = 0; j < bone_num_ * C * H * W; j++) top_data[Tid + j] = 0.0;
			for (int b = 0; b < bone_num_; b++)
			{
				//cout << "FP: " << t << " " << b << "\n";
				for (int z = 0; z < C; z++)
				{
					for (int y = 0; y < H; y++)
					{
						for (int x = 0; x < W; x++)
						{
							double ori_x = x - W / 2;
							double ori_y = y - H / 2;
							double ori_z = z - C / 2;
							int new_x, new_y, new_z;
							//Euclidean distance matrix 
							//Rotate camera around hand local coordiante axis Y, X theta degree
							//amounts to rotate hand point clouds around camera local coordinate axis Y, X -theta degree
							if (rot_axis_ == 2) //rot around axis Y
							{
								new_x = int(ctheta_ * ori_x + stheta_ * ori_z);
								new_y = int(ori_y);
								new_z = int(-stheta_ * ori_x + ctheta_ * ori_z);
							}
							else if (rot_axis_ == 3) // rot around axis X
							{
								new_x = int(ori_x);
								new_y = int(ctheta_ * ori_y - stheta_ * ori_z);
								new_z = int(stheta_ * ori_y + ctheta_ * ori_z);
							}
							//boundary check 
							if (new_x >= -W / 2 && new_x < W / 2 &&
								new_y >= -H / 2 && new_y < H / 2 &&
								new_z >= -C / 2 && new_z < C / 2)
							{
								//cout << "t: " << t << " b: " << b << " z: " << z << " y: " << y << " x: " << x << " ori_x: " << ori_x << " ori_y: " << ori_y << " ori_z: " << ori_z << " new_x: " << new_x << " new_y: " << new_y << " new_z: " << new_z << "\n";
								new_x = new_x - (-W / 2);
								new_y = new_y - (-H / 2);
								new_z = new_z - (-C / 2);
								top_data[Tid + b * C * H * W + new_z * H * W + new_y * W + new_x] = bottom_data[Bid + b * C * H * W + z * H * W + y * W + x];
							}
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelRotPointCloudLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int batSize = (bottom[0]->shape())[0];
		int C = (bottom[0]->shape())[1] / bone_num_;
		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];

		if (!perform_backprop_) return;
		
		for (int t = 0; t < batSize; t++)
		{
			int Bid = t * bone_num_ * C * H * W;
			int Tid = t * bone_num_ * C * H * W;
			for (int j = 0; j < bone_num_ * C * H * W; j++) bottom_diff[Bid + j] = 0.0;
			for (int b = 0; b < bone_num_; b++)
			{
				//cout << "BP: " << t << " " << b << "\n";
				for (int z = 0; z < C; z++)
				{
					for (int y = 0; y < H; y++)
					{
						for (int x = 0; x < W; x++)
						{
							double ori_x = x - W / 2;
							double ori_y = y - H / 2;
							double ori_z = z - C / 2;
							int new_x, new_y, new_z;
							//Euclidean distance matrix 
							//Rotate camera around hand local coordiante axis Y, X theta degree
							//amounts to rotate hand point clouds around camera local coordinate axis Y, X -theta degree
							if (rot_axis_ == 2) //rot around axis Y
							{
								new_x = int(ctheta_ * ori_x + stheta_ * ori_z);
								new_y = int(ori_y);
								new_z = int(-stheta_ * ori_x + ctheta_ * ori_z);
							}
							else if (rot_axis_ == 3) // rot around axis X
							{
								new_x = int(ori_x);
								new_y = int(ctheta_ * ori_y - stheta_ * ori_z);
								new_z = int(stheta_ * ori_y + ctheta_ * ori_z);
							}
							//boundary check 
							if (new_x >= -W / 2 && new_x < W / 2 &&
								new_y >= -H / 2 && new_y < H / 2 &&
								new_z >= -C / 2 && new_z < C / 2)
							{
								new_x = new_x - (-W / 2);
								new_y = new_y - (-H / 2);
								new_z = new_z - (-C / 2);
								//top_data[Tid + new_z * H * W + new_y * W + new_x] = bottom_data[Bid + z * H * W + y * W + x];
								bottom_diff[Bid + b * C * H * W + z * H * W + y * W + x] = top_diff[Tid + b * C * H * W + new_z * H * W + new_y * W + new_x] * 1.0;
							}
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelRotPointCloudLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelRotPointCloudLayer);
	REGISTER_LAYER_CLASS(DeepHandModelRotPointCloud);
}

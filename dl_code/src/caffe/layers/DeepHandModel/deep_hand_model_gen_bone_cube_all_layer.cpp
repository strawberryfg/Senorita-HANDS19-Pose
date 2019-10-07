#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"

using namespace cv;
//Generate bone patch from gt 2d
//output blob image data in [0, 1]
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenBoneCubeAllLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		img_size_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().img_size();
		focusx_ = this->layer_param_.pinhole_camera_origin_param().focusx();
		focusy_ = this->layer_param_.pinhole_camera_origin_param().focusy();
		u0offset_ = this->layer_param_.pinhole_camera_origin_param().u0offset();
		v0offset_ = this->layer_param_.pinhole_camera_origin_param().v0offset();

		depth_dims_ = this->layer_param_.gen_3d_skeleton_map_param().depth_dims();
		map_size_ = this->layer_param_.gen_3d_skeleton_map_param().map_size();

	}

	template <typename Dtype>
	void DeepHandModelGenBoneCubeAllLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		channel_num_ = (bottom[0]->shape())[1]; //if depth voxel grid channel num = depth_dims other wise BoneNum * depth_dims_

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(BoneNum * depth_dims_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);

		
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGenBoneCubeAllLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* patch_bbx_data = bottom[1]->cpu_data();
		
		const Dtype* bbx_x1_data = bottom[2]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[3]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[4]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[5]->cpu_data(); //bbx_y2

		const Dtype* x_lb_data = bottom[6]->cpu_data(); //min_x of cube
		const Dtype* x_ub_data = bottom[7]->cpu_data(); //max_x of cube
		const Dtype* y_lb_data = bottom[8]->cpu_data(); //min_y of cube
		const Dtype* y_ub_data = bottom[9]->cpu_data(); //max_y of cube
		const Dtype* z_lb_data = bottom[10]->cpu_data(); //min_z of cube
		const Dtype* z_ub_data = bottom[11]->cpu_data(); //max_z of cube


		Dtype* top_data = top[0]->mutable_cpu_data();

		
		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++)
		{
			int Tid = t * BoneNum * depth_dims_ * map_size_ * map_size_;
			int Bid = t * channel_num_ * map_size_ * map_size_;
			int Pid = t * BoneNum * 4; //patch bbx
			//for (int j = 0; j < BoneNum; j++) 
			//clear
			for (int j = 0; j < BoneNum * depth_dims_ * map_size_ * map_size_; j++) top_data[Tid + j] = 0.0;
			double bbx_x1 = bbx_x1_data[t];
			double bbx_y1 = bbx_y1_data[t];
			double bbx_x2 = bbx_x2_data[t];
			double bbx_y2 = bbx_y2_data[t];
			double x_lb = x_lb_data[t];
			double x_ub = x_ub_data[t];
			double y_lb = y_lb_data[t];
			double y_ub = y_ub_data[t];
			double z_lb = z_lb_data[t];
			double z_ub = z_ub_data[t];

			for (int k = 0; k < depth_dims_; k++)
			{
				for (int row = 0; row < map_size_; row++)
				{
					for (int col = 0; col < map_size_; col++)
					{
						//True 3D bounding box
						double global_x = double(col) / double(map_size_) * (x_ub - x_lb) + x_lb;
						double global_y = double(row) / double(map_size_) * (y_ub - y_lb) + y_lb;
						double global_z = double(k) / double(depth_dims_) * (z_ub - z_lb) + z_lb;

						//Project to 2D
						double global_u = focusx_ * global_x / global_z + u0offset_;
						double global_v = focusy_ * global_y / global_z + v0offset_;

						//Local [0, 1] projection * depth_size_
						double local_u = (global_u - bbx_x1) / (bbx_x2 - bbx_x1) * img_size_;
						double local_v = (global_v - bbx_y1) / (bbx_y2 - bbx_y1) * img_size_;

						int t_col = min(max(0, (int)local_u), img_size_ - 1);
						int t_row = min(max(0, (int)local_v), img_size_ - 1);

						//Check if projected 2d falls in the bone patch
						for (int j = 0; j < BoneNum; j++)
						{
							double bone_patch_bbx_x1 = patch_bbx_data[Pid + j * 4];
							double bone_patch_bbx_y1 = patch_bbx_data[Pid + j * 4 + 1];
							double bone_patch_bbx_x2 = patch_bbx_data[Pid + j * 4 + 2];
							double bone_patch_bbx_y2 = patch_bbx_data[Pid + j * 4 + 3];

							if (t_col >= bone_patch_bbx_x1 && t_col < bone_patch_bbx_x2 &&
								t_row >= bone_patch_bbx_y1 && t_row < bone_patch_bbx_y2)
							{
								//FALLS WITHIN THE BONE PATCH BBX
								if (channel_num_ / depth_dims_ != 1) top_data[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = bottom_data[Bid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col];
								else top_data[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = bottom_data[Bid + k * map_size_ * map_size_ + row * map_size_ + col];
							}
						}						
					}
				}
			}
		}
	}


	template <typename Dtype>
	void DeepHandModelGenBoneCubeAllLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {


	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenBoneCubeAllLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenBoneCubeAllLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenBoneCubeAll);
}  // namespace caffe

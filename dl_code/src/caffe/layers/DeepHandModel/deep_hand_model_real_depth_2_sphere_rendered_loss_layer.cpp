
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DeepHandModelRealDepthMap2SphereRenderedLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_size_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().depth_size();


		focusx_ = this->layer_param_.pinhole_camera_origin_param().focusx();
		focusy_ = this->layer_param_.pinhole_camera_origin_param().focusy();
		u0offset_ = this->layer_param_.pinhole_camera_origin_param().u0offset();
		v0offset_ = this->layer_param_.pinhole_camera_origin_param().v0offset();

		if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
		}
	}
	template <typename Dtype>
	void DeepHandModelRealDepthMap2SphereRenderedLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> loss_shape(0);
		top[0]->Reshape(loss_shape);

		joint_num_ = (bottom[0]->shape())[1] / 3;
		valid_nonzero_depth_point_num_ = 0;
	}

	template <typename Dtype>
	void DeepHandModelRealDepthMap2SphereRenderedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* sphere_center_data = bottom[0]->cpu_data(); //sphere center e.g. 41 spheres
		const Dtype* bbx_x1_data = bottom[1]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[2]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[3]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[4]->cpu_data(); //bbx_y2
		const Dtype* avgz_data = bottom[5]->cpu_data(); //z of centroid
		const Dtype* sphere_radius_data = bottom[6]->cpu_data();
		const Dtype* depth_map_data = bottom[7]->cpu_data(); //real depth map input

		Dtype loss = 0;
		for (int t = 0; t < batSize; t++)
		{
			int Gid = t * joint_num_ * 3;
			int Rid = t * joint_num_;
			int Did = t * 1 * depth_size_ * depth_size_;

			double bbx_x1 = bbx_x1_data[t];
			double bbx_y1 = bbx_y1_data[t];
			double bbx_x2 = bbx_x2_data[t];
			double bbx_y2 = bbx_y2_data[t];

			double avg_d = avgz_data[t];

			for (int row = 0; row < depth_size_; row++)
			{
				for (int col = 0; col < depth_size_; col++)
				{
					double global_u = col / double(depth_size_) * (bbx_x2 - bbx_x1) + bbx_x1;
					double global_v = row / double(depth_size_) * (bbx_y2 - bbx_y1) + bbx_y1;

					//int cur_col = (-(double(min_point_p_depth - avg_d) / double(100.0)) + 1.0) / 2.0 * 255;

					//FIND THE DEPTH VALUE FOR EACH PIXEL IN ORIGINAL DEPTH IMAGE INPUT
					if (depth_map_data[Did + row * depth_size_ + col] > 1e-6) //NOT A BLACK INVALID PIXEL
					{
						valid_nonzero_depth_point_num_++;

						double cur_d = -(depth_map_data[Did + row * depth_size_ + col] * 2.0 - 1.0) * 100.0 + avg_d;
						double global_z = cur_d;

						double global_x = (global_u - u0offset_) / focusx_ * global_z;
						double global_y = (global_v - v0offset_) / focusy_ * global_z;

						double min_dist_2_surface = 15000000;
						int min_dist_2_surface_id = -1;

						//ICP
						//Minimize the distance between every point p from depth map input, and its projection on
						//the estimated hand model sphere surface

						//NOW FIND THE CLOSEST DISTANCE OF DEPTH POINT -> PROJECTED SPHERE SURFACE
						for (int j = 0; j < joint_num_; j++)
						{
							double sphere_center_x = sphere_center_data[Gid + j * 3];
							double sphere_center_y = sphere_center_data[Gid + j * 3 + 1];
							double sphere_center_z = sphere_center_data[Gid + j * 3 + 2];

							double sphere_radius = sphere_radius_data[Rid + j];

							//Dist of point p in depth map (row, col) -> sphere center point
							double dist_2_sphere_center = sqrt(pow(global_x - sphere_center_x, 2) + pow(global_y - sphere_center_y, 2) + pow(global_z - sphere_center_z, 2));
							double dist_2_surface = fabs(dist_2_sphere_center - sphere_radius);
							//IF < SPHERE_RADIUS MEANS P FALLS WITHIN THE SPHERE
							//OTHERWISE P IS OUTSIDE THE SPHERE

							if (min_dist_2_surface - dist_2_surface > 1e-6)
							{
								min_dist_2_surface = dist_2_surface;
								min_dist_2_surface_id = j;
							}
						}

						//NOT BACKGROUND PIXEL 
						if (min_dist_2_surface_id != -1)
						{
							loss += min_dist_2_surface;
						}
					}
				}
			}
		}

		top[0]->mutable_cpu_data()[0] = loss / double(batSize) / double(valid_nonzero_depth_point_num_ + 1);
		//Glossed over all valid non-black pixel in real depth image input
	}

	template <typename Dtype>
	void DeepHandModelRealDepthMap2SphereRenderedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* sphere_center_data = bottom[0]->cpu_data(); //sphere center e.g. 41 spheres
		const Dtype* bbx_x1_data = bottom[1]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[2]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[3]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[4]->cpu_data(); //bbx_y2
		const Dtype* avgz_data = bottom[5]->cpu_data(); //z of centroid
		const Dtype* sphere_radius_data = bottom[6]->cpu_data();
		const Dtype* depth_map_data = bottom[7]->cpu_data(); //real depth map input

		Dtype top_diff = top[0]->cpu_diff()[0] / double(batSize) / double(valid_nonzero_depth_point_num_ + 1);

		Dtype* bottom_3d_diff = bottom[0]->mutable_cpu_diff(); //sphere center gradient

		for (int t = 0; t < batSize; t++)
		{
			int Gid = t * joint_num_ * 3;
			int Rid = t * joint_num_;
			int Tid = t * 1 * depth_size_ * depth_size_;
			int Did = t * 1 * depth_size_ * depth_size_;

			//clear
			for (int j = 0; j < joint_num_ * 3; j++) bottom_3d_diff[Gid + j] = 0.0;
			double bbx_x1 = bbx_x1_data[t];
			double bbx_y1 = bbx_y1_data[t];
			double bbx_x2 = bbx_x2_data[t];
			double bbx_y2 = bbx_y2_data[t];

			double avg_d = avgz_data[t];

			for (int row = 0; row < depth_size_; row++)
			{
				for (int col = 0; col < depth_size_; col++)
				{
					double global_u = col / double(depth_size_) * (bbx_x2 - bbx_x1) + bbx_x1;
					double global_v = row / double(depth_size_) * (bbx_y2 - bbx_y1) + bbx_y1;

					//int cur_col = (-(double(min_point_p_depth - avg_d) / double(100.0)) + 1.0) / 2.0 * 255;

					//FIND THE DEPTH VALUE FOR EACH PIXEL IN ORIGINAL DEPTH IMAGE INPUT
					if (depth_map_data[Did + row * depth_size_ + col] > 1e-6) //NOT A BLACK INVALID PIXEL
					{
						valid_nonzero_depth_point_num_++;

						double cur_d = -(depth_map_data[Did + row * depth_size_ + col] * 2.0 - 1.0) * 100.0 + avg_d;
						double global_z = cur_d;

						double global_x = (global_u - u0offset_) / focusx_ * global_z;
						double global_y = (global_v - v0offset_) / focusy_ * global_z;

						double min_dist_2_surface = 15000000;
						int min_dist_2_surface_id = -1;
						double min_ddist2surfacedspherex = 0.0;
						double min_ddist2surfacedspherey = 0.0;
						double min_ddist2surfacedspherez = 0.0;

						//ICP
						//Minimize the distance between every point p from depth map input, and its projection on
						//the estimated hand model sphere surface

						//NOW FIND THE CLOSEST DISTANCE OF DEPTH POINT -> PROJECTED SPHERE SURFACE
						for (int j = 0; j < joint_num_; j++)
						{
							double sphere_center_x = sphere_center_data[Gid + j * 3];
							double sphere_center_y = sphere_center_data[Gid + j * 3 + 1];
							double sphere_center_z = sphere_center_data[Gid + j * 3 + 2];

							double sphere_radius = sphere_radius_data[Rid + j];

							//Dist of point p in depth map (row, col) -> sphere center point
							double dist_2_sphere_center = sqrt(pow(global_x - sphere_center_x, 2) + pow(global_y - sphere_center_y, 2) + pow(global_z - sphere_center_z, 2));
							double dist_2_surface = fabs(dist_2_sphere_center - sphere_radius);
							//IF < SPHERE_RADIUS MEANS P FALLS WITHIN THE SPHERE
							//OTHERWISE P IS OUTSIDE THE SPHERE

							double ddist2surfaceddist2spherecenter = 0.0;
							if (dist_2_surface < 1e-6) ddist2surfaceddist2spherecenter = 0.0;
							else if (dist_2_sphere_center - sphere_radius > 1e-6) ddist2surfaceddist2spherecenter = 1.0;
							else ddist2surfaceddist2spherecenter = -1.0;

							double ddist2spherecenterdspherex = 1.0 / (2.0 * dist_2_sphere_center) * 2.0 * (global_x - sphere_center_x) * -1.0;
							double ddist2spherecenterdspherey = 1.0 / (2.0 * dist_2_sphere_center) * 2.0 * (global_y - sphere_center_y) * -1.0;
							double ddist2spherecenterdspherez = 1.0 / (2.0 * dist_2_sphere_center) * 2.0 * (global_z - sphere_center_z) * -1.0;

							double ddist2surfacedspherex = ddist2surfaceddist2spherecenter * ddist2spherecenterdspherex;
							double ddist2surfacedspherey = ddist2surfaceddist2spherecenter * ddist2spherecenterdspherey;
							double ddist2surfacedspherez = ddist2surfaceddist2spherecenter * ddist2spherecenterdspherez;

							if (min_dist_2_surface - dist_2_surface > 1e-6)
							{
								min_dist_2_surface = dist_2_surface;
								min_dist_2_surface_id = j;
								min_ddist2surfacedspherex = ddist2surfacedspherex;
								min_ddist2surfacedspherey = ddist2surfacedspherey;
								min_ddist2surfacedspherez = ddist2surfacedspherez;
							}
						}

						//NOT BACKGROUND PIXEL 
						if (min_dist_2_surface_id != -1)
						{
							//loss += min_dist_2_surface;
							bottom_3d_diff[Gid + min_dist_2_surface_id * 3] = top_diff * min_ddist2surfacedspherex;
							bottom_3d_diff[Gid + min_dist_2_surface_id * 3 + 1] = top_diff * min_ddist2surfacedspherey;
							bottom_3d_diff[Gid + min_dist_2_surface_id * 3 + 2] = top_diff * min_ddist2surfacedspherez;
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelRealDepthMap2SphereRenderedLossLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelRealDepthMap2SphereRenderedLossLayer);
	REGISTER_LAYER_CLASS(DeepHandModelRealDepthMap2SphereRenderedLoss);
}

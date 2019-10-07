
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DeepHandModelSphereModel2DepthMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_size_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().depth_size();


		focusx_ = this->layer_param_.pinhole_camera_origin_param().focusx();
		focusy_ = this->layer_param_.pinhole_camera_origin_param().focusy();
		u0offset_ = this->layer_param_.pinhole_camera_origin_param().u0offset();
		v0offset_ = this->layer_param_.pinhole_camera_origin_param().v0offset();
	}
	template <typename Dtype>
	void DeepHandModelSphereModel2DepthMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		joint_num_ = (bottom[0]->shape())[1] / 3;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top_shape.push_back(depth_size_);
		top_shape.push_back(depth_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelSphereModel2DepthMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* sphere_center_data = bottom[0]->cpu_data(); //sphere center e.g. 41 spheres
		const Dtype* bbx_x1_data = bottom[1]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[2]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[3]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[4]->cpu_data(); //bbx_y2
		const Dtype* avgz_data = bottom[5]->cpu_data(); //z of centroid
		const Dtype* sphere_radius_data = bottom[6]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int Gid = t * joint_num_ * 3;
			int Rid = t * joint_num_;
			int Tid = t * 1 * depth_size_ * depth_size_;
			
			//clear
			for (int j = 0; j < 1 * depth_size_ * depth_size_; j++) top_data[Tid + j] = 0.0;

			double bbx_x1 = bbx_x1_data[t];
			double bbx_y1 = bbx_y1_data[t];
			double bbx_x2 = bbx_x2_data[t];
			double bbx_y2 = bbx_y2_data[t];
			
			double avg_d = avgz_data[t];

			//For each pixel find its depth
			Mat img = Mat::zeros(depth_size_, depth_size_, CV_8UC1);
			for (int row = 0; row < depth_size_; row++)
			{
				for (int col = 0; col < depth_size_; col++)
				{
					double global_u = col / double(depth_size_) * (bbx_x2 - bbx_x1) + bbx_x1;
					double global_v = row / double(depth_size_) * (bbx_y2 - bbx_y1) + bbx_y1;

					//fx / (u - u0) * X + fy / (v - v0) * Y - 2 * Z = 0
					double A = focusx_ / (global_u - u0offset_);
					double B = focusy_ / (global_v - v0offset_);
					double C = -2.0;

					double min_point_p_depth = 1500;
					int min_point_p_depth_sphere_id = -1;

					//min among all spheres
					//min (dist(center_sphere -> ray))
					for (int j = 0; j < joint_num_; j++)
					{
						double sphere_center_x = sphere_center_data[Gid + j * 3];
						double sphere_center_y = sphere_center_data[Gid + j * 3 + 1];
						double sphere_center_z = sphere_center_data[Gid + j * 3 + 2];

						double x0 = sphere_center_x;
						double y0 = sphere_center_y;
						double z0 = sphere_center_z;
						double sphere_radius = sphere_radius_data[Rid + j];

						//two endpoints
						double z1 = sphere_center_z - sphere_radius;
						double x1 = 1.0 / A * z1;
						double y1 = 1.0 / B * z1;

						double z2 = sphere_center_z + sphere_radius;
						double x2 = 1.0 / A * z2;
						double y2 = 1.0 / B * z2;

						//dist sphere center 3d point -> ray
						double fz = (x2 - x1) * (x2 - x0) + (y2 - y1) * (y2 - y0) + (z2 - z1) * (z2 - z0);
						double fm = pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2) + 1e-8;
						double r = fz / fm;
						double d = sqrt(pow(x0 - (r * x1 + (1 - r) * x2), 2) + pow(y0 - (r * y1 + (1 - r) * y2), 2) + pow(z0 - (r * z1 + (1 - r) * z2), 2));

						double point_p_depth = 500.0;
						if (d < sphere_radius) //ray intersects with the sphere
						{
							point_p_depth = z0 - sqrt(pow(sphere_radius, 2) - pow(d, 2));
							if (min_point_p_depth - point_p_depth > 1e-6 && point_p_depth > 250)
							{
								min_point_p_depth = point_p_depth;
								min_point_p_depth_sphere_id = j;
							}
						}
						
					}

					//NOT BACKGROUND PIXEL 
					if (min_point_p_depth_sphere_id != -1)
					{
						int cur_col = (-(double(min_point_p_depth - avg_d) / double(100.0)) + 1.0) / 2.0 * 255;
						img.at<uchar>(row, col) = cur_col;
						//cout << row << " " << col << " " << min_point_p_depth << " " << avg_d << "\n";
						top_data[Tid + row * depth_size_ + col] = cur_col / 255.0; // [0, 1]
					}
				}
			}

			//imshow("img", img);
			//waitKey(0);
		}
	}

	template <typename Dtype>
	void DeepHandModelSphereModel2DepthMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* sphere_center_data = bottom[0]->cpu_data(); //sphere center e.g. 41 spheres
		const Dtype* bbx_x1_data = bottom[1]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[2]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[3]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[4]->cpu_data(); //bbx_y2
		const Dtype* avgz_data = bottom[5]->cpu_data(); //z of centroid
		const Dtype* sphere_radius_data = bottom[6]->cpu_data();


		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_3d_diff = bottom[0]->mutable_cpu_diff(); //sphere center gradient

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int Gid = t * joint_num_ * 3;
			int Rid = t * joint_num_;
			int Tid = t * 1 * depth_size_ * depth_size_;

			//clear
			for (int j = 0; j < joint_num_ * 3; j++) bottom_3d_diff[Gid + j] = 0.0;
			double bbx_x1 = bbx_x1_data[t];
			double bbx_y1 = bbx_y1_data[t];
			double bbx_x2 = bbx_x2_data[t];
			double bbx_y2 = bbx_y2_data[t];

			double avg_d = avgz_data[t];

			//For each pixel find its depth
			for (int row = 0; row < depth_size_; row++)
			{
				for (int col = 0; col < depth_size_; col++)
				{
					double global_u = col / double(depth_size_) * (bbx_x2 - bbx_x1) + bbx_x1;
					double global_v = row / double(depth_size_) * (bbx_y2 - bbx_y1) + bbx_y1;

					//fx / (u - u0) * X + fy / (v - v0) * Y - 2 * Z = 0
					double A = focusx_ / (global_u - u0offset_);
					double B = focusy_ / (global_v - v0offset_);
					double C = -2.0;

					double min_point_p_depth = 1500;
					int min_point_p_depth_sphere_id = -1;
					double min_dsynthdx0 = 0;
					double min_dsynthdy0 = 0;
					double min_dsynthdz0 = 0;


					//min among all spheres
					//min (dist(center_sphere -> ray))
					for (int j = 0; j < joint_num_; j++)
					{
						double sphere_center_x = sphere_center_data[Gid + j * 3];
						double sphere_center_y = sphere_center_data[Gid + j * 3 + 1];
						double sphere_center_z = sphere_center_data[Gid + j * 3 + 2];

						double x0 = sphere_center_x;
						double y0 = sphere_center_y;
						double z0 = sphere_center_z;
						double sphere_radius = sphere_radius_data[Rid + j];

						//two endpoints
						//X1 X2 Y1 Y2 Z1 Z2 ARE JUST CONSTANTS
						//two endpoints
						double z1 = sphere_center_z - sphere_radius;
						double x1 = 1.0 / A * z1;
						double y1 = 1.0 / B * z1;

						double z2 = sphere_center_z + sphere_radius;
						double x2 = 1.0 / A * z2;
						double y2 = 1.0 / B * z2;

						//dist sphere center 3d point -> ray
						double fz = (x2 - x1) * (x2 - x0) + (y2 - y1) * (y2 - y0) + (z2 - z1) * (z2 - z0);
						double fm = pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2) + 1e-8;
						double r = fz / fm;
						double d = sqrt(pow(x0 - (r * x1 + (1 - r) * x2), 2) + pow(y0 - (r * y1 + (1 - r) * y2), 2) + pow(z0 - (r * z1 + (1 - r) * z2), 2));

						

						double point_p_depth = 500.0;
						if (d < sphere_radius) //ray intersects with the sphere d < r
						{
							double t = sqrt(pow(sphere_radius, 2) - pow(d, 2));
							point_p_depth = z0 - t;
							double dtdd = 1.0 / (2.0 * t) * -2.0 * d;
							double dddx0 = 1.0 / (2.0 * d) * 2 * (x0 - (r * x1 + (1 - r) * x2)) * 1.0;
							double dddy0 = 1.0 / (2.0 * d) * 2 * (y0 - (r * y1 + (1 - r) * y2)) * 1.0;
							double dddz0 = 1.0 / (2.0 * d) * 2 * (z0 - (r * z1 + (1 - r) * z2)) * 1.0;
							//SYNTHESIZED DEPTH (ANALYTICAL FORM)
							//(- (point_p_depth - avg_d) / 100.0 + 1) / 2.0
							double dsynthdpointpdepth = -1.0 / 100.0 / 2.0;
							double dpointpdepthdt = -1;
							double dsynthdx0 = dsynthdpointpdepth * dpointpdepthdt * dtdd * dddx0;
							double dsynthdy0 = dsynthdpointpdepth * dpointpdepthdt * dtdd * dddy0;
							double dsynthdz0 = dsynthdpointpdepth * dpointpdepthdt * dtdd * dddz0 + dsynthdpointpdepth * 1.0; //-t + z0
							if (min_point_p_depth - point_p_depth > 1e-6) //minimum depth
							{
								min_point_p_depth = point_p_depth;
								min_point_p_depth_sphere_id = j;
								min_dsynthdx0 = dsynthdx0;
								min_dsynthdy0 = dsynthdy0;
								min_dsynthdz0 = dsynthdz0;
							}
						}

					}

					//NOT BACKGROUND PIXEL 
					if (min_point_p_depth_sphere_id != -1)
					{
						//int cur_col = (-(double(min_point_p_depth - avg_d) / double(100.0)) + 1.0) / 2.0 * 255;
						//top_data[Tid + row * depth_size_ + col] = cur_col / 255.0; // [0, 1]
						bottom_3d_diff[Gid + min_point_p_depth_sphere_id * 3] = min_dsynthdx0 * top_diff[Tid + row * depth_size_ + col];
						bottom_3d_diff[Gid + min_point_p_depth_sphere_id * 3 + 1] = min_dsynthdy0 * top_diff[Tid + row * depth_size_ + col];
						bottom_3d_diff[Gid + min_point_p_depth_sphere_id * 3 + 2] = min_dsynthdz0 * top_diff[Tid + row * depth_size_ + col];
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelSphereModel2DepthMapLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelSphereModel2DepthMapLayer);
	REGISTER_LAYER_CLASS(DeepHandModelSphereModel2DepthMap);
}

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"


#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include <iostream>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layer.hpp"

#include <sstream>

#include "caffe/data_transformer.hpp"


using namespace cv;
//#define _DEBUG



namespace caffe {

	template <typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_size_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().depth_size();
		file_prefix_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().file_prefix();

		focusx_ = this->layer_param_.pinhole_camera_origin_param().focusx();
		focusy_ = this->layer_param_.pinhole_camera_origin_param().focusy();
		u0offset_ = this->layer_param_.pinhole_camera_origin_param().u0offset();
		v0offset_ = this->layer_param_.pinhole_camera_origin_param().v0offset();

		minus_pixel_value_ = this->layer_param_.transform_param().minus_pixel_value();

		is_divide_ = this->layer_param_.transform_param().is_divide();

		//output point cloud
		o_pt_cl_ = this->layer_param_.deep_hand_model_hands19_param().o_pt_cl();

		//output 3d seg gt
		o_3d_seg_ = this->layer_param_.deep_hand_model_hands19_param().o_3d_seg();

		//output 2d seg gt
		o_2d_seg_ = this->layer_param_.deep_hand_model_hands19_param().o_2d_seg();

		depth_dims_ = this->layer_param_.gen_3d_skeleton_map_param().depth_dims();
		map_size_ = this->layer_param_.gen_3d_skeleton_map_param().map_size();

		focusx_ = this->layer_param_.pinhole_camera_origin_param().focusx();
		focusy_ = this->layer_param_.pinhole_camera_origin_param().focusy();
		u0offset_ = this->layer_param_.pinhole_camera_origin_param().u0offset();
		v0offset_ = this->layer_param_.pinhole_camera_origin_param().v0offset();
		gamma_ = this->layer_param_.gen_3d_skeleton_map_param().gamma();

		o_depth_voxel_ = this->layer_param_.deep_hand_model_hands19_param().o_depth_voxel();
		o_layered_depth_ = this->layer_param_.deep_hand_model_hands19_param().o_layered_depth();

		real_n_synth_ = this->layer_param_.deep_hand_model_hands19_param().real_n_synth();

	}
	template <typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//augmented image
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(3);
		top_shape.push_back(depth_size_);
		top_shape.push_back(depth_size_);
		top[0]->Reshape(top_shape);

		//augmented global joint 3d (ground truth)
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(JointNum * 3);
		top[1]->Reshape(top_shape);

		top_shape.clear();
		int top_id = 2;
		if (o_pt_cl_)
		{
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(depth_size_ * depth_size_ * 3);
			top[top_id]->Reshape(top_shape);
			top_id++;
		}

		if (o_3d_seg_)
		{
			top_shape.clear();
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(BoneNum * depth_dims_);
			top_shape.push_back(map_size_);
			top_shape.push_back(map_size_);
			top[top_id]->Reshape(top_shape);
			top_id++;
		}

		if (o_2d_seg_)
		{
			top_shape.clear();
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(BoneNum);
			top_shape.push_back(depth_size_);
			top_shape.push_back(depth_size_);
			top[top_id]->Reshape(top_shape);
			top_id++;

			top_shape.clear();
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(3);
			top_shape.push_back(depth_size_);
			top_shape.push_back(depth_size_);
			top[top_id]->Reshape(top_shape);
			top_id++;

		}

		if (o_depth_voxel_)
		{
			top_shape.clear();
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(depth_dims_);
			top_shape.push_back(map_size_);
			top_shape.push_back(map_size_);
			top[top_id]->Reshape(top_shape);
			top_id++;
		}

		if (o_layered_depth_)
		{
			top_shape.clear();
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(depth_dims_);
			top_shape.push_back(depth_size_);
			top_shape.push_back(depth_size_);
			top[top_id]->Reshape(top_shape);
			top_id++;
		}
	}




	template<typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::augmentation_scale(Dtype *joint_data, Dtype scale_self, Dtype *objpos_x, Dtype *objpos_y) {
		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		float scale_multiplier;
		//float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
		if (dice > this->layer_param_.transform_param().scale_prob()) {
			img_temp = img_src.clone();

			scale_multiplier = 1;
		}
		else {
			float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
			scale_multiplier = (this->layer_param_.transform_param().scale_max() - this->layer_param_.transform_param().scale_min()) * dice2 + this->layer_param_.transform_param().scale_min(); //linear shear into [scale_min, scale_max]
		}
		float scale_abs = this->layer_param_.transform_param().target_dist() / scale_self;

		if (real_n_synth_ && is_synth_) //cur: synth use last: real
		{
			scale_multiplier = last_scale_multiplier;
		}
		last_scale_multiplier = scale_multiplier;

		float scale = scale_abs * scale_multiplier;
		resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);
		//modify meta data
		*objpos_x = (*objpos_x) * scale;
		*objpos_y = (*objpos_y) * scale;

		for (int j = 0; j < JointNum; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				joint_data[j * 2 + k] *= scale;
			}
		}
	}


	template<typename Dtype>
	bool DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::onPlane(cv::Point p, Size img_size) {
		if (p.x < 0 || p.y < 0) return false;
		if (p.x >= img_size.width || p.y >= img_size.height) return false;
		return true;
	}



	template<typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::RotatePoint(cv::Point2f& p, Mat R) {
		Mat point(3, 1, CV_64FC1);
		point.at<double>(0, 0) = p.x;
		point.at<double>(1, 0) = p.y;
		point.at<double>(2, 0) = 1;
		Mat new_point = R * point;
		p.x = new_point.at<double>(0, 0);
		p.y = new_point.at<double>(1, 0);

		point.release();
		new_point.release();
	}





	template<typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::augmentation_rotate(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y) {

		float degree;
		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		degree = (dice - 0.5) * 2 * this->layer_param_.transform_param().max_rotate_degree();

		if (real_n_synth_ && is_synth_)
		{
			degree = last_degree;
		}
		last_degree = degree;

		Point2f center(img_temp.cols / 2.0, img_temp.rows / 2.0);
		Mat R = getRotationMatrix2D(center, degree, 1.0);
		Rect bbox = RotatedRect(center, img_temp.size(), degree).boundingRect();
		// adjust transformation matrix
		R.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		R.at<double>(1, 2) += bbox.height / 2.0 - center.y;
		//LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
		//          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
		//refill the border with color 128, 128, 128 (gray)
		warpAffine(img_temp, img_temp2, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));

		//adjust meta data

		Point2f objpos(*objpos_x, *objpos_y);
		RotatePoint(objpos, R);
		*objpos_x = objpos.x;
		*objpos_y = objpos.y;


		for (int j = 0; j < JointNum; j++)
		{
			Point2f joint_j(joint_data[j * 2], joint_data[j * 2 + 1]);
			RotatePoint(joint_j, R);

			joint_data[j * 2] = joint_j.x;
			joint_data[j * 2 + 1] = joint_j.y;

		}

		R.release();
	}




	template<typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::augmentation_croppad(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y) {
		float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		int crop_x = depth_size_;
		int crop_y = depth_size_;

		float x_offset = int((dice_x - 0.5) * 2 * this->layer_param_.transform_param().center_perterb_max());
		float y_offset = int((dice_y - 0.5) * 2 * this->layer_param_.transform_param().center_perterb_max());

		if (real_n_synth_ && is_synth_)
		{
			x_offset = last_x_offset;
			y_offset = last_y_offset;
		}
		last_x_offset = x_offset;
		last_y_offset = y_offset;

		//LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
		//LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
		Point2i center(*objpos_x + x_offset, *objpos_y + y_offset);

		int offset_left = -(center.x - (crop_x / 2));
		int offset_up = -(center.y - (crop_y / 2));
		// int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
		// int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);

		for (int i = 0; i<crop_y; i++) {
			for (int j = 0; j<crop_x; j++) { //i,j on cropped
				int coord_x_on_img = center.x - crop_x / 2 + j;
				int coord_y_on_img = center.y - crop_y / 2 + i;
				if (onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_temp2.cols, img_temp2.rows))) {
					img_aug.at<Vec3b>(i, j) = img_temp2.at<Vec3b>(coord_y_on_img, coord_x_on_img);
				}
			}
		}

		//gt joint 2d in raw -> gt joint 2d in [0, 1] ground truth 2d joint in bounding box
		//  +offset.x is tantamount to - bbx_x1
		//  +offset.y is tantamount to - bbx_y1
		Point2f offset(offset_left, offset_up);
		*objpos_x = (*objpos_x) + offset.x;
		*objpos_y = (*objpos_y) + offset.y;

		for (int j = 0; j < JointNum; j++)
		{
			joint_data[j * 2] += offset.x;
			joint_data[j * 2 + 1] += offset.y;
		}
	}




	template <typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int batSize = (bottom[0]->shape())[0];
		const Dtype* index_data = bottom[0]->cpu_data();  //index        
		const Dtype* avgz_data = bottom[1]->cpu_data(); //z of centroid
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

		const Dtype* gt_joint_3d_raw_data = bottom[12]->cpu_data(); //original joint 3d raw global gt
		const Dtype* gt_joint_2d_raw_data = bottom[13]->cpu_data(); //original joint 2d raw global gt on large depth image

		Dtype* depth_data = top[0]->mutable_cpu_data(); //depth     
		Dtype* gt_joint_3d_local_data = top[1]->mutable_cpu_data(); //joint 3d local (in cube [-1, 1])

		Dtype* pt_cl_data;
		int top_id = 2;
		if (o_pt_cl_) pt_cl_data = top[top_id++]->mutable_cpu_data(); //256x256 (typically) point cloud
		Dtype* gt_3d_seg_data;
		if (o_3d_seg_) gt_3d_seg_data = top[top_id++]->mutable_cpu_data();
		Dtype* gt_2d_seg_data;
		Dtype* gt_2d_seg_rgb_data;
		if (o_2d_seg_)
		{
			gt_2d_seg_data = top[top_id++]->mutable_cpu_data();
			gt_2d_seg_rgb_data = top[top_id++]->mutable_cpu_data();
		}

		Dtype* depth_voxel_data;
		if (o_depth_voxel_)
		{
			depth_voxel_data = top[top_id++]->mutable_cpu_data();
		}

		Dtype* layered_depth_data;
		if (o_layered_depth_)
		{
			layered_depth_data = top[top_id++]->mutable_cpu_data();
		}
		for (int t = 0; t < batSize; t++)
		{
			is_synth_ = (t % 2 == 1);
			int index = (int)index_data[t];
			char depthfile[maxlen];
			sprintf(depthfile, "%s%d%s", file_prefix_.c_str(), index, ".png");
			Mat depth = imread(depthfile, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
			Mat norm_depth = Mat::zeros(Size(depth.cols, depth.rows), CV_8UC3);

			double avg_d = avgz_data[t];

			for (int row = 0; row < depth.rows; row++)
			{
				for (int col = 0; col < depth.cols; col++)
				{
					int cur_d = depth.at<ushort>(row, col);
					//if (cur_d && cur_d >= depth_lb && cur_d <= depth_ub) //already reading a 640x480 cropped hand depth image w/ raw depth values
					//SYNTH DEPTH SOMETIMES > 850
					if (cur_d && cur_d >= 250 && cur_d <= 1500)
					{
						int cur_col = (-(double(cur_d - avg_d) / double(100.0)) + 1.0) / 2.0 * 255;
						//<avg_d e.g. -1    -> (1+1)/2=1.0   
						norm_depth.at<Vec3b>(row, col)[0] = norm_depth.at<Vec3b>(row, col)[1] = norm_depth.at<Vec3b>(row, col)[2] = cur_col;
					}
				}
			}

			resize(norm_depth, norm_depth, Size(depth_size_, depth_size_));

			img_src = norm_depth.clone();

			//------fetch gt joint 3d raw (to utilize z)
			double gt_joint_3d_raw[JointNum * 3];
			for (int j = 0; j < JointNum; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					int Bid = t * JointNum * 3;
					gt_joint_3d_raw[j * 3 + k] = gt_joint_3d_raw_data[Bid + j * 3 + k];
				}
			}

			//-----fetch gt joint 2d raw
			double gt_joint_2d_raw[JointNum * 2];
			for (int j = 0; j < JointNum; j++)
			{
				for (int k = 0; k < 2; k++)
				{
					int Bid = t * JointNum * 2;
					gt_joint_2d_raw[j * 2 + k] = gt_joint_2d_raw_data[Bid + j * 2 + k];
				}
			}

			//-----fetch bbx (2d & 3d)
			bbx_x1 = bbx_x1_data[t];
			bbx_y1 = bbx_y1_data[t];
			bbx_x2 = bbx_x2_data[t];
			bbx_y2 = bbx_y2_data[t];
			x_lb = x_lb_data[t];
			x_ub = x_ub_data[t];
			y_lb = y_lb_data[t];
			y_ub = y_ub_data[t];
			z_lb = z_lb_data[t];
			z_ub = z_ub_data[t];

			Dtype objpos_x = depth_size_ / 2.0;
			Dtype objpos_y = depth_size_ / 2.0;
			Dtype scale_provided = depth_size_ / 200.0; //200 is standard size from MPII dataset

														//-----calculated joint_data from gt_joint_2d_raw and 2d bbx
			Dtype t_joint_2d_data[JointNum * 2];
			for (int j = 0; j < JointNum; j++)
			{
				t_joint_2d_data[j * 2] = (gt_joint_2d_raw[j * 2] - bbx_x1) / (bbx_x2 - bbx_x1) * depth_size_; //here the "raw image" to be augmented is of depth_size_ * depth_size_
				t_joint_2d_data[j * 2 + 1] = (gt_joint_2d_raw[j * 2 + 1] - bbx_y1) / (bbx_y2 - bbx_y1) * depth_size_;
			}

			augmentation_scale(t_joint_2d_data, scale_provided, &objpos_x, &objpos_y);
			augmentation_rotate(t_joint_2d_data, &objpos_x, &objpos_y);
			img_aug = Mat::zeros(depth_size_, depth_size_, CV_8UC3) + Scalar(0, 0, 0); //depth map background set to black; so border fill color is black
			augmentation_croppad(t_joint_2d_data, &objpos_x, &objpos_y);

			//-----augmented image now stored in img_aug


			for (int row = 0; row < depth_size_; row++)
			{
				for (int col = 0; col < depth_size_; col++)
				{
					for (int c = 0; c < 3; c++)
					{
						int Tid = t * 3 * depth_size_ * depth_size_;
						depth_data[Tid + c * depth_size_ * depth_size_ + row * depth_size_ + col] = (img_aug.at<Vec3b>(row, col)[c] - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);							
					}

					if (o_pt_cl_)
					{
						//(-(double(cur_d - avg_d) / double(100.0)) + 1.0) / 2.0 * 255;
						double d = -(img_aug.at<Vec3b>(row, col)[0] / 255.0 * 2.0 - 1.0) * 100.0 + avg_d;
						double u = double(col) / depth_size_ * (bbx_x2 - bbx_x1) + bbx_x1;
						double v = double(row) / depth_size_ * (bbx_y2 - bbx_y1) + bbx_y1;

						double x = (u - u0offset_) * d / focusx_;
						double y = (v - v0offset_) * d / focusy_;
						double z = d; //remains unchanged
						int Pid = t * depth_size_ * depth_size_ * 3;
						//pseudo 3-channel x, y, z
						pt_cl_data[Pid + row * depth_size_ + col] = x;
						pt_cl_data[Pid + depth_size_ * depth_size_ + row * depth_size_ + col] = y;
						pt_cl_data[Pid + 2 * depth_size_ * depth_size_ + row * depth_size_ + col] = d;
					}

					if (o_layered_depth_)
					{
						//See if depth value >= k * (maxZ -minZ) / depth_dims + minZ
						double d = -(img_aug.at<Vec3b>(row, col)[0] / 255.0 * 2.0 - 1.0) * 100.0 + avg_d;
						for (int k = 0; k < depth_dims_; k++)
						{
							int Lid = t * depth_dims_ * depth_size_ * depth_size_;
							if (d - (k * (z_ub - z_lb) / depth_dims_ + z_lb) > 1e-6) layered_depth_data[Lid + k * depth_size_ * depth_size_ + row * depth_size_ + col] = (img_aug.at<Vec3b>(row, col)[0] - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);
							else layered_depth_data[Lid + k * depth_size_ * depth_size_ + row * depth_size_ + col] = 0.0;
						}
					}

					if (o_depth_voxel_)
					{
						double d = -(img_aug.at<Vec3b>(row, col)[0] / 255.0 * 2.0 - 1.0) * 100.0 + avg_d;
						double u = double(col) / depth_size_ * (bbx_x2 - bbx_x1) + bbx_x1;
						double v = double(row) / depth_size_ * (bbx_y2 - bbx_y1) + bbx_y1;

						double x = (u - u0offset_) * d / focusx_;
						double y = (v - v0offset_) * d / focusy_;
						double z = d; //remains unchanged

						//Local [0, 1]
						double local_x = (x - x_lb) / (x_ub - x_lb);
						double local_y = -((y - y_lb) / (y_ub - y_lb));
						double local_z = -((z - z_lb) / (z_ub - z_lb));

						local_x = max(0.0, min(1.0, local_x));
						local_y = max(0.0, min(1.0, local_y));
						local_z = max(0.0, min(1.0, local_z));
						int t_z = local_z * depth_dims_;
						int t_col = local_x * map_size_;
						int t_row = local_y * map_size_;
						if (t_row >= 0 && t_row < map_size_ && t_col >= 0 && t_col < map_size_ && t_z >= 0 && t_z < depth_dims_)
						{
							int Did = t * depth_dims_ * map_size_ * map_size_;
							depth_voxel_data[Did + t_z * map_size_ * map_size_ + t_row * map_size_ + t_col] = 1.0;
							//e.g. 32x32x32 depth voxel occupancy grid
						}
					}
				}
			}

			//---------get augmented 2d in raw image(large depth map)
			for (int j = 0; j < JointNum; j++)
			{
				t_joint_2d_data[j * 2] = t_joint_2d_data[j * 2] / depth_size_ * (bbx_x2 - bbx_x1) + bbx_x1; //in [0, 200]
				t_joint_2d_data[j * 2 + 1] = t_joint_2d_data[j * 2 + 1] / depth_size_ * (bbx_y2 - bbx_y1) + bbx_y1;
			}

			double gt_joint_3d_global_data[JointNum * 3];
			//--------f * x / z + u0 -> modify x' y' z' (z remains the same because it is just in-plane image rotation)
			for (int j = 0; j < JointNum; j++)
			{
				double x = gt_joint_3d_raw[j * 3];
				double y = gt_joint_3d_raw[j * 3 + 1];
				double z = gt_joint_3d_raw[j * 3 + 2];

				//inverse of pinholecameraorigin layer
				double u = t_joint_2d_data[j * 2]; //after aug "x" on raw image plane
				double v = t_joint_2d_data[j * 2 + 1];
				double x_after = (u - u0offset_) * z / focusx_;
				double y_after = (v - v0offset_) * z / focusy_;
				double z_after = z; //remains unchanged
				gt_joint_3d_global_data[j * 3] = x_after;
				gt_joint_3d_global_data[j * 3 + 1] = y_after;
				gt_joint_3d_global_data[j * 3 + 2] = z_after;

									//inverse of cuboidintolocalv2layer
									//========double global_x = (joint_3d_data[Bid + i * 3] + 1.0) / 2.0 * (x_ub - x_lb) + x_lb;
									//========double global_y = (-joint_3d_data[Bid + i * 3 + 1] + 1.0) / 2.0 * (y_ub - y_lb) + y_lb;
									//========double global_z = (-joint_3d_data[Bid + i * 3 + 2] + 1.0) / 2.0 * (z_ub - z_lb) + z_lb;
				double local_x = (x_after - x_lb) / (x_ub - x_lb) * 2.0 - 1.0;
				double local_y = -((y_after - y_lb) / (y_ub - y_lb) * 2.0 - 1.0);
				double local_z = -((z_after - z_lb) / (z_ub - z_lb) * 2.0 - 1.0);

				//=======save to top blobs (only in local coordinate space)
				int Tid = t * JointNum * 3;
				gt_joint_3d_local_data[Tid + j * 3] = local_x;
				gt_joint_3d_local_data[Tid + j * 3 + 1] = local_y;
				gt_joint_3d_local_data[Tid + j * 3 + 2] = local_z;
			}

			//output gt 2d segmentation
			if (o_2d_seg_)
			{
				for (int row = 0; row < depth_size_; row++)
				{
					for (int col = 0; col < depth_size_; col++)
					{
						int Sid = t * BoneNum * depth_size_ * depth_size_;
						for (int b = 0; b < BoneNum; b++)
						{
							gt_2d_seg_data[Sid + b * depth_size_ * depth_size_ + row * depth_size_ + col] = 0.0;
						}

						int Rid = t * 3 * depth_size_ * depth_size_;
						for (int c = 0; c < 3; c++)
						{
							gt_2d_seg_rgb_data[Rid + c * depth_size_ * depth_size_ + row * depth_size_ + col] = 0.0;
						}

						//double u = double(col) / depth_size_ * (bbx_x2 - bbx_x1) + bbx_x1;
						//double v = double(row) / depth_size_ * (bbx_y2 - bbx_y1) + bbx_y1;
						//double d = -(img_aug.at<Vec3b>(row, col)[0] / 255.0 * 2.0 - 1.0) * 100.0 + avg_d;

						//UVD -> XYZ
						//double global_x = (u - u0offset_) * d / focusx_;
						//double global_y = (v - v0offset_) * d / focusy_;
						//double global_z = d; //remains unchanged

						double global_x = double(col) / double(depth_size_) * (x_ub - x_lb) + x_lb;
						double global_y = double(row) / double(depth_size_) * (y_ub - y_lb) + y_lb;
						double global_z = -(img_aug.at<Vec3b>(row, col)[0] / 255.0 * 2.0 - 1.0) * 100.0 + avg_d;



						if (img_aug.at<Vec3b>(row, col)[0] > 0) //non-zero
						{
							int min_bone_id = 0;
							double min_bone_dist = 100000;
							for (int b = 0; b < BoneNum; b++)
							{
								double x_mid = (gt_joint_3d_global_data[bones[b][0] * 3] + gt_joint_3d_global_data[bones[b][1] * 3]) * 0.5;
								double y_mid = (gt_joint_3d_global_data[bones[b][0] * 3 + 1] + gt_joint_3d_global_data[bones[b][1] * 3 + 1]) * 0.5;
								double z_mid = (gt_joint_3d_global_data[bones[b][0] * 3 + 2] + gt_joint_3d_global_data[bones[b][1] * 3 + 2]) * 0.5;
								double dist_2_bone = sqrt(pow(global_x - x_mid, 2) + pow(global_y - y_mid, 2) + pow(global_z - z_mid, 2));
								if (min_bone_dist - dist_2_bone > 1e-6)
								{
									min_bone_dist = dist_2_bone;
									min_bone_id = b;
								}
							}

							gt_2d_seg_data[Sid + min_bone_id * depth_size_ * depth_size_ + row * depth_size_ + col] = 1.0;
							for (int c = 0; c < 3; c++)
							{
								gt_2d_seg_rgb_data[Rid + c * depth_size_ * depth_size_ + row * depth_size_ + col] = skeleton_color_bone_gt[min_bone_id][c] / 256.0;
							}

						}
					}
				}
			}

			//Output gt 3d segmentation
			if (o_3d_seg_)
			{
				//Enumerate all voxel
				//For each voxel (the depth value of which is non-zero in the image plane) find closest point
				for (int z = 0; z < depth_dims_; z++)
				{
					for (int row = 0; row < map_size_; row++)
					{
						for (int col = 0; col < map_size_; col++)
						{
							int Sid = t * BoneNum * depth_dims_ * map_size_ * map_size_;
							for (int b = 0; b < BoneNum; b++)
							{
								gt_3d_seg_data[Sid + b * depth_dims_ * map_size_ * map_size_ + z * map_size_ * map_size_ + row * map_size_ + col] = 0.0;
							}

							double global_x = double(col) / double(map_size_) * (x_ub - x_lb) + x_lb;
							double global_y = double(row) / double(map_size_) * (y_ub - y_lb) + y_lb;
							double global_z = double(z) / double(depth_dims_) * (z_ub - z_lb) + z_lb;

							//Project to 2D
							double global_u = focusx_* global_x / global_z + u0offset_;
							double global_v = focusy_ * global_y / global_z + v0offset_;

							//Local [0, 1] projection * depth_size_
							double local_u = (global_u - bbx_x1) / (bbx_x2 - bbx_x1) * depth_size_;
							double local_v = (global_v - bbx_y1) / (bbx_y2 - bbx_y1) * depth_size_;

							local_u = min(max(0, (int)local_u), depth_size_ - 1);
							local_v = min(max(0, (int)local_v), depth_size_ - 1);
							//See its depth value in augmented depth image
							double d = img_aug.at<Vec3b>(local_v, local_u)[0];
							if (d > 0)
							{
								//This voxel belongs to the hand 
								int min_bone_id = 0;
								double min_bone_dist = 100000;
								for (int b = 0; b < BoneNum; b++)
								{
									double x_mid = (gt_joint_3d_global_data[bones[b][0] * 3] + gt_joint_3d_global_data[bones[b][1] * 3]) * 0.5;
									double y_mid = (gt_joint_3d_global_data[bones[b][0] * 3 + 1] + gt_joint_3d_global_data[bones[b][1] * 3 + 1]) * 0.5;
									double z_mid = (gt_joint_3d_global_data[bones[b][0] * 3 + 2] + gt_joint_3d_global_data[bones[b][1] * 3 + 2]) * 0.5;
									double dist_2_bone = sqrt(pow(global_x - x_mid, 2) + pow(global_y - y_mid, 2) + pow(global_z - z_mid, 2));
									if (min_bone_dist - dist_2_bone > 1e-6)
									{
										min_bone_dist = dist_2_bone;
										min_bone_id = b;
									}
								}

								gt_3d_seg_data[Sid + min_bone_id * depth_dims_ * map_size_ * map_size_ + z * map_size_ * map_size_ + row * map_size_ + col] = exp(-gamma_ * min_bone_dist);// 1.0;
							}
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelReadDepthNoBBXWithAVGZAugLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelReadDepthNoBBXWithAVGZAugLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelReadDepthNoBBXWithAVGZAugLayer);
	REGISTER_LAYER_CLASS(DeepHandModelReadDepthNoBBXWithAVGZAug);
}

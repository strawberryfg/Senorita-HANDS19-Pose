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
#include <random>
#include <chrono>

using namespace cv;


namespace caffe {
	template <typename Dtype>
	void DeepHandModelAugLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//aug param (use transform_param)
		crop_x_ = this->layer_param_.transform_param().crop_size_x();
		crop_y_ = this->layer_param_.transform_param().crop_size_y();
		joint_num_ = this->layer_param_.transform_param().num_parts();  //note here is num_parts not num_parts + 1
		
		minus_pixel_value_ = this->layer_param_.transform_param().minus_pixel_value();

		is_divide_ = this->layer_param_.transform_param().is_divide();
		use_integral_aug_ = this->layer_param_.transform_param().use_integral_aug();
		color_scale_factor_ = this->layer_param_.transform_param().color_scale_factor();

		file_prefix_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().file_prefix();
		l_fr_d_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().l_fr_d();

		if (l_fr_d_)
		{
			img_src = Mat::zeros(Size(640, 480), CV_8UC3);
		}
		img_aug = Mat::zeros(crop_y_, crop_x_, CV_8UC3) + Scalar(128, 128, 128);
	}


	template <typename Dtype>
	void DeepHandModelAugLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//aug image blob
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(3);
		top_shape.push_back(crop_y_);
		top_shape.push_back(crop_x_);
		top[0]->Reshape(top_shape);

		//gt joint 2d global 
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(2 * (joint_num_));
		top[1]->Reshape(top_shape);

	}

	template<typename Dtype>
	void DeepHandModelAugLayer<Dtype>::augmentation_scale(Dtype *joint_data, Dtype scale_self, Dtype *objpos_x, Dtype *objpos_y) {
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

			if (use_integral_aug_)
			{
				unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

				std::default_random_engine generator(seed);
				std::normal_distribution<double> distribution(0.0, 1.0);

				double number = distribution(generator);
				number = min(number, 1.0);
				number = max(number, -1.0);

				//[-1, 1] -> [0, 2] -> [0, 1]
				double t_scale = (number + 1.0) * 0.5 * (this->layer_param_.transform_param().scale_max() - this->layer_param_.transform_param().scale_min()) + this->layer_param_.transform_param().scale_min();

				scale_multiplier = t_scale;

			}
		}
		float scale_abs = this->layer_param_.transform_param().target_dist() / scale_self;
		float scale = scale_abs * scale_multiplier;



		//printf("Scale is %12.6f Rows: %d Cols: %d \n", scale, img_src.rows, img_src.cols);
		resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);
		//modify meta data
		*objpos_x = (*objpos_x) * scale;
		*objpos_y = (*objpos_y) * scale;

		for (int j = 0; j < joint_num_; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				joint_data[j * 2 + k] *= scale;
			}
		}
	}

	template<typename Dtype>
	bool DeepHandModelAugLayer<Dtype>::onPlane(cv::Point p, Size img_size) {
		if (p.x < 0 || p.y < 0) return false;
		if (p.x >= img_size.width || p.y >= img_size.height) return false;
		return true;
    }

	template<typename Dtype>
	void DeepHandModelAugLayer<Dtype>::RotatePoint(cv::Point2f& p, Mat R) {
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
	void DeepHandModelAugLayer<Dtype>::augmentation_rotate(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y) {

		float degree;
		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		degree = (dice - 0.5) * 2 * this->layer_param_.transform_param().max_rotate_degree();

		if (use_integral_aug_)
		{
			if (dice <= 0.6)
			{
				unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

				std::default_random_engine generator(seed);
				std::normal_distribution<double> distribution(0.0, 1.0);

				double number = distribution(generator);
				number = min(number, 2.0);
				number = max(number, -2.0);

				double rot = number * this->layer_param_.transform_param().max_rotate_degree();
				degree = rot;
			}
			else
			{
				degree = 0.0;
			}
		}

		Point2f center(img_src.cols / 2.0, img_src.rows / 2.0);
		Mat R = getRotationMatrix2D(center, degree, 1.0);
		Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
		// adjust transformation matrix
		R.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		R.at<double>(1, 2) += bbox.height / 2.0 - center.y;
		//LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
		//          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
		//refill the border with color 128, 128, 128 (gray)
		warpAffine(img_temp, img_temp2, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128, 128, 128));

		//adjust meta data

		Point2f objpos(*objpos_x, *objpos_y);
		RotatePoint(objpos, R);
		*objpos_x = objpos.x;
		*objpos_y = objpos.y;


		for (int j = 0; j < joint_num_; j++)
		{

			Point2f joint_j(joint_data[j * 2], joint_data[j * 2 + 1]);
			RotatePoint(joint_j, R);

			joint_data[j * 2] = joint_j.x;
			joint_data[j * 2 + 1] = joint_j.y;

		}
		R.release();
	}



	template<typename Dtype>
	void DeepHandModelAugLayer<Dtype>::augmentation_croppad(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y) {
		float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		int crop_x = this->layer_param_.transform_param().crop_size_x();
		int crop_y = this->layer_param_.transform_param().crop_size_y();

		float x_offset = int((dice_x - 0.5) * 2 * this->layer_param_.transform_param().center_perterb_max());
		float y_offset = int((dice_y - 0.5) * 2 * this->layer_param_.transform_param().center_perterb_max());

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
				if (onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))) {
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

		for (int j = 0; j < joint_num_; j++)
		{
			joint_data[j * 2] += offset.x;
			joint_data[j * 2 + 1] += offset.y;
		}
	}

	template <typename Dtype>
	void DeepHandModelAugLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* index_data = bottom[0]->cpu_data();  //index        
		const Dtype* bbx_x1_data = bottom[1]->cpu_data(); //bbx_x1
		const Dtype* bbx_y1_data = bottom[2]->cpu_data(); //bbx_y1
		const Dtype* bbx_x2_data = bottom[3]->cpu_data(); //bbx_x2
		const Dtype* bbx_y2_data = bottom[4]->cpu_data(); //bbx_y2
		const Dtype* gt_joint_2d_raw_data = bottom[5]->cpu_data(); //original joint 2d raw global gt on large depth image
		const Dtype* depth_data;
		if (!l_fr_d_) depth_data = bottom[6]->cpu_data(); //depth blob from GetHands19
		Dtype* transformed_data = top[0]->mutable_cpu_data();
		Dtype* transformed_label = top[1]->mutable_cpu_data();

		for (int t = 0; t < batSize; ++t)
		{
			Dtype bbx_x1 = bbx_x1_data[t];
			Dtype bbx_y1 = bbx_y1_data[t];
			Dtype bbx_x2 = bbx_x2_data[t];
			Dtype bbx_y2 = bbx_y2_data[t];
			Dtype objpos_x = (bbx_x1 + bbx_x2) / 2.0;
			Dtype objpos_y = (bbx_y1 + bbx_y2) / 2.0;

			Dtype depth_size = bbx_x2 - bbx_x1;
			Dtype scale_provided = depth_size / 200.0;

			Dtype gt_joint_2d_raw[JointNum * 2];
			for (int i = 0; i < joint_num_ * 2; i++) gt_joint_2d_raw[i] = gt_joint_2d_raw_data[t * joint_num_ * 2 + i];

			if (!l_fr_d_) //from bottom depth blob
			{
				for (int row = 0; row < 480; row++)
				{
					for (int col = 0; col < 640; col++)
					{
						for (int c = 0; c < 3; c++)
						{
							img_src.at<Vec3b>(row, col)[c] = 0;
						}
					}
				}

				for (int row = bbx_y1; row <= bbx_y2; row++)
				{
					for (int col = bbx_x1; col <= bbx_x2; col++)
					{
						int Bid = t * 1 * 480 * 640;
						for (int c = 0; c < 3; c++)
						{
							img_src.at<Vec3b>(row, col)[c] = depth_data[Bid + row * 640 + col] * 255.0;
						}
					}
				}
			}
			else
			{
				char depthfile[maxlen];
				sprintf(depthfile, "%s%d%s", file_prefix_.c_str(), (int)index_data[t], ".png");
				Mat depth = imread(depthfile, 0);
				cv::cvtColor(depth, img_src, COLOR_GRAY2BGR);
				//imshow("img_src", img_src);
				//waitKey(0);
			}

			//imshow("img", img_src);
			//waitKey(0);

			//temporary mat
			
			//Start augmentation
			augmentation_scale(gt_joint_2d_raw, scale_provided, &objpos_x, &objpos_y);
			//imshow("img", img_temp);
			//waitKey(0);

			augmentation_rotate(gt_joint_2d_raw, &objpos_x, &objpos_y);
			//imshow("img", img_temp2);
			//waitKey(0);
			
			augmentation_croppad(gt_joint_2d_raw, &objpos_x, &objpos_y);
			//imshow("img", img_aug);
			//waitKey(0);

			//Color augmentation
			double color_scale;
			std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(1.0 - color_scale_factor_, 1.0 + color_scale_factor_);
			color_scale = distribution(generator);

			for (int row = 0; row < img_aug.rows; row++)
			{
				for (int col = 0; col < img_aug.cols; col++)
				{
					for (int c = 0; c < 3; c++)
					{
						double this_pixel_color = (double)img_aug.at<Vec3b>(row, col)[c];
						this_pixel_color *= color_scale;
						this_pixel_color = min(255.0, this_pixel_color);
						this_pixel_color = max(0.0, this_pixel_color);
						img_aug.at<Vec3b>(row, col)[c] = (int)this_pixel_color;
					}
				}
			}

			//imshow("img_aug", img_aug);
			//waitKey(0);
			//save aug image to top blob [0]
			int offset = img_aug.rows * img_aug.cols;
			for (int row = 0; row < img_aug.rows; row++)
			{
				for (int col = 0; col < img_aug.cols; col++)
				{
					int Tid = t * 3 * offset;
					transformed_data[Tid + 0 * offset + row * img_aug.cols + col] = (img_aug.at<Vec3b>(row, col)[0] - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);
					transformed_data[Tid + 1 * offset + row * img_aug.cols + col] = (img_aug.at<Vec3b>(row, col)[1] - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);
					transformed_data[Tid + 2 * offset + row * img_aug.cols + col] = (img_aug.at<Vec3b>(row, col)[2] - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);
				}
			}

			//save aug label to top blob[1]
			// last one is center(nothing; background) can be ignored
			int Tid = t * joint_num_ * 2;
			for (int j = 0; j < 2 * (joint_num_); j++)
			{
				transformed_label[Tid + j] = 0.0;
			}
			//LOG(INFO) << "label cleaned";

			for (int j = 0; j < joint_num_; j++)
			{
				//joints is point2f 
				transformed_label[Tid + 2 * j + 0] = gt_joint_2d_raw[j * 2];
				transformed_label[Tid + 2 * j + 1] = gt_joint_2d_raw[j * 2 + 1];
			}
		}

	}


	template <typename Dtype>
	void DeepHandModelAugLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelAugLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelAugLayer);
	REGISTER_LAYER_CLASS(DeepHandModelAug);
}

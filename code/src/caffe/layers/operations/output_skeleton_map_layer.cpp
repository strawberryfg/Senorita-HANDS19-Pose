#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"
#define maxlen 1111
using namespace cv;

namespace caffe {

	template <typename Dtype>
	void OutputSkeletonMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		save_path_ = this->layer_param_.output_skeleton_param().save_path();
		save_size_ = this->layer_param_.output_skeleton_param().save_size();

		skeleton_size_ = this->layer_param_.output_skeleton_param().skeleton_size();

		normalize_rgb_ = this->layer_param_.output_skeleton_param().normalize_rgb();
		pixel_value_threshold_ = this->layer_param_.output_skeleton_param().pixel_value_threshold();
		set_to_color_b_ = this->layer_param_.output_skeleton_param().set_to_color_b();
		set_to_color_g_ = this->layer_param_.output_skeleton_param().set_to_color_g();
		set_to_color_r_ = this->layer_param_.output_skeleton_param().set_to_color_r();
	}
	template <typename Dtype>
	void OutputSkeletonMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	}

	template <typename Dtype>
	void OutputSkeletonMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //skeleton
		const Dtype* index_data = bottom[1]->cpu_data(); //index
		for (int t = 0; t < batSize; t++) {
			Mat img = Mat::zeros(Size(skeleton_size_, skeleton_size_), CV_8UC3);
			int Bid = t * 3 * skeleton_size_ * skeleton_size_;

			if (!normalize_rgb_)
			{
				for (int row = 0; row < skeleton_size_; row++)
				{
					for (int col = 0; col < skeleton_size_; col++)
					{
						for (int c = 0; c < 3; c++)
						{
							img.at<Vec3b>(row, col)[c] = std::min(1.0, std::max(0.0, (double)bottom_data[Bid + c * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col])) * 255.0;
						}
					}
				}
			}
			else
			{
				double min_pixel_value_b = 1111.0, max_pixel_value_b = -1111.0;
				double min_pixel_value_g = 1111.0, max_pixel_value_g = -1111.0;
				double min_pixel_value_r = 1111.0, max_pixel_value_r = -1111.0;
				for (int row = 0; row < skeleton_size_; row++)
				{
					for (int col = 0; col < skeleton_size_; col++)
					{
						min_pixel_value_b = std::min(min_pixel_value_b, (double)bottom_data[Bid + 0 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col]);
						max_pixel_value_b = std::max(max_pixel_value_b, (double)bottom_data[Bid + 0 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col]);
						min_pixel_value_g = std::min(min_pixel_value_g, (double)bottom_data[Bid + 1 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col]);
						max_pixel_value_g = std::max(max_pixel_value_g, (double)bottom_data[Bid + 1 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col]);
						min_pixel_value_r = std::min(min_pixel_value_r, (double)bottom_data[Bid + 2 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col]);
						max_pixel_value_r = std::max(max_pixel_value_r, (double)bottom_data[Bid + 2 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col]);
					}
				}

				for (int row = 0; row < skeleton_size_; row++)
				{
					for (int col = 0; col < skeleton_size_; col++)
					{
						double v = (double)bottom_data[Bid + 0 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col];
						img.at<Vec3b>(row, col)[0] = (v - min_pixel_value_b) / (max_pixel_value_b - min_pixel_value_b) * 255.0;
						v = (double)bottom_data[Bid + 1 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col];
						img.at<Vec3b>(row, col)[1] = (v - min_pixel_value_g) / (max_pixel_value_g - min_pixel_value_g) * 255.0;
						v = (double)bottom_data[Bid + 2 * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col];
						img.at<Vec3b>(row, col)[2] = (v - min_pixel_value_r) / (max_pixel_value_r - min_pixel_value_r) * 255.0;
						for (int c = 0; c < 3; c++)
						{
							if (img.at<Vec3b>(row, col)[c] < pixel_value_threshold_)
							{
								if (c == 0) img.at<Vec3b>(row, col)[c] = set_to_color_b_;
								if (c == 1) img.at<Vec3b>(row, col)[c] = set_to_color_g_;
								if (c == 2) img.at<Vec3b>(row, col)[c] = set_to_color_r_;
							}
						}
					}
				}
			}

			resize(img, img, Size(save_size_, save_size_));
			int id = index_data[t]; //index
			char filename[maxlen];
			sprintf(filename, "%s%d%s", save_path_.c_str(), id, ".png");
			imwrite(filename, img);
		}
	}

	template <typename Dtype>
	void OutputSkeletonMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(OutputSkeletonMapLayer);
#endif

	INSTANTIATE_CLASS(OutputSkeletonMapLayer);
	REGISTER_LAYER_CLASS(OutputSkeletonMap);
}

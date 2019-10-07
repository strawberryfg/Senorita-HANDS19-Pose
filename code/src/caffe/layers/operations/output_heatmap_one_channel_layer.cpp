#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"

using namespace cv;

namespace caffe {

	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		save_path_ = this->layer_param_.output_heatmap_param().save_path();
		save_size_ = this->layer_param_.output_heatmap_param().save_size();

		heatmap_size_ = this->layer_param_.output_heatmap_param().heatmap_size();

		show_mmcp_ = this->layer_param_.output_heatmap_param().show_mmcp();
		save_size_h_ = this->layer_param_.output_heatmap_param().save_size_h();
		save_size_w_ = this->layer_param_.output_heatmap_param().save_size_w();

	}
	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		heatmap_size_w_ = (bottom[0]->shape())[3];
		heatmap_size_h_ = (bottom[0]->shape())[2];
	}

	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //heatmap
		const Dtype* index_data = bottom[1]->cpu_data(); //index

		const Dtype* ext_bbx_x1_data;
		const Dtype* ext_bbx_y1_data;
		const Dtype* ext_bbx_x2_data;
		const Dtype* ext_bbx_y2_data;
		const Dtype* real_mmcp_x_data;
		const Dtype* real_mmcp_y_data;
		const Dtype* pred_mmcp_x_data;
		const Dtype* pred_mmcp_y_data;
		if (show_mmcp_)
		{
			ext_bbx_x1_data = bottom[2]->cpu_data();
			ext_bbx_y1_data = bottom[3]->cpu_data();
			ext_bbx_x2_data = bottom[4]->cpu_data();
			ext_bbx_y2_data = bottom[5]->cpu_data();
			real_mmcp_x_data = bottom[6]->cpu_data();
			real_mmcp_y_data = bottom[7]->cpu_data();
			pred_mmcp_x_data = bottom[8]->cpu_data();
			pred_mmcp_y_data = bottom[9]->cpu_data();
		}
		for (int t = 0; t < batSize; t++) 
		{
			Mat img;
			if (show_mmcp_)
			{
				//show both pred and gt mmcp to see if the regressed offset makes sense
				img = Mat::zeros(Size(heatmap_size_w_, heatmap_size_h_), CV_8UC3);
				int Bid = t * 1 * heatmap_size_h_ * heatmap_size_w_;
				for (int row = 0; row < heatmap_size_h_; row++)
				{
					for (int col = 0; col < heatmap_size_w_; col++)
					{
						for (int c = 0; c < 3;c++)
							img.at<Vec3b>(row, col)(c) = std::min(std::max(0.0, (double)bottom_data[Bid + row * heatmap_size_w_ + col]), 1.0) * 255.0;
					}
				}
				//Green: pred Red: gt
				double ext_bbx_x1 = ext_bbx_x1_data[t];
				double ext_bbx_y1 = ext_bbx_y1_data[t];
				double ext_bbx_x2 = ext_bbx_x2_data[t];
				double ext_bbx_y2 = ext_bbx_y2_data[t];
				double real_mmcp_x = real_mmcp_x_data[t];
				double real_mmcp_y = real_mmcp_y_data[t];
				double pred_mmcp_x = pred_mmcp_x_data[t];
				double pred_mmcp_y = pred_mmcp_y_data[t];
				real_mmcp_x = (real_mmcp_x - ext_bbx_x1) / (ext_bbx_x2 - ext_bbx_x1) * heatmap_size_w_;
				real_mmcp_y = (real_mmcp_y - ext_bbx_y1) / (ext_bbx_y2 - ext_bbx_y1) * heatmap_size_h_;

				pred_mmcp_x = (pred_mmcp_x - ext_bbx_x1) / (ext_bbx_x2 - ext_bbx_x1) * heatmap_size_w_;
				pred_mmcp_y = (pred_mmcp_y - ext_bbx_y1) / (ext_bbx_y2 - ext_bbx_y1) * heatmap_size_h_;
				circle(img, Point(pred_mmcp_x, pred_mmcp_y), 3, Scalar(0, 255, 0), -1);
				circle(img, Point(real_mmcp_x, real_mmcp_y), 7, Scalar(0, 0, 255), 2);

				resize(img, img, Size(save_size_, save_size_));
			}
			else
			{
				img = Mat::zeros(Size(heatmap_size_w_, heatmap_size_h_), CV_8UC1);
				int Bid = t * 1 * heatmap_size_h_ * heatmap_size_w_;
				for (int row = 0; row < heatmap_size_h_; row++)
				{
					for (int col = 0; col < heatmap_size_w_; col++)
					{
						img.at<uchar>(row, col) = std::min(std::max(0.0, (double)bottom_data[Bid + row * heatmap_size_w_ + col]), 1.0) * 255.0;
					}
				}
				resize(img, img, Size(save_size_w_, save_size_h_));
			}
			int id = index_data[t]; //index
			char filename[maxlen];
			sprintf(filename, "%s%d%s", save_path_.c_str(), id, ".png");
			imwrite(filename, img);
		}
	}

	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(OutputHeatmapOneChannelLayer);
#endif

	INSTANTIATE_CLASS(OutputHeatmapOneChannelLayer);
	REGISTER_LAYER_CLASS(OutputHeatmapOneChannel);
}

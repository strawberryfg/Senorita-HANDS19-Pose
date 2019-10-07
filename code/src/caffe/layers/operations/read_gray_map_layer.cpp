#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


using namespace cv;

namespace caffe {

	template <typename Dtype>
	void ReadGrayMapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		read_path_ = this->layer_param_.read_gray_map_param().read_path();
		resize_size_ = this->layer_param_.read_gray_map_param().resize_size();
		map_num_ = this->layer_param_.read_gray_map_param().map_num();
	}
	template <typename Dtype>
	void ReadGrayMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(map_num_);
		top_shape.push_back(resize_size_);
		top_shape.push_back(resize_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void ReadGrayMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* index_data = bottom[0]->cpu_data(); //index
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int id = index_data[t]; //index
			char filename[maxlen];
			for (int c = 0; c < map_num_; c++)
			{
				sprintf(filename, "%s%d%s", read_path_.c_str(), id, ".png");
				Mat img = imread(filename, 0);
				resize(img, img, Size(resize_size_, resize_size_));
				for (int row = 0; row < resize_size_; row++)
				{
					for (int col = 0; col < resize_size_; col++)
					{
						int Tid = t * map_num_ * resize_size_ * resize_size_;
						top_data[Tid + c * resize_size_ * resize_size_ + row * resize_size_ + col] = img.at<uchar>(row, col) / 255.0;
					}
				}
			}
		}
	}

	template <typename Dtype>
	void ReadGrayMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(ReadGrayMapLayer);
#endif

	INSTANTIATE_CLASS(ReadGrayMapLayer);
	REGISTER_LAYER_CLASS(ReadGrayMap);
}

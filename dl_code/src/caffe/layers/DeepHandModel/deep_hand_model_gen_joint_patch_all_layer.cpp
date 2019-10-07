#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"

using namespace cv;
//Generate bone patch from gt 2d
//output blob image data in [0, 1]
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenJointPatchAllLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		img_size_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().img_size();
		//both origin image size & resize size
		resize_size_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().resize_size();
		alpha_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().alpha();
		beta_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().beta();
		
		crop_size_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().crop_size();
	}

	template <typename Dtype>
	void DeepHandModelGenJointPatchAllLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		channel_num_ = (bottom[1]->shape())[1];
		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(channel_num_ * JointNum);
		top_shape.push_back(resize_size_);
		top_shape.push_back(resize_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGenJointPatchAllLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* joint_2d_data = bottom[0]->cpu_data();
		const Dtype* img_data = bottom[1]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();


		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++)
		{
			int Bid = t * JointNum * 2;
			int Iid = t * channel_num_ * img_size_ * img_size_;

			for (int joint_id_ = 0; joint_id_ < JointNum; joint_id_++)
			{
				double x = joint_2d_data[Bid + joint_id_ * 2];
				double y = joint_2d_data[Bid + joint_id_ * 2 + 1];
				
				double x1 = x - crop_size_;
				double y1 = y - crop_size_;
				double x2 = x + crop_size_;
				double y2 = y + crop_size_;

				for (int c = 0; c < channel_num_; c++)
				{
					Mat show = Mat::zeros(Size(x2 - x1 + 1, y2 - y1 + 1), CV_8UC1);

					for (int row = y1; row < y2; row++)
					{
						for (int col = x1; col < x2; col++)
						{
							//see if intersects with the connected band
							if (row >= 0 && row < img_size_ && col >= 0 && col < img_size_)
							{
								show.at<uchar>(row - y1, col - x1) = img_data[Iid + c * img_size_ * img_size_ + row * img_size_ + col] * alpha_ + beta_;
							}
						}
					}

					resize(show, show, Size(resize_size_, resize_size_));


					for (int row = 0; row < resize_size_; row++)
					{
						for (int col = 0; col < resize_size_; col++)
						{
							int Tid = t * channel_num_ * JointNum * resize_size_ * resize_size_;
							top_data[Tid + joint_id_ * channel_num_ * resize_size_ * resize_size_ + c * resize_size_ * resize_size_ + row * resize_size_ + col] = show.at<Vec3b>(row, col)[c] / 256.0;
						}
					}
				}

				/*
				Mat show;
				if (channel_num_ == 1) show = Mat::zeros(Size(x2 - x1 + 1, y2 - y1 + 1), CV_8UC1);
				else show = Mat::zeros(Size(x2 - x1 + 1, y2 - y1 + 1), CV_8UC3);
				for (int row = y1; row < y2; row++)
				{
					for (int col = x1; col < x2; col++)
					{
						//see if intersects with the connected band
						if (row >= 0 && row < img_size_ && col >= 0 && col < img_size_)
						{
							if (channel_num_ == 1)
							{
								show.at<uchar>(row - y1, col - x1) = img_data[Iid + row * img_size_ + col] * alpha_ + beta_;
							}
							else
							{
								for (int c = 0; c < 3; c++)
								{
									show.at<Vec3b>(row - y1, col - x1)[c] = img_data[Iid + c * img_size_ * img_size_ + row * img_size_ + col] * alpha_ + beta_;
								}
							}
						}
					}
				}


				resize(show, show, Size(resize_size_, resize_size_));

				//imshow("show", show);
				//waitKey(0);

				for (int row = 0; row < resize_size_; row++)
				{
					for (int col = 0; col < resize_size_; col++)
					{
						if (channel_num_ == 1)
						{
							int Tid = t * 1 * JointNum * resize_size_ * resize_size_;
							top_data[Tid + 1 * joint_id_ * resize_size_ * resize_size_ + row * resize_size_ + col] = show.at<uchar>(row, col) / 256.0;
						}
						else
						{
							for (int c = 0; c < 3; c++)
							{
								int Tid = t * 3 * JointNum * resize_size_ * resize_size_;
								top_data[Tid + 3 * joint_id_ * resize_size_ * resize_size_ + c * resize_size_ * resize_size_ + row * resize_size_ + col] = show.at<Vec3b>(row, col)[c] / 256.0;
							}
						}
					}
				}*/


			}
		}
	}


	template <typename Dtype>
	void DeepHandModelGenJointPatchAllLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {


	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenJointPatchAllLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenJointPatchAllLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenJointPatchAll);
}  // namespace caffe

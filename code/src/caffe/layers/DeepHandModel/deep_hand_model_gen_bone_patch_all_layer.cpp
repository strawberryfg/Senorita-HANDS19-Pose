#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"

using namespace cv;
//Generate bone patch from gt 2d
//output blob image data in [0, 1]
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGenBonePatchAllLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		img_size_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().img_size();
		//both origin image size & resize size
		extend_ratio_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().extend_ratio();
		resize_size_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().resize_size();
		alpha_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().alpha();
		beta_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().beta();
		min_wh_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().min_wh();

		line_width_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().line_width();

		o_patch_bbx_ = this->layer_param_.deep_hand_model_gen_bone_patch_param().o_patch_bbx();

		
	}

	template <typename Dtype>
	void DeepHandModelGenBonePatchAllLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		channel_num_ = (bottom[1]->shape())[1];
		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(BoneNum * channel_num_);
		top_shape.push_back(resize_size_);
		top_shape.push_back(resize_size_);
		top[0]->Reshape(top_shape);

		if (o_patch_bbx_)
		{
			top_shape.clear();
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(BoneNum * 4);			
			top[1]->Reshape(top_shape);
		}
	}

	template <typename Dtype>
	void DeepHandModelGenBonePatchAllLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* joint_2d_data = bottom[0]->cpu_data();
		const Dtype* img_data = bottom[1]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();

		Dtype* patch_bbx_data;
		if (o_patch_bbx_) patch_bbx_data = top[1]->mutable_cpu_data();

		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++)
		{
			int Bid = t * JointNum * 2;
			int Iid = t * channel_num_ * img_size_ * img_size_;
			int BBXid = t * BoneNum * 4;
			
			for (int bone_id_ = 0; bone_id_ < BoneNum; bone_id_++)
			{
				int u = bones[bone_id_][0];
				int v = bones[bone_id_][1];
				double x1 = joint_2d_data[Bid + u * 2];
				double y1 = joint_2d_data[Bid + u * 2 + 1];
				double x2 = joint_2d_data[Bid + v * 2];
				double y2 = joint_2d_data[Bid + v * 2 + 1];

				double save_x1 = x1;
				double save_y1 = y1;
				double save_x2 = x2;
				double save_y2 = y2;

				//Mat tmp = Mat::zeros(Size(img_size_, img_size_), CV_8UC1);
				//for (int row = 0; row < img_size_; row++)
				//{
				//	for (int col = 0; col < img_size_; col++)
				//	{
				//		tmp.at<uchar>(row, col) = img_data[Iid + row * img_size_ + col] * alpha_ + beta_;
				//	}
				//}
				//line(tmp, Point(x1, y1), Point(x2, y2), Scalar(255), 5, -3);
				//printf("bone %d %d %d %d %d\n", bone_id_, (int)x1, (int)y1, (int)x2, (int)y2);
				//imshow("tmp", tmp);
				//waitKey(0);

				double min_x = min(x1, x2);
				double max_x = max(x1, x2);
				double min_y = min(y1, y2);
				double max_y = max(y1, y2);

				for (int channel_id_ = 0; channel_id_ < channel_num_; channel_id_++)
				{
					//left top -> right bottom 
					x1 = min_x;
					x2 = max_x;
					y1 = min_y;
					y2 = max_y;

					double c_x = (x1 + x2) / 2.0;
					double c_y = (y1 + y2) / 2.0;

					int width = fabs(x2 - x1);
					int height = fabs(y2 - y1);


					//int s = max(width, height) * pad_ratio;
					width = max(min_wh_, width);
					height = max(min_wh_, height);
					int s = max(width, height) * extend_ratio_;

					//extended square bbx
					x1 = c_x - s / 2.0;
					x2 = c_x + s / 2.0;
					y1 = c_y - s / 2.0;
					y2 = c_y + s / 2.0;

					//connect the bone
					Mat connect = Mat::zeros(Size(img_size_, img_size_), CV_8UC1);
					line(connect, Point(save_x1, save_y1), Point(save_x2, save_y2), Scalar(255), line_width_);
					//imshow("connect", connect);
					//waitKey(0);
					Mat show;
					show = Mat::zeros(Size(x2 - x1 + 1, y2 - y1 + 1), CV_8UC1);
					
					for (int row = y1; row < y2; row++)
					{
						for (int col = x1; col < x2; col++)
						{
							//see if intersects with the connected band
							if (row >= 0 && row < img_size_ && col >= 0 && col < img_size_)
							{
								if (connect.at<uchar>(row, col))
								{
									show.at<uchar>(row - y1, col - x1) = img_data[Iid + channel_id_ * img_size_ * img_size_ + row * img_size_ + col] * alpha_ + beta_;																	
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
							//Bone Channel
							int Tid = t * BoneNum * channel_num_ * resize_size_ * resize_size_;
							top_data[Tid + bone_id_ * channel_num_ * resize_size_ * resize_size_ + channel_id_ * resize_size_ * resize_size_ + row * resize_size_ + col] = show.at<uchar>(row, col) / 256.0;							
						}
					}
				}
				
				if (o_patch_bbx_)
				{
					//x1 y1 x2 y2
					patch_bbx_data[BBXid + bone_id_ * 4] = x1;
					patch_bbx_data[BBXid + bone_id_ * 4 + 1] = y1;
					patch_bbx_data[BBXid + bone_id_ * 4 + 2] = x2;
					patch_bbx_data[BBXid + bone_id_ * 4 + 3] = y2;			
					//cout << x1 << " " << y1 << " " << x2 << " " << y2 << " " << bone_id_ << " " << t << "\n";
				}

				
			}
		}
	}


	template <typename Dtype>
	void DeepHandModelGenBonePatchAllLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {


	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGenBonePatchAllLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGenBonePatchAllLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGenBonePatchAll);
}  // namespace caffe

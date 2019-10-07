#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"
using namespace cv;
namespace caffe {

    template <typename Dtype>
    void DeepHandModelOutputJointOnSkeletonMapLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        use_raw_rgb_image_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().use_raw_rgb_image();
        read_from_disk_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().read_from_disk();
        //whether to read the raw depth image from disk
        raw_rgb_image_path_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().raw_rgb_image_path();
        show_gt_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().show_gt();

        save_path_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().save_path();
        save_size_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().save_size();


        //read the skeleton size (because the size of the skeleton map may vary?)
        skeleton_size_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().skeleton_size();
        
        load_skeleton_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().load_skeleton();

        //newly added
        dataset_name_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().dataset_name();
        joint_num_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().joint_num();

		alpha_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().alpha();

		beta_ = this->layer_param_.deep_hand_model_output_joint_on_skeleton_param().beta();



    }
    template <typename Dtype>
    void DeepHandModelOutputJointOnSkeletonMapLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        
    }

    template <typename Dtype>
	void DeepHandModelOutputJointOnSkeletonMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* skeleton_image_data = bottom[0]->cpu_data(); //skeleton image
		const Dtype* depth_image_data = bottom[1]->cpu_data(); //if read from disk then set depth_image_data blob to skeleton image blob
		const Dtype* index_data = bottom[2]->cpu_data(); //index        
		const Dtype* pred_2d_data = bottom[3]->cpu_data(); //pred 2d
		const Dtype* gt_2d_data = bottom[4]->cpu_data(); //ground truth 2d

		for (int t = 0; t < batSize; t++) {
			int id = index_data[t]; //index
			Mat img = Mat::zeros(Size(skeleton_size_, skeleton_size_), CV_8UC3);
			int Bid = t * 3 * skeleton_size_ * skeleton_size_;

			//read raw rgb image if exists use it as background
			if (use_raw_rgb_image_)
			{
				Mat raw_rgb;
				if (read_from_disk_)
				{
					char rgbname[maxlen];
					sprintf(rgbname, "%s%d%s", raw_rgb_image_path_.c_str(), id, ".png");
					raw_rgb = imread(rgbname);
				}
				else //read depth patch from blob
				{
					int depth_size_ = (bottom[1]->shape())[2]; //should equal to (bottom[1]->shape())[3], which is the width(both height) of depth image
					raw_rgb = Mat::zeros(Size(depth_size_, depth_size_), CV_8UC3);
					int Did = t * 3 * depth_size_ * depth_size_;
					for (int row = 0; row < depth_size_; row++)
					{
						for (int col = 0; col < depth_size_; col++)
						{
							for (int c = 0; c < 3; c++)
							{
								raw_rgb.at<Vec3b>(row, col)[c] = depth_image_data[Did + c * depth_size_ * depth_size_ + row * depth_size_ + col];
							}
						}
					}
				}

				resize(raw_rgb, raw_rgb, Size(skeleton_size_, skeleton_size_)); //resize it to skeleton size so as to align with skeleton map

				for (int row = 0; row < skeleton_size_; row++) {
					for (int col = 0; col < skeleton_size_; col++) {
						for (int c = 0; c < 3; c++) {
							img.at<Vec3b>(row, col)[c] = raw_rgb.at<Vec3b>(row, col)[c];
						}
					}
				}
			}
			else
				//read from bottom[1] depth blob
			{
				for (int row = 0; row < skeleton_size_; row++)
				{
					for (int col = 0; col < skeleton_size_; col++)
					{
						for (int c = 0; c < 3; c++)
						{
							int channel_num_ = (bottom[1]->shape())[1];
							int Bid = t * channel_num_ * skeleton_size_ * skeleton_size_;
							img.at<Vec3b>(row, col)[c] = depth_image_data[Bid + c * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col] * alpha_ + beta_;
						}
					}
				}
			}

			//load skeleton image data
#ifdef _DEBUG
			cout << t << " " << "load skeleton image data" << "\n";
#endif




			//visualize on skeleton map
#ifdef _DEBUG
			cout << t << " " << "visualize on skeleton map" << "\n";
#endif

			if (strcmp(dataset_name_.c_str(), "standard") == 0)
			{
				for (int i = 0; i < JointNum; i++) {
					int Bid = t * JointNum * 2;
					circle(img, Point2d(pred_2d_data[Bid + i * 2] * skeleton_size_, pred_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_pred_joint[i][0], color_pred_joint[i][1], color_pred_joint[i][2]), -3);
					if (show_gt_) circle(img, Point2d(gt_2d_data[Bid + i * 2] * skeleton_size_, gt_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_gt_joint[i][0], color_gt_joint[i][1], color_gt_joint[i][2]), -3);
				}
			}
			else if (strcmp(dataset_name_.c_str(), "NYU") == 0)
			{
				if (joint_num_ == 31)
				{
					for (int i = 0; i < JointNum_NYU; i++)
					{
						int Bid = t * JointNum_NYU * 2;
						circle(img, Point2d(pred_2d_data[Bid + i * 2] * skeleton_size_, pred_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_pred_joint_NYU[i][0], color_pred_joint_NYU[i][1], color_pred_joint_NYU[i][2]), -3);
						if (show_gt_) circle(img, Point2d(gt_2d_data[Bid + i * 2] * skeleton_size_, gt_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_gt_joint_NYU[i][0], color_gt_joint_NYU[i][1], color_gt_joint_NYU[i][2]), -3);
					}
				}
				else
				{
					for (int i = 0; i < JointNum_NYU_Eval; i++)
					{
						int Bid = t * JointNum_NYU * 2;
						int id = NYU_joints[i];
						circle(img, Point2d(pred_2d_data[Bid + id * 2] * skeleton_size_, pred_2d_data[Bid + id * 2 + 1] * skeleton_size_), 5, Scalar(color_pred_joint_NYU_Eval[i][0], color_pred_joint_NYU_Eval[i][1], color_pred_joint_NYU_Eval[i][2]), -3);
						if (show_gt_) circle(img, Point2d(gt_2d_data[Bid + id * 2] * skeleton_size_, gt_2d_data[Bid + id * 2 + 1] * skeleton_size_), 5, Scalar(color_gt_joint_NYU_Eval[i][0], color_gt_joint_NYU_Eval[i][1], color_gt_joint_NYU_Eval[i][2]), -3);
					}
				}
			}
			else if (strcmp(dataset_name_.c_str(), "ICVL") == 0)
			{
				for (int i = 0; i < JointNum_ICVL; i++)
				{
					int Bid = t * JointNum_ICVL * 2;
					circle(img, Point2d(pred_2d_data[Bid + i * 2] * skeleton_size_, pred_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_pred_joint_ICVL[i][0], color_pred_joint_ICVL[i][1], color_pred_joint_ICVL[i][2]), -3);
					if (show_gt_) circle(img, Point2d(gt_2d_data[Bid + i * 2] * skeleton_size_, gt_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_gt_joint_ICVL[i][0], color_gt_joint_ICVL[i][1], color_gt_joint_ICVL[i][2]), -3);
				}
			}
			else if (strcmp(dataset_name_.c_str(), "MSRA") == 0)
			{
				for (int i = 0; i < JointNum_MSRA; i++)
				{
					int Bid = t * JointNum_MSRA * 2;
					circle(img, Point2d(pred_2d_data[Bid + i * 2] * skeleton_size_, pred_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_pred_joint_MSRA[i][0], color_pred_joint_MSRA[i][1], color_pred_joint_MSRA[i][2]), -3);
					if (show_gt_) circle(img, Point2d(gt_2d_data[Bid + i * 2] * skeleton_size_, gt_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_gt_joint_MSRA[i][0], color_gt_joint_MSRA[i][1], color_gt_joint_MSRA[i][2]), -3);
				}
			}

#ifdef _DEBUG
			cout << t << " " << "connecting bone" << "\n";
#endif 
			if (strcmp(dataset_name_.c_str(), "standard") == 0)
			{
				for (int i = 0; i < BoneNum; i++) {
					int Bid = t * JointNum * 2;
					line(img, Point2d(pred_2d_data[Bid + bones[i][0] * 2] * skeleton_size_, pred_2d_data[Bid + bones[i][0] * 2 + 1] * skeleton_size_), Point2d(pred_2d_data[Bid + bones[i][1] * 2] * skeleton_size_, pred_2d_data[Bid + bones[i][1] * 2 + 1] * skeleton_size_), Scalar(color_pred_bone[i][0], color_pred_bone[i][1], color_pred_bone[i][2]), 3);
					if (show_gt_) line(img, Point2d(gt_2d_data[Bid + bones[i][0] * 2] * skeleton_size_, gt_2d_data[Bid + bones[i][0] * 2 + 1] * skeleton_size_), Point2d(gt_2d_data[Bid + bones[i][1] * 2] * skeleton_size_, gt_2d_data[Bid + bones[i][1] * 2 + 1] * skeleton_size_), Scalar(color_gt_bone[i][0], color_gt_bone[i][1], color_gt_bone[i][2]), 3);
				}
			}
			else if (strcmp(dataset_name_.c_str(), "NYU") == 0)
			{
				if (joint_num_ == 31)
				{
					for (int i = 0; i < BoneNum_NYU; i++)
					{
						int Bid = t * JointNum_NYU * 2;
						line(img, Point2d(pred_2d_data[Bid + bones_NYU[i][0] * 2] * skeleton_size_, pred_2d_data[Bid + bones_NYU[i][0] * 2 + 1] * skeleton_size_), Point2d(pred_2d_data[Bid + bones_NYU[i][1] * 2] * skeleton_size_, pred_2d_data[Bid + bones_NYU[i][1] * 2 + 1] * skeleton_size_), Scalar(color_pred_bone_NYU[i][0], color_pred_bone_NYU[i][1], color_pred_bone_NYU[i][2]), 3);
						if (show_gt_) line(img, Point2d(gt_2d_data[Bid + bones_NYU[i][0] * 2] * skeleton_size_, gt_2d_data[Bid + bones_NYU[i][0] * 2 + 1] * skeleton_size_), Point2d(gt_2d_data[Bid + bones_NYU[i][1] * 2] * skeleton_size_, gt_2d_data[Bid + bones_NYU[i][1] * 2 + 1] * skeleton_size_), Scalar(color_gt_bone_NYU[i][0], color_gt_bone_NYU[i][1], color_gt_bone_NYU[i][2]), 3);
					}
				}
				else
				{
					for (int i = 0; i < BoneNum_NYU_Eval; i++) {
						int Bid = t * JointNum_NYU * 2;
						line(img, Point2d(pred_2d_data[Bid + bones_NYU_Eval[i][0] * 2] * skeleton_size_, pred_2d_data[Bid + bones_NYU_Eval[i][0] * 2 + 1] * skeleton_size_), Point2d(pred_2d_data[Bid + bones_NYU_Eval[i][1] * 2] * skeleton_size_, pred_2d_data[Bid + bones_NYU_Eval[i][1] * 2 + 1] * skeleton_size_), Scalar(color_pred_bone_NYU_Eval[i][0], color_pred_bone_NYU_Eval[i][1], color_pred_bone_NYU_Eval[i][2]), 3);
						if (show_gt_) line(img, Point2d(gt_2d_data[Bid + bones_NYU_Eval[i][0] * 2] * skeleton_size_, gt_2d_data[Bid + bones_NYU_Eval[i][0] * 2 + 1] * skeleton_size_), Point2d(gt_2d_data[Bid + bones_NYU_Eval[i][1] * 2] * skeleton_size_, gt_2d_data[Bid + bones_NYU_Eval[i][1] * 2 + 1] * skeleton_size_), Scalar(color_gt_bone_NYU_Eval[i][0], color_gt_bone_NYU_Eval[i][1], color_gt_bone_NYU_Eval[i][2]), 3);
					}
				}
			}
			else if (strcmp(dataset_name_.c_str(), "ICVL") == 0)
			{
				for (int i = 0; i < BoneNum_ICVL; i++) 
				{
					int Bid = t * JointNum_ICVL * 2;
					line(img, Point2d(pred_2d_data[Bid + bones_ICVL[i][0] * 2] * skeleton_size_, pred_2d_data[Bid + bones_ICVL[i][0] * 2 + 1] * skeleton_size_), Point2d(pred_2d_data[Bid + bones_ICVL[i][1] * 2] * skeleton_size_, pred_2d_data[Bid + bones_ICVL[i][1] * 2 + 1] * skeleton_size_), Scalar(color_pred_bone_ICVL[i][0], color_pred_bone_ICVL[i][1], color_pred_bone_ICVL[i][2]), 3);
					if (show_gt_) line(img, Point2d(gt_2d_data[Bid + bones_ICVL[i][0] * 2] * skeleton_size_, gt_2d_data[Bid + bones_ICVL[i][0] * 2 + 1] * skeleton_size_), Point2d(gt_2d_data[Bid + bones_ICVL[i][1] * 2] * skeleton_size_, gt_2d_data[Bid + bones_ICVL[i][1] * 2 + 1] * skeleton_size_), Scalar(color_gt_bone_ICVL[i][0], color_gt_bone_ICVL[i][1], color_gt_bone_ICVL[i][2]), 3);
				}	
			}
			else if (strcmp(dataset_name_.c_str(), "MSRA") == 0)
			{
				for (int i = 0; i < BoneNum_MSRA; i++)
				{
					int Bid = t * JointNum_MSRA * 2;
					line(img, Point2d(pred_2d_data[Bid + bones_MSRA[i][0] * 2] * skeleton_size_, pred_2d_data[Bid + bones_MSRA[i][0] * 2 + 1] * skeleton_size_), Point2d(pred_2d_data[Bid + bones_MSRA[i][1] * 2] * skeleton_size_, pred_2d_data[Bid + bones_MSRA[i][1] * 2 + 1] * skeleton_size_), Scalar(color_pred_bone_MSRA[i][0], color_pred_bone_MSRA[i][1], color_pred_bone_MSRA[i][2]), 3);
					if (show_gt_) line(img, Point2d(gt_2d_data[Bid + bones_MSRA[i][0] * 2] * skeleton_size_, gt_2d_data[Bid + bones_MSRA[i][0] * 2 + 1] * skeleton_size_), Point2d(gt_2d_data[Bid + bones_MSRA[i][1] * 2] * skeleton_size_, gt_2d_data[Bid + bones_MSRA[i][1] * 2 + 1] * skeleton_size_), Scalar(color_gt_bone_MSRA[i][0], color_gt_bone_MSRA[i][1], color_gt_bone_MSRA[i][2]), 3);
				}
			}
#ifdef _DEBUG
            imshow("", img);
            waitKey(0);
            cout << t << " " << "resize" << "\n";
#endif
            resize(img, img, Size(save_size_, save_size_));

            char filename[maxlen];

            //save image to folder
            sprintf(filename, "%s%d%s", save_path_.c_str(), id, ".png");
            imwrite(filename, img);

        }
    }

    template <typename Dtype>
    void DeepHandModelOutputJointOnSkeletonMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
       
        
    }

#ifdef CPU_ONLY
    STUB_GPU(DeepHandModelOutputJointOnSkeletonMapLayer);
#endif

    INSTANTIATE_CLASS(DeepHandModelOutputJointOnSkeletonMapLayer);
    REGISTER_LAYER_CLASS(DeepHandModelOutputJointOnSkeletonMap);
}

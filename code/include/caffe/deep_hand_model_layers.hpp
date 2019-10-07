#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

#include <numeric/vector4.h>
#include <numeric/matrix4.h>


#include <vector>
#include "basic.h"

using namespace numeric;
using namespace cv;
namespace caffe {


	//argmax operation on 3d hm
	template <typename Dtype>
	class DeepHandModelArgmax3DHMLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelArgmax3DHMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelArgmax3DHM"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;


		double x_lb_;
		double x_ub_;
		double y_lb_;
		double y_ub_;
		double z_lb_;
		double z_ub_;
		int joint_num_;
	};


	//argmax operation of 2d hm
	template <typename Dtype>
	class DeepHandModelArgmaxHMLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelArgmaxHMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelArgmaxHM"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int gen_size_;
		int channels_;
	};


	//augmentation online
	//read depth image from disk file directly and normalize it no bounding box
	template <typename Dtype>
	class DeepHandModelAugLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelAugLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelAug"; }
		virtual inline int ExactNumBottomBlobs() const { return 7; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		


		double bbx_x1, bbx_x2, bbx_y1, bbx_y2; //bounding box from original big image
		double x_lb, x_ub, y_lb, y_ub, z_lb, z_ub; //for recovering original global joint 3d

		Mat img_src, img_temp, img_temp2, img_aug;

		double focusx_, focusy_, u0offset_, v0offset_; //from "....PinholeCameraOrigin" layer

		void augmentation_scale(Dtype *joint_data, Dtype scale_self, Dtype *objpos_x, Dtype *objpos_y);
		void RotatePoint(cv::Point2f& p, Mat R);
		void augmentation_rotate(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y);
		bool onPlane(cv::Point p, Size img_size);
		void augmentation_croppad(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y);
		int crop_x_, crop_y_;
		int joint_num_;
		double minus_pixel_value_;

		//whether divide 256.0
		bool is_divide_;
		bool use_integral_aug_;
		double color_scale_factor_;

		bool l_fr_d_;
		string file_prefix_;
	};


	template <typename Dtype>
	class DeepHandModelBone2JointLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelBone2JointLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelBone2Joint"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int which_bone[JointNum][JointNum];
	};


	template <typename Dtype>
	class DeepHandModelCubiodIntoGlobalV2Layer : public Layer<Dtype> {
	public:
		explicit DeepHandModelCubiodIntoGlobalV2Layer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelCubiodIntoGlobalV2"; }
		virtual inline int ExactNumBottomBlobs() const { return 7; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int joint_num_;
	};



	//sigma in each direction e^(-()/(2*sigma^2))*1/(2*pi*sigma^2)
	template <typename Dtype>
	class DeepHandModelGen3DHeatmapInMoreDetailLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DHeatmapInMoreDetailLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DHeatmapInMoreDetail"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		double sigma_x_;
		double sigma_y_;
		double sigma_z_;
		int joint_num_;

		//stats on training set (min max of x y z component sub)
		double x_lower_bound_;
		double x_upper_bound_;
		double y_lower_bound_;
		double y_upper_bound_;
		double z_lower_bound_;
		double z_upper_bound_;

		bool perform_backprop_;
	};

	//3D XYZ location heatmap 
	//....min_x max_x
	//....min_y max_y
	//....min_z max_z
	//v2
	template <typename Dtype>
	class DeepHandModelGen3DHeatmapInMoreDetailV2Layer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DHeatmapInMoreDetailV2Layer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DHeatmapInMoreDetailV2"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		double sigma_x_;
		double sigma_y_;
		double sigma_z_;
		int joint_num_;

		//stats on training set (min max of x y z component sub)
		double x_lower_bound_;
		double x_upper_bound_;
		double y_lower_bound_;
		double y_upper_bound_;
		double z_lower_bound_;
		double z_upper_bound_;

		bool perform_backprop_;
	};

	//3D Segmentation Map
	template <typename Dtype>
	class DeepHandModelGen3DSegMapPerChannelLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DSegMapPerChannelLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DSegMapPerChannel"; }
		virtual inline int ExactNumBottomBlobs() const { return 13; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		
		float gamma_;
		bool perform_backprop_;

		int depth_size_;

		double focusx_, focusy_, u0offset_, v0offset_; 
	};

	template <typename Dtype>
	class DeepHandModelGen3DSkeletonMapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DSkeletonMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DSkeletonMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		float line_width_;
		int color_label_option_;
		float x_lb_;
		float x_ub_;
		float y_lb_;
		float y_ub_;
		float z_lb_;
		float z_ub_;

		float endpoint_dist_threshold_;

	};


	template <typename Dtype>
	class DeepHandModelGen3DSkeletonMapPerChannelLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DSkeletonMapPerChannelLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DSkeletonMapPerChannel"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		float line_width_;
		int color_label_option_;
		float x_lb_;
		float x_ub_;
		float y_lb_;
		float y_ub_;
		float z_lb_;
		float z_ub_;

		float endpoint_dist_threshold_;
		bool dot_product_;
		float gamma_;
		bool perform_backprop_;
	};


	template <typename Dtype>
	class DeepHandModelGen3DSkeletonMapPerChannelV2Layer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DSkeletonMapPerChannelV2Layer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DSkeletonMapPerChannelV2"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		float line_width_;
		int color_label_option_;
		float x_lb_;
		float x_ub_;
		float y_lb_;
		float y_ub_;
		float z_lb_;
		float z_ub_;

		float endpoint_dist_threshold_;
		bool dot_product_;
		float gamma_;
		bool perform_backprop_;
	};

	//20 parent -> child +
	//64 nearby fingers + 
	//all other pairs of joints
	template <typename Dtype>
	class DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DSkeletonMapPerChannelAllPairs"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		float line_width_;
		int color_label_option_;
		float x_lb_;
		float x_ub_;
		float y_lb_;
		float y_ub_;
		float z_lb_;
		float z_ub_;

		float endpoint_dist_threshold_;
		bool dot_product_;
		float gamma_;
		bool perform_backprop_;
	};

	//20 parent -> child +
	//64 nearby fingers
	template <typename Dtype>
	class DeepHandModelGen3DSkeletonMapPerChannelNearbyFingerLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DSkeletonMapPerChannelNearbyFingerLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DSkeletonMapPerChannelNearbyFinger"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		float line_width_;
		int color_label_option_;
		float x_lb_;
		float x_ub_;
		float y_lb_;
		float y_ub_;
		float z_lb_;
		float z_ub_;

		float endpoint_dist_threshold_;
		bool dot_product_;
		float gamma_;
		bool perform_backprop_;
	};


	//all limb patch
	template <typename Dtype>
	class DeepHandModelGenBoneCubeAllLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenBoneCubeAllLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenBoneCubeAll"; }
		virtual inline int ExactNumBottomBlobs() const { return 12; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 1; }


	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int img_size_;
		int depth_dims_;
		int map_size_;

		double focusx_;
		double focusy_;
		double u0offset_;
		double v0offset_;

		int channel_num_;
	};



	//all limb patch
	template <typename Dtype>
	class DeepHandModelGenBonePatchAllLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenBonePatchAllLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenBonePatchAll"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 1; }


	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int img_size_;
		double extend_ratio_;
		int resize_size_;
		double alpha_;
		double beta_;
		int min_wh_;
		int channel_num_;

		int line_width_;

		bool o_patch_bbx_;
	};

	//gen depth map (hand pose estimation via latent 2.5D heatmap regression
	template <typename Dtype>
	class DeepHandModelGenDepthMapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenDepthMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenDepthMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;


		int joint_num_;
		double depth_lower_bound_;
		double depth_upper_bound_;
	};

	template <typename Dtype>
	class DeepHandModelGenHeatmapAllChannelsLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenHeatmapAllChannelsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenHeatmapAllChannels"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int gen_size_;
		double render_sigma_;
		
		int joint_num_;
	};


	//patches around joint center
	template <typename Dtype>
	class DeepHandModelGenJointPatchAllLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenJointPatchAllLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenJointPatchAll"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 1; }


	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int img_size_;
		int resize_size_;
		double alpha_;
		double beta_;
		int channel_num_;

		int crop_size_;
	};

	//Gen the vector point to wrist to parent of wrist (which is (0, 0, 0))
	template <typename Dtype>
	class DeepHandModelGenNegWristBlobLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenNegWristBlobLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenNegWristBlob"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	//One and for all (the bone num be it 20, or 210, or more)
	template <typename Dtype>
	class DeepHandModelGenProjImgPlaneMapAllLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenProjImgPlaneMapAllLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenProjImgPlaneMapAll"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		//int map_size_;
		//int depth_dims_;

		int squeeze_axis_;
		bool perform_backprop_;

		double dim_lb_;
		double dim_ub_;
		int bone_num_; //(bottom[0]->shape())[1] (20 or 210)
	};

	//Project point cloud voxel onto image plane XY, ZY, or ZX
	template <typename Dtype>
	class DeepHandModelGenProjImgPlaneMapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenProjImgPlaneMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenProjImgPlaneMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		//int map_size_;
		//int depth_dims_;
		
		int squeeze_axis_; 
		bool perform_backprop_;

		double dim_lb_;
		double dim_ub_;
	};

	//For each pair of joints, generate relative joint location difference from the bone vectors
	template <typename Dtype>
	class DeepHandModelGenRelativeJointPositionLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenRelativeJointPositionLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenRelativeJointPosition"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int which_bone[JointNum][JointNum];
		int f[JointNum][JointNum][BoneNum];
		//f[i][j][k] means whether the k-th bone is in the term of relative joint location difference of i-th joint and j-th joint
	};

	//Generate groud truth skeleton map
	template <typename Dtype>
	class DeepHandModelGenSkeletonMapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenSkeletonMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenSkeletonMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int S;
		float line_width;

		int joint_num_;
	};


	template <typename Dtype>
	class DeepHandModelGenSkeletonMapPerChannelLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenSkeletonMapPerChannelLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenSkeletonMapPerChannel"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int S;
		float line_width;

		int joint_num_;
	};



	template <typename Dtype>
	class DeepHandModelGenVNectMapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenVNectMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenVNectMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;


		int joint_num_;
		double x_lower_bound_;
		double x_upper_bound_;
		double y_lower_bound_;
		double y_upper_bound_;
		double depth_lower_bound_;
		double depth_upper_bound_;
	};


	//Generate JointNum * 3 - D vector (duplicating wrist)
	template <typename Dtype>
	class DeepHandModelGenWristJointBlobLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGenWristJointBlobLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGenWristJointBlob"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	};



	//Get Hands 2019 Challenge Dataset Depth
	template <typename Dtype>
	class DeepHandModelGetHands19ChaDepthLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGetHands19ChaDepthLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGetHands19ChaDepth"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 6; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 10; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int crop_size_;
		string file_prefix_;
		int depth_threshold_;
		string img_prefix_;

		bool o_add_depth_;


	private:
		void reproject2DTo3D(double &x, double &y, const double z);
		void Project3DTo2D(double &x, double &y, const double z);
		bool calcBoxCenter(const cv::Mat &handMat, int x, int y, double &avgX, double &avgY, double &avgZ, double &avgU, double &avgV, int opt);
		ushort drawHistogram(const Mat &handMat, int opt);
		cv::Mat convertDepthToRGB(const cv::Mat &depthImg);
		bool GetRidOfBackground(vector<double> vect, char imgpath[maxlen], Mat &handMat, Rect &extendRect, int opt);

	};


	template <typename Dtype>
	class DeepHandModelIntegralVectorLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralVector"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double dim_lb_;
		double dim_ub_;
	};


	template <typename Dtype>
	class DeepHandModelIntegralXLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralXLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralX"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	template <typename Dtype>
	class DeepHandModelIntegralYLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralYLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralY"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	template <typename Dtype>
	class DeepHandModelIntegralZLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralZLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralZ"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};


	//The transformation from joint vector to bone vector
	template <typename Dtype>
	class DeepHandModelJoint2BoneLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelJoint2BoneLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelJoint2Bone"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	//Interpolate keypoints (3D only 21 * 3-d vector) 
	//For each bone (U, V) in original 20 array
	//Interpolate U + interpolate_id / (interpolate_num + 1) (interpolate_id = 1..interpolate_num) (V - U)
	//Interpolate 21 joints at a time
	//First is always wrist
	//Follow the original joint order
	//No gradient (only for ground truth interpolation)
	template <typename Dtype>
	class DeepHandModelInterpolateKeypointsLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelInterpolateKeypointsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelInterpolateKeypoints"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int interpolate_num_;
		int interpolate_id_;
		
	};


	//The Forward Kinematics Hand Model Layer
	template <typename Dtype>
	class DeepHandModelLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModel"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


	private:

		//1. Related to parameter
		int isFixed[ParamNum + 5];        //whether to fix that DoF
		double InitParam[MaxBatch][ParamNum + 5];     //InitialRotationDegree

												  //2. Related to shape parameters(bone length) 
		double BoneLen[BoneNum];
		double saveBoneLen[BoneNum];

		//3. Related to transformation
		Matrix4d ConstMatr[ConstMatrNum];                    //constant matrices
		Matrix4d ConstMatr_grad[ConstMatrNum];               //d Joint.x y or z / d (1.0 + cnn_fc_output_scale_correction)

		std::vector<std::pair<matrix_operation, int> > HomoMatr[JointNum]; //homogenous matrices (rotation & translation matrices)
		Matrix4d PrevMatr[JointNum];                           //prevmat * resttransformation
		Vector4d Jacobian[JointNum][ParamNum + 5];                       //the gradient of joint with respect to parameter
		//5 scales (for each finger)
																	 //4. Related to joint locations
		Vector4d temp_Joint[JointNum];

		//5. Main functions
		void Forward(int image_id, Matrix4d mat, int i, int Bid, int prev_size, const Dtype *bottom_data);
		void Backward(int image_id, int Bid, int i, const Dtype* bottom_data);
		void SetupConstantMatrices();
		void SetupTransformation();
		Matrix4d GetMatrix(int image_id, int Bid, matrix_operation opt, int id, bool is_gradient, const Dtype *bottom_data);

		//6. Fixed bone length statistics on training set
		bool use_training_bone_len_stats_;

		//7. Tunable (Learnable) bone length scale parameter
		bool use_constant_scale_;
	};


	//Normalize 3d locations into a specific cubiod
	template <typename Dtype>
	class DeepHandModelNormalize3DIntoCubiodV2Layer : public Layer<Dtype> {
	public:
		explicit DeepHandModelNormalize3DIntoCubiodV2Layer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelNormalize3DIntoCubiodV2"; }
		virtual inline int ExactNumBottomBlobs() const { return 7; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		bool perform_back_prop_;
	};



	//output skeleton map(with joints visualized on the skeleton map) to file 
	template <typename Dtype>
	class DeepHandModelOutputJointOnSkeletonMapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelOutputJointOnSkeletonMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelOutputJointOnSkeletonMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 5; }
		virtual inline int ExactNumTopBlobs() const { return 0; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		bool use_raw_rgb_image_;
		bool read_from_disk_;
		string raw_rgb_image_path_;
		bool show_gt_;
		string save_path_;
		int save_size_;

		int skeleton_size_;
		bool load_skeleton_;

		string dataset_name_;
		int joint_num_;

		double alpha_, beta_;
	};


	//Pinhole Camera in origin space
	template <typename Dtype>
	class DeepHandModelPinholeCameraOriginLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelPinholeCameraOriginLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelPinholeCameraOrigin"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double focusx_;
		double focusy_;
		double u0offset_;
		double v0offset_;
	};


	//Project depth map to XY, ZY, ZX plane like multi view cnn liuhao ge
	template <typename Dtype>
	class DeepHandModelProjectDepthLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelProjectDepthLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelProjectDepth"; }
		virtual inline int ExactNumBottomBlobs() const { return 10; }
		virtual inline int ExactNumTopBlobs() const { return 3; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int depth_size_;

		double minus_pixel_value_;

		//whether divide 256.0
		bool is_divide_;
	};


	template <typename Dtype>
	class DeepHandModelProjectionLocal2Global3DLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelProjectionLocal2Global3DLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelProjectionLocal2Global3D"; }
		virtual inline int ExactNumBottomBlobs() const { return 8; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double focusx_;
		double focusy_;
		double u0offset_;
		double v0offset_;

		bool perform_back_prop_; //img coordinate 2d + depth -> global 3d
	};


	template <typename Dtype>
	class DeepHandModelProjectionGlobal2LocalLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelProjectionGlobal2LocalLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelProjectionGlobal2Local"; }
		virtual inline int ExactNumBottomBlobs() const { return 5; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};


	//augmentation online
	//read depth image from disk file directly and normalize it no bounding box
	template <typename Dtype>
	class DeepHandModelReadDepthNoBBXWithAVGZAugLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelReadDepthNoBBXWithAVGZAugLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelReadDepthNoBBXWithAVGZAug"; }
		virtual inline int ExactNumBottomBlobs() const { return 14; }
		virtual inline int ExactNumTopBlobs() const { return -1; }


		virtual inline int MinTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string file_prefix_;
		int depth_size_;


		double bbx_x1, bbx_x2, bbx_y1, bbx_y2; //bounding box from original big image
		double x_lb, x_ub, y_lb, y_ub, z_lb, z_ub; //for recovering original global joint 3d

		Mat img_src, img_temp, img_temp2, img_aug;

		double focusx_, focusy_, u0offset_, v0offset_; //from "....PinholeCameraOrigin" layer

		void augmentation_scale(Dtype *joint_data, Dtype scale_self, Dtype *objpos_x, Dtype *objpos_y);
		void RotatePoint(cv::Point2f& p, Mat R);
		void augmentation_rotate(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y);
		bool onPlane(cv::Point p, Size img_size);
		void augmentation_croppad(Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y);

		
		
		double minus_pixel_value_;

		//whether divide 256.0
		bool is_divide_;
		bool o_pt_cl_;

		bool o_3d_seg_;
		int depth_dims_;
		int map_size_;

		bool o_2d_seg_;


		double gamma_;
		//output depth voxel grid
		bool o_depth_voxel_;

		bool o_layered_depth_;

		bool real_n_synth_;

		bool is_synth_; //real; synth; real; synth...

		double last_scale_multiplier;
		double last_degree;
		double last_x_offset;
		double last_y_offset;
	};


	//Self supervision
	//Real depth map input -> sphere model rendered depth map
	template <typename Dtype>
	class DeepHandModelRealDepthMap2SphereRenderedLossLayer : public LossLayer<Dtype> {
	public:
		explicit DeepHandModelRealDepthMap2SphereRenderedLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelRealDepthMap2SphereRenderedLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 8; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int depth_size_;

		double focusx_, focusy_, u0offset_, v0offset_;

		int joint_num_; //number of spheres
		int valid_nonzero_depth_point_num_;
	};


	//Rotate the camera (viewer) around hand coordinate axis Y or X \theta degree
	//Rotate the point cloud (voxel) around axis Y or X -\theta degree
	template <typename Dtype>
	class DeepHandModelRotPointCloudLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelRotPointCloudLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelRotPointClouod"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		//int map_size_;
		//int depth_dims_;

		int rot_axis_;
		bool perform_backprop_;
		double rot_degree_;
		double ctheta_, stheta_;

		int bone_num_;
	};


	//Self supervision
	//Sphere model -> rendered depth map
	template <typename Dtype>
	class DeepHandModelSphereModel2DepthMapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelSphereModel2DepthMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelSphereModel2DepthMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 7; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int depth_size_;

		double focusx_, focusy_, u0offset_, v0offset_;

		int joint_num_; //number of spheres
	};
}

#endif  // CAFFE_COMMON_LAYERS_HPP_

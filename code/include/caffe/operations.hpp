#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#define maxnum 555
#define maxlen 555

namespace caffe {


	//======adaptive weight euclidean L2 loss
	template <typename Dtype>
	class AdaptiveWeightEucLossLayer : public LossLayer<Dtype> {
	public:
		explicit AdaptiveWeightEucLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "AdaptiveWeightEucLoss"; }
		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int ExactNumBottomBlobs() const { return -1; } //====prevent it from checking bottom blob counts

		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		int num_of_losses_;
		double avg_abs_diff_each_dim[111]; //===average of |x_i - x_i ^GT| (each dimension of each objective func) in the batch 
	};


	//Add vector by constant
	template <typename Dtype>
	class AddVectorByConstantLayer : public Layer<Dtype> {
	public:
		explicit AddVectorByConstantLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "AddVectorByConstant"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		float add_value_;
		int dim_size_;
	};


	//one  (one element value predicted by CNN) + the whole vector
	template <typename Dtype>
	class AddVectorBySingleVectorLayer : public Layer<Dtype> {
	public:
		explicit AddVectorBySingleVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "AddVectorBySingleVector"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int dim_size_;
	};

	//Bounded Correction: min(K, ||u||) * u / ||u||
	template <typename Dtype>
	class BoundPoseLayer : public Layer<Dtype> {
	public:
		explicit BoundPoseLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BoundPose"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int joint_num_;
		double upper_bound_;
	};



	//Cross Validation ten-fold leave one out choose a index from several indexes
	template <typename Dtype>
	class CrossValidationRandomChooseIndexLayer : public Layer<Dtype> {
	public:
		explicit CrossValidationRandomChooseIndexLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "CrossValidationRandomChooseIndex"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	//Multiply the [-1, 1] value by standard deviation and add the mean average number
	template <typename Dtype>
	class DenormalizeLayer : public Layer<Dtype> {
	public:
		explicit DenormalizeLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Denormalize"; }
		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		bool mul_hyp_;
		bool sub_root_;
	};



	//Use skeleton map as mask to filter out useless hand segments from the depth patch
	template <typename Dtype>
	class EltAddChannelsLayer : public Layer<Dtype> {
	public:
		explicit EltAddChannelsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "EltAddChannels"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int dim_lb_;
		int dim_ub_;

	};


	//c2f baseline human pose 3d heatmap render operation
	template <typename Dtype>
	class Gen3DHeatmapInMoreDetailV3Layer : public Layer<Dtype> {
	public:
		explicit Gen3DHeatmapInMoreDetailV3Layer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Gen3DHeatmapInMoreDetailV3"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		int crop_size_;
		int render_sigma_;
		double stride_;
		int joint_num_;

		//stats on training set (min max of x y z component sub)
		double x_lower_bound_;
		double x_upper_bound_;
		double y_lower_bound_;
		double y_upper_bound_;
		double z_lower_bound_;
		double z_upper_bound_;
		int output_res_;
	};


	//universal heatmap render (gaussian heatmap) layer
	template <typename Dtype>
	class GenHeatmapAllChannelsLayer : public Layer<Dtype> {
	public:
		explicit GenHeatmapAllChannelsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GenHeatmapAllChannels"; }
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
		bool all_one_; //binary classification (G-RMI)

		bool use_one_row_; //one row heatmap 01/11/2019

		bool use_cpm_render_;
		bool use_baseline_render_;
		int crop_size_;
		int stride_;
		int grid_size_;
	};


	//Generate random index
	template <typename Dtype>
	class GenRandIndexLayer : public Layer<Dtype> {
	public:
		explicit GenRandIndexLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GenRandIndex"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int index_lower_bound_;
		int index_upper_bound_;
		int batch_size_;
		string missing_index_file_;

		int num_of_missing_;

		int rand_generator_option_;
	};


	//Generate sequential index
	template <typename Dtype>
	class GenSequentialIndexLayer : public Layer<Dtype> {
	public:
		explicit GenSequentialIndexLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GenSequentialIndex"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string current_index_file_path_; //stores only one single value denoting the current index        
		int batch_size_;
		int num_of_samples_;
		int start_index_;
	};


	//Multiply the vector by a constant number
	template <typename Dtype>
	class IdentityVectorNoGradientLayer : public Layer<Dtype> {
	public:
		explicit IdentityVectorNoGradientLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "IdentityVectorNoGradient"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }

		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};


	//square root 3D Joint Location Loss
	template <typename Dtype>
	class Joint3DSquareRootLossLayer : public LossLayer<Dtype> {
	public:
		explicit Joint3DSquareRootLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Joint3DSquareRootLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		int joint_num_;

	};


	template <typename Dtype>
	class JointAbsLossLayer : public LossLayer<Dtype> {
	public:
		explicit JointAbsLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "JointAbsLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		int joint_num_;
		
	};



	//Jenson Shannon Regularization
	template <typename Dtype>
	class JSRegularizationLossLayer : public LossLayer<Dtype> {
	public:
		explicit JSRegularizationLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "JSRegularizationLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double min_eps_;
		int channel_num_;
	};


	//KL Divergence
	template <typename Dtype>
	class KLRegularizationLossLayer : public LossLayer<Dtype> {
	public:
		explicit KLRegularizationLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "KLRegularizationLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double min_eps_;
		int channel_num_;
	};


	//Use feature map as mask to filter out 
	//Thus preventing outliers from predominating the loss
	template <typename Dtype>
	class MaskFeatureMapByFeatureMapLayer : public Layer<Dtype> {
	public:
		explicit MaskFeatureMapByFeatureMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MaskFeatureMapByFeatureMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int skeleton_threshold_;
	};

	//Use skeleton map as mask to filter out useless hand segments from the depth patch
	template <typename Dtype>
	class MaskFeatureMapBySkeletonLayer : public Layer<Dtype> {
	public:
		explicit MaskFeatureMapBySkeletonLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MaskFeatureMapBySkeleton"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int skeleton_threshold_;
	};

	//Use skeleton map as mask to filter out useless hand segments from the depth patch
	template <typename Dtype>
	class MaskFeatureMapOutByMapLayer : public Layer<Dtype> {
	public:
		explicit MaskFeatureMapOutByMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MaskFeatureMapOutByMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int skeleton_threshold_;
	};


	//Write the blob to disk
	template <typename Dtype>
	class OutputBlobLayer : public Layer<Dtype> {
	public:
		explicit OutputBlobLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "OutputBlob"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 0; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string save_path_;
		string blob_name_;

		bool if_per_section_output_;
		int per_section_row_num_;
		int per_section_col_num_;
	};


	//Output all-channel heat map to disk     
	template <typename Dtype>
	class OutputHeatmapAllChannelsLayer : public Layer<Dtype> {
	public:
		explicit OutputHeatmapAllChannelsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "OutputHeatmapAllChannels"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 0; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string save_path_;
		int save_size_;

		int heatmap_size_;
		int joint_num_;
	};


	//Output one-channel heat map to disk     
	template <typename Dtype>
	class OutputHeatmapOneChannelLayer : public Layer<Dtype> {
	public:
		explicit OutputHeatmapOneChannelLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "OutputHeatmapOneChannel"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int ExactNumTopBlobs() const { return 0; }
		virtual inline int MinBottomBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string save_path_;
		int save_size_;

		int heatmap_size_;
		//is to show outcome of MMCP offset regression
		bool show_mmcp_;

		int heatmap_size_w_;
		int heatmap_size_h_;

		int save_size_w_;
		int save_size_h_;
	};



	//Read blob from disk file just one file
	template <typename Dtype>
	class ReadBlobFromFileLayer : public Layer<Dtype> {
	public:
		explicit ReadBlobFromFileLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ReadBlobFromFile"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string file_path_;
		int num_to_read_;
		double t_data[333];
		int batch_size_;
	};


	//Output skeleton map to disk     
	template <typename Dtype>
	class OutputSkeletonMapLayer : public Layer<Dtype> {
	public:
		explicit OutputSkeletonMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "OutputSkeletonMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 0; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string save_path_;
		int save_size_;

		int skeleton_size_;


		bool normalize_rgb_;
		int pixel_value_threshold_;
		int set_to_color_b_;
		int set_to_color_g_;
		int set_to_color_r_;
	};


	//Read blob from disk file indexing
	template <typename Dtype>
	class ReadBlobFromFileIndexingLayer : public Layer<Dtype> {
	public:
		explicit ReadBlobFromFileIndexingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ReadBlobFromFileIndexing"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string file_prefix_;
		int num_to_read_;
		double t_data[333];

	};


	//read gray map from file
	template <typename Dtype>
	class ReadGrayMapLayer : public Layer<Dtype> {
	public:
		explicit ReadGrayMapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ReadGrayMap"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string read_path_;
		int resize_size_;

		int map_num_; //e.g. BoneNum or JointNum
	};



	//Read index from disk file just one file (for testing)
	template <typename Dtype>
	class ReadIndexFromFileLayer : public Layer<Dtype> {
	public:
		explicit ReadIndexFromFileLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ReadIndexFromFile"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string index_file_path_;
		string current_index_file_path_; //stores only one single value denoting the current index
		int batch_size_;
		int num_of_samples_;
	};


	//one Scale (constant value ; not predicted by CNN) / the whole vector
	template <typename Dtype>
	class ScaleDivideVectorLayer : public Layer<Dtype> {
	public:
		explicit ScaleDivideVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ScaleDivideVector"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		float scale_factor_;
		int dim_size_;
	};


	//one  (one element value predicted by CNN) * the whole vector
	template <typename Dtype>
	class ScaleVectorBySingleVectorLayer : public Layer<Dtype> {
	public:
		explicit ScaleVectorBySingleVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ScaleVectorBySingleVector"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int dim_size_;
	};





	//Multiply the vector by a constant number
	template <typename Dtype>
	class ScaleVectorLayer : public Layer<Dtype> {
	public:
		explicit ScaleVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ScaleVector"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 1; };
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		float scale_factor_;
		int dim_size_;

	};


	//Multiply the vector by a vector blob
	template <typename Dtype>
	class ScaleVectorWithABlobLayer : public Layer<Dtype> {
	public:
		explicit ScaleVectorWithABlobLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ScaleVectorWithABlob"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int dim_size_;
		float base_scale_;
	};


	//one vector subtracts another vector    
	template <typename Dtype>
	class TwoVectorSubtractLayer : public Layer<Dtype> {
	public:
		explicit TwoVectorSubtractLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "TwoVectorSubtract"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int dim_size_;
		bool take_abs_;

	};



	//Multiply the [-1, 1] value by standard deviation and add the mean average number
	template <typename Dtype>
	class UnnormalizeLayer : public Layer<Dtype> {
	public:
		explicit UnnormalizeLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Unnormalize"; }
		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	};



	//Step-by-step iterative pose update loss
	template <typename Dtype>
	class UpdatePoseLossLayer : public LossLayer<Dtype> {
	public:
		explicit UpdatePoseLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "UpdatePoseLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int joint_num_;
		double lambda_;
	};


}

#endif  // CAFFE_COMMON_LAYERS_HPP_

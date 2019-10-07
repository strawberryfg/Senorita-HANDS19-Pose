
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

#define PI 3.14159265359
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_dims_ = this->layer_param_.gen_3d_skeleton_map_param().depth_dims();
		map_size_ = this->layer_param_.gen_3d_skeleton_map_param().map_size();

		line_width_ = this->layer_param_.gen_3d_skeleton_map_param().line_width();

		color_label_option_ = this->layer_param_.gen_3d_skeleton_map_param().color_label_option();


		x_lb_ = this->layer_param_.gen_3d_skeleton_map_param().x_lb();
		x_ub_ = this->layer_param_.gen_3d_skeleton_map_param().x_ub();

		y_lb_ = this->layer_param_.gen_3d_skeleton_map_param().y_lb();
		y_ub_ = this->layer_param_.gen_3d_skeleton_map_param().y_ub();

		z_lb_ = this->layer_param_.gen_3d_skeleton_map_param().z_lb();
		z_ub_ = this->layer_param_.gen_3d_skeleton_map_param().z_ub();
		endpoint_dist_threshold_ = this->layer_param_.gen_3d_skeleton_map_param().endpoint_dist_threshold();

		dot_product_ = this->layer_param_.gen_3d_skeleton_map_param().dot_product();

		gamma_ = this->layer_param_.gen_3d_skeleton_map_param().gamma();

		//whether to back propogate the gradient flow to the level of joint coordinates
		perform_backprop_ = this->layer_param_.gen_3d_skeleton_map_param().perform_backprop();
	}
	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((JointNum * (JointNum - 1) / 2) * depth_dims_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_2d_data = bottom[0]->cpu_data(); //gt joint 2d [0,  1]
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{
			int Jid = t * JointNum * 2;
			int Did = t * JointNum;
			int Tid = t * (JointNum * (JointNum - 1) / 2) * depth_dims_ * map_size_ * map_size_;
			//for (int j = 0; j < BoneNum; j++) 
			//clear
			for (int j = 0; j < (JointNum * (JointNum - 1) / 2) * depth_dims_ * map_size_ * map_size_; j++) top_data[Tid + j] = 0.0;

			//for each voxel find the closest bone
			int tj = -1;
			for (int u = 0; u < JointNum - 1; u++)
			{
				for (int v = u + 1; v < JointNum; v++)
				{
					tj++;
					for (int k = 0; k < depth_dims_; k++)
					{
						for (int row = 0; row < map_size_; row++)
						{
							for (int col = 0; col < map_size_; col++)
							{
								double x0 = double(col) / double(map_size_) * (x_ub_ - x_lb_) + x_lb_;
								double y0 = double(row) / double(map_size_) * (y_ub_ - y_lb_) + y_lb_;
								double z0 = double(k) / double(depth_dims_) * (z_ub_ - z_lb_) + z_lb_;
								//====two endpoints 
								double x1 = gt_joint_2d_data[Jid + u * 2];
								double x2 = gt_joint_2d_data[Jid + v * 2];
								double y1 = gt_joint_2d_data[Jid + u * 2 + 1];
								double y2 = gt_joint_2d_data[Jid + v * 2 + 1];
								double z1 = gt_depth_data[Did + u];
								double z2 = gt_depth_data[Did + v];
								if (!dot_product_)
								{
									//=========solve distance of point 2 line (3D)
									//====see details here: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
									double fz = (x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1);
									double fm = pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2) + 1e-8;
									double t = -fz / fm;
									double d = sqrt(pow((x1 - x0) + (x2 - x1) * t, 2) + pow((y1 - y0) + (y2 - y1) * t, 2) + pow((z1 - z0) + (z2 - z1) * t, 2));
									//========distance from point X0 to line X1->X2 X0 = [x0, y0, z0]^T, X1 = [x1, y1, z1]^T, X2 = [x2, y2, z2]^T
									bool flag_x_a;
									bool flag_x_b;
									//x2 > x0 > x1 or x1 > x0 > x2 x0 = x1 (or x2) endpoint 
									if (x0 - x1 > 1e-6 || fabs(x0 - x1) < 1e-6) flag_x_a = true; else flag_x_a = false;
									if (x2 - x0 > 1e-6 || fabs(x2 - x0) < 1e-6) flag_x_b = true; else flag_x_b = false;

									bool flag_y_a;
									bool flag_y_b;
									//y2 > y0 > y1 or y1 > y0 > y2 or endpoint
									if (y0 - y1 > 1e-6 || fabs(y0 - y1) < 1e-6) flag_y_a = true; else flag_y_a = false;
									if (y2 - y0 > 1e-6 || fabs(y2 - y0) < 1e-6) flag_y_b = true; else flag_y_b = false;

									bool flag_z_a;
									bool flag_z_b;
									//z2 > z0 > z1 or z1 > z0 > z2 or endpoint
									if (z0 - z1 > 1e-6 || fabs(z0 - z1) < 1e-6) flag_z_a = true; else flag_z_a = false;
									if (z2 - z0 > 1e-6 || fabs(z2 - z0) < 1e-6) flag_z_b = true; else flag_z_b = false;

									//brute force set if the x, y, or z turns out to be the same
									if (fabs(x1 - x2) < endpoint_dist_threshold_)
									{
										flag_x_a = true;
										flag_x_b = true;
									}
									if (fabs(y1 - y2) < endpoint_dist_threshold_)
									{
										flag_y_a = true;
										flag_y_b = true;
									}
									if (fabs(z1 - z2) < endpoint_dist_threshold_)
									{
										flag_z_a = true;
										flag_z_b = true;
									}

									if (line_width_ - d > 1e-6) //distance from point to line within threshold
																//next step: consider occlusion
									{
										if (flag_x_a == flag_x_b && flag_y_a == flag_y_b && flag_z_a == flag_z_b)
										{
											if (color_label_option_ == 0)
											{
												top_data[Tid + tj * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = 1.0; //voxel occupied
											}
										}
									}
								}
								else
								{
									double fz = (x2 - x1) * (x2 - x0) + (y2 - y1) * (y2 - y0) + (z2 - z1) * (z2 - z0);
									double fm = pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2) + 1e-8;
									double r = fz / fm;
									double d = sqrt(pow(x0 - (r * x1 + (1 - r) * x2), 2) + pow(y0 - (r * y1 + (1 - r) * y2), 2) + pow(z0 - (r * z1 + (1 - r) * z2), 2));
									if (r > 1e-6 && r <= 1.0)
									{
										double prob = exp(-gamma_ * d);
										top_data[Tid + tj * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = prob;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_2d_data = bottom[0]->cpu_data(); //gt joint 2d [0,  1]
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_2d_diff = bottom[0]->mutable_cpu_diff();
		Dtype* bottom_depth_diff = bottom[1]->mutable_cpu_diff();
		if (perform_backprop_)
		{
			for (int t = 0; t < batSize; t++)
			{
				int Jid = t * JointNum * 2;
				int Did = t * JointNum;
				int Tid = t * (JointNum * (JointNum - 1) / 2) * depth_dims_ * map_size_ * map_size_;
				//for (int j = 0; j < BoneNum; j++) 
				//clear
				for (int j = 0; j < JointNum * 2; j++) bottom_2d_diff[Jid + j] = 0.0;
				for (int j = 0; j < JointNum; j++) bottom_depth_diff[Did + j] = 0.0;

				int tj = -1;
				//for each bone
				for (int u = 0; u < JointNum - 1; u++)
				{
					for (int v = u + 1; v < JointNum; v++)
					{
						tj++;
						for (int k = 0; k < depth_dims_; k++)
						{
							for (int row = 0; row < map_size_; row++)
							{
								for (int col = 0; col < map_size_; col++)
								{
									double x0 = double(col) / double(map_size_) * (x_ub_ - x_lb_) + x_lb_;
									double y0 = double(row) / double(map_size_) * (y_ub_ - y_lb_) + y_lb_;
									double z0 = double(k) / double(depth_dims_) * (z_ub_ - z_lb_) + z_lb_;
									//====two endpoints 
									double x1 = gt_joint_2d_data[Jid + u * 2];
									double x2 = gt_joint_2d_data[Jid + v * 2];
									double y1 = gt_joint_2d_data[Jid + u * 2 + 1];
									double y2 = gt_joint_2d_data[Jid + v * 2 + 1];
									double z1 = gt_depth_data[Did + u];
									double z2 = gt_depth_data[Did + v];
									if (dot_product_) //exponential representation
									{
										double fz = (x2 - x1) * (x2 - x0) + (y2 - y1) * (y2 - y0) + (z2 - z1) * (z2 - z0);
										double fm = pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2) + 1e-8;
										double r = fz / fm;
										double t = pow(x0 - (r * x1 + (1 - r) * x2), 2) + pow(y0 - (r * y1 + (1 - r) * y2), 2) + pow(z0 - (r * z1 + (1 - r) * z2), 2);
										double d = sqrt(t);
										if (r > 1e-6 && r <= 1.0) //within segment
										{
											double prob = exp(-gamma_ * d);
											//top_data[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = prob;
											double top_diff_value = top_diff[Tid + tj * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col];

											//r = fz / fm drdsymbol = (dfzdsymbol * fm - fz * dfmdsymbol) / fm^2
											double dfzdx1 = -1 * (x2 - x0);
											double dfzdx2 = 1 * (x2 - x0) + (x2 - x1) * 1;

											double dfzdy1 = -1 * (y2 - y0);
											double dfzdy2 = 1 * (y2 - y0) + (y2 - y1) * 1;

											double dfzdz1 = -1 * (z2 - z0);
											double dfzdz2 = 1 * (z2 - z0) + (z2 - z1) * 1;

											double dfmdx1 = 2 * (x2 - x1) * -1;
											double dfmdx2 = 2 * (x2 - x1) * 1;

											double dfmdy1 = 2 * (y2 - y1) * -1;
											double dfmdy2 = 2 * (y2 - y1) * 1;

											double dfmdz1 = 2 * (z2 - z1) * -1;
											double dfmdz2 = 2 * (z2 - z1) * 1;

											double drdx1 = 1.0 / pow(fm, 2) * (dfzdx1 * fm - fz * dfmdx1);
											double drdx2 = 1.0 / pow(fm, 2) * (dfzdx2 * fm - fz * dfmdx2);

											double drdy1 = 1.0 / pow(fm, 2) * (dfzdy1 * fm - fz * dfmdy1);
											double drdy2 = 1.0 / pow(fm, 2) * (dfzdy2 * fm - fz * dfmdy2);

											double drdz1 = 1.0 / pow(fm, 2) * (dfzdz1 * fm - fz * dfmdz1);
											double drdz2 = 1.0 / pow(fm, 2) * (dfzdz2 * fm - fz * dfmdz2);
											//inside sqrt symbol
											double dtdx1 = 2.0 * (x0 - (r * x1 + (1 - r) * x2)) * (-drdx1 * x1 - r);
											double dtdx2 = 2.0 * (x0 - (r * x1 + (1 - r) * x2)) * (drdx2 * x2 + (r - 1));

											double dtdy1 = 2.0 * (y0 - (r * y1 + (1 - r) * y2)) * (-drdy1 * y1 - r);
											double dtdy2 = 2.0 * (y0 - (r * y1 + (1 - r) * y2)) * (drdy2 * y2 + (r - 1));

											double dtdz1 = 2.0 * (z0 - (r * z1 + (1 - r) * z2)) * (-drdz1 * z1 - r);
											double dtdz2 = 2.0 * (z0 - (r * z1 + (1 - r) * z2)) * (drdz2 * z2 + (r - 1));

											//d d / d ... 
											double dddx1 = 1.0 / (2.0 * d + 1e-8) * dtdx1;
											double dddx2 = 1.0 / (2.0 * d + 1e-8) * dtdx2;

											double dddy1 = 1.0 / (2.0 * d + 1e-8) * dtdy1;
											double dddy2 = 1.0 / (2.0 * d + 1e-8) * dtdy2;

											double dddz1 = 1.0 / (2.0 * d + 1e-8) * dtdz1;
											double dddz2 = 1.0 / (2.0 * d + 1e-8) * dtdz2;

											//d loss / d symbol = d loss / d top * d top / d symbol
											//d top / d symbol = d(exp(-gamma * distance)) / d symbol
											//                 = exp(-gamma * distance) * -gamma * d distance / d symbol
											//                 = prob * dddsymbol
											// Final = top_diff_value * prob * -gamma dddsymbol
											double dldx1 = top_diff_value * prob * -gamma_ * dddx1;
											double dldx2 = top_diff_value * prob * -gamma_ * dddx2;

											double dldy1 = top_diff_value * prob * -gamma_ * dddy1;
											double dldy2 = top_diff_value * prob * -gamma_ * dddy2;

											double dldz1 = top_diff_value * prob * -gamma_ * dddz1;
											double dldz2 = top_diff_value * prob * -gamma_ * dddz2;
											bottom_2d_diff[Jid + u * 2] = dldx1;
											bottom_2d_diff[Jid + v * 2] = dldx2;

											bottom_2d_diff[Jid + u * 2 + 1] = dldy1;
											bottom_2d_diff[Jid + v * 2 + 1] = dldy2;

											bottom_depth_diff[Did + u] = dldz1;
											bottom_depth_diff[Did + v] = dldz2;
										}
									}
								}
							}
						}
					}
				}				
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGen3DSkeletonMapPerChannelAllPairsLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGen3DSkeletonMapPerChannelAllPairs);
}

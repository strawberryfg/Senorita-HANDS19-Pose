#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"
#define max_distance 1111
namespace caffe {

	template <typename Dtype>
	void DeepHandModelProjectDepthLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		depth_size_ = this->layer_param_.read_depth_no_bbx_with_avgz_param().depth_size();

		minus_pixel_value_ = this->layer_param_.transform_param().minus_pixel_value();

		is_divide_ = this->layer_param_.transform_param().is_divide();
	}

	template <typename Dtype>
	void DeepHandModelProjectDepthLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//augmented projection onto XY plane
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top_shape.push_back(depth_size_);
		top_shape.push_back(depth_size_);
		top[0]->Reshape(top_shape);

		//projection onto ZY plane
		top[1]->Reshape(top_shape);

		//projection onto ZX plane
		top[2]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelProjectDepthLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* point_cloud_data = bottom[0]->cpu_data();
		const Dtype* x_lb_data = bottom[1]->cpu_data();
		const Dtype* x_ub_data = bottom[2]->cpu_data();
		const Dtype* y_lb_data = bottom[3]->cpu_data();
		const Dtype* y_ub_data = bottom[4]->cpu_data();
		const Dtype* z_lb_data = bottom[5]->cpu_data();
		const Dtype* z_ub_data = bottom[6]->cpu_data();

		const Dtype* avgx_data = bottom[7]->cpu_data(); 
		const Dtype* avgy_data = bottom[8]->cpu_data();
		const Dtype* avgz_data = bottom[9]->cpu_data();


		Dtype* top_xy_data = top[0]->mutable_cpu_data();
		Dtype* top_zy_data = top[1]->mutable_cpu_data();
		Dtype* top_zx_data = top[2]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		int point_num_ = (bottom[0]->shape())[1] / 3;
		for (int t = 0; t < batSize; t++) {
			double x_lb = x_lb_data[t];
			double x_ub = x_ub_data[t];
			double y_lb = y_lb_data[t];
			double y_ub = y_ub_data[t];
			double z_lb = z_lb_data[t];
			double z_ub = z_ub_data[t];

			double avgx = avgx_data[t];
			double avgy = avgy_data[t];
			double avgz = avgz_data[t];

			int Tid = t * depth_size_ * depth_size_;
			
			for (int row = 0; row < depth_size_; row++)
			{
				for (int col = 0; col < depth_size_; col++)
				{
					top_xy_data[Tid + row * depth_size_ + col] = max_distance;
					top_zy_data[Tid + row * depth_size_ + col] = max_distance;
					top_zx_data[Tid + row * depth_size_ + col] = max_distance;
				}
			}
			//printf("done!\n");
			for (int i = 0; i < point_num_; i++)
			{
				//printf("point ", i);
				//point_num_ = depth_size_ * depth_size_
				int Bid = t * point_num_ * 3;
				double x = point_cloud_data[Bid + i];
				double y = point_cloud_data[Bid + depth_size_ * depth_size_ + i];
				double z = point_cloud_data[Bid + 2 * depth_size_ * depth_size_ + i];

				//printf("%12.6f %12.6f %12.6f\n", x, y, z);
				double u, v, d;
				int row, col;
				u = min(1.0, max(0.0, (x - x_lb) / (x_ub - x_lb))) * depth_size_;
				v = min(1.0, max(0.0, (y - y_lb) / (y_ub - y_lb))) * depth_size_;
				d = min(1.0, max(0.0, (z - z_lb) / (z_ub - z_lb))) * depth_size_;
				//printf(" %12.6f %12.6f %12.6f\n", u, v, d);

				//XY plane
				col = (int)u; row = (int)v;
				//min(z)
				if (row >= 0 && row < depth_size_ && col >= 0 && col < depth_size_)
				{
					if (top_xy_data[Tid + row * depth_size_ + col] - z > 1e-6) top_xy_data[Tid + row * depth_size_ + col] = z;
				}
				//printf("xy\n");
				//ZY plane
				col = (int)d; row = (int)v;
				//min(-x)
				if (row >= 0 && row < depth_size_ && col >= 0 && col < depth_size_)
				{
					if (top_zy_data[Tid + row * depth_size_ + col] - (-x) > 1e-6) top_zy_data[Tid + row * depth_size_ + col] = -x;
				}
				//printf("zy\n");
				//ZX plane
				col = (int)d; row = (int)u;
				//min(y)
				if (row >= 0 && row < depth_size_ && col >= 0 && col < depth_size_)
				{
					if (top_zx_data[Tid + row * depth_size_ + col] - y > 1e-6) top_zx_data[Tid + row * depth_size_ + col] = y;
				}
				//printf("zx\n");

			}

			//take minimum distance
			//printf("taking\n");
			
			for (int row = 0; row < depth_size_; row++)
			{
				for (int col = 0; col < depth_size_; col++)
				{
					double cur_d, avg_d;
					
					//XY plane
					cur_d = top_xy_data[Tid + row * depth_size_ + col];
					//std::cout << row << " " << col << " " << Tid <<" " << cur_d << " xy\n";

					avg_d = avgz;
					//std::cout << avg_d << "\n";
					if (cur_d >= z_lb && cur_d < z_ub)
					{
						cur_d = (-(double(cur_d - avg_d) / double(100.0)) + 1.0) / 2.0 * 255;						
					}
					else
					{
						cur_d = 0; //black pixel 
					}				
					//std::cout << row << " " << col << " " << Tid << " " << cur_d << " xy\n";
					top_xy_data[Tid + row * depth_size_ + col] = (cur_d - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);
					//std::cout << row << " " << col << " " << Tid << " " << top_xy_data[Tid + row * depth_size_ + col] << " xy\n";
					//std::cout << "hello\n";
					//ZY plane
					cur_d = top_zy_data[Tid + row * depth_size_ + col];
					//std::cout << row << " " << col << " " << cur_d << " zy\n";

					avg_d = -avgx; //avg(-x) = -avg(x)
					if (cur_d >= -x_ub && cur_d < -x_lb) cur_d = (-(double(cur_d - avg_d) / double(100.0)) + 1.0) / 2.0 * 255; //black pixel 
					else cur_d = 0;
					top_zy_data[Tid + row * depth_size_ + col] = (cur_d - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);

					//ZX plane
					cur_d = top_zx_data[Tid + row * depth_size_ + col];
					//std::cout << row << " " << col << " " << cur_d << " zx\n";

					avg_d = avgy;
					if (cur_d >= y_lb && cur_d < y_ub) cur_d = (-(double(cur_d - avg_d) / double(100.0)) + 1.0) / 2.0 * 255; //black pixel 
					else cur_d = 0;
					top_zx_data[Tid + row * depth_size_ + col] = (cur_d - minus_pixel_value_) / (is_divide_ ? 256.0 : 1.0);
				
				}
			}
		}
	}


	template <typename Dtype>
	void DeepHandModelProjectDepthLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelProjectDepthLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelProjectDepthLayer);
	REGISTER_LAYER_CLASS(DeepHandModelProjectDepth);
}  // namespace caffe

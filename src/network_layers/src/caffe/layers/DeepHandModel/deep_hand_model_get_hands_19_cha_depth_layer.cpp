#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
Mat handMat;
//#define _DEBUG

//Challenge (2017 2019) dataset
double avgX = 0, avgY = 0, avgZ = 0, avgU = 0, avgV = 0, avgX_real = 0, avgY_real = 0, avgZ_real = 0, avgU_real = 0, avgV_real = 0;
double mmcp_x, mmcp_y;
int vis[480][640];
int cur_round = 0;
struct queue_type
{
	int x, y;
}q[maxq];
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGetHands19ChaDepthLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		crop_size_ = this->layer_param_.transform_param().crop_size();
		file_prefix_ = this->layer_param_.read_blob_from_file_indexing_param().file_prefix(); //image prefix default "../../Task 1/"
		depth_threshold_ = this->layer_param_.deep_hand_model_hands19_param().depth_threshold();
		img_prefix_ = this->layer_param_.deep_hand_model_hands19_param().img_prefix();
		//output additional 640x480 depth
		o_add_depth_ = this->layer_param_.deep_hand_model_hands19_param().o_add_depth();
	}
	template <typename Dtype>
	void DeepHandModelGetHands19ChaDepthLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		//depth handMat
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1); //grayscale
		top_shape.push_back(crop_size_);
		top_shape.push_back(crop_size_);
		top[0]->Reshape(top_shape);
		//extendrect bbx_x1 bbx_y1 bbx_x2 bbx_y2
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1); 
		top[1]->Reshape(top_shape);
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[2]->Reshape(top_shape);
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[3]->Reshape(top_shape);
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[4]->Reshape(top_shape);
		//avgX
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[5]->Reshape(top_shape);
		//avgY
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[6]->Reshape(top_shape);
		//avgZ
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[7]->Reshape(top_shape);
		//avgU
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[8]->Reshape(top_shape);
		//avgV
		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[9]->Reshape(top_shape);

		if (o_add_depth_)
		{
			//additional 640x480 with only hand mat
			top_shape.clear();
			top_shape.push_back((bottom[0]->shape())[0]);
			top_shape.push_back(1);
			top_shape.push_back(480);
			top_shape.push_back(640);
			top[10]->Reshape(top_shape);

		}
		

	}

	template <typename Dtype>
	void DeepHandModelGetHands19ChaDepthLayer<Dtype>::reproject2DTo3D(double &x, double &y, const double z)
	{
		x = z * (x - u0) / fx;
		y = z * (y - v0) / fy;
	}

	template <typename Dtype>
	void DeepHandModelGetHands19ChaDepthLayer<Dtype>::Project3DTo2D(double &x, double &y, const double z)
	{
		x = x / z * fx + u0;
		y = y / z * fy + v0;
	}


	template <typename Dtype>
	bool DeepHandModelGetHands19ChaDepthLayer<Dtype>::calcBoxCenter(const cv::Mat &handMat, int x, int y, double &avgX, double &avgY, double &avgZ, double &avgU, double &avgV, int opt)
	{
		double accumX, accumY, accumZ, count;
		accumX = accumY = accumZ = count = 0.0;
		for (int row = 0; row < handMat.rows; ++row)
		{
			for (int col = 0; col < handMat.cols; ++col)
			{
				ushort pixel = handMat.at<ushort>(row, col);
				//invalid pixel
				if (pixel < 1)
				{
					continue;
				}
				//(u, v) in the image plane
				double x3D = static_cast<double>(x + col);
				double y3D = static_cast<double>(y + row);
				//directly accumulate depth
				double z = static_cast<double>(pixel);

				accumX += x3D;
				accumY += y3D;
				accumZ += z;
				count += 1.0;
			}
		}
		if (count < 1)
		{
			return false;
		}
		//the average of (u, v)
		avgX = accumX / double(count);
		avgY = accumY / double(count);
		avgZ = accumZ / double(count);
		avgU = avgX;
		avgV = avgY;
		if (opt == 1) //for real
		{
			reproject2DTo3D(avgX, avgY, avgZ);
		}
		else //only avgZ is useful
		{
			avgZ = handMat.at<ushort>(int(handMat.rows / 2), int(handMat.cols / 2)); //use center point provided by official bbx
			//cout << "AVGZ " << avgZ << "\n";
			if (avgZ == 0) //BFS to find nearest non-zero pixel
			{
				int head = 0, tail = 0;
				q[0].x = handMat.cols / 2; q[0].y = handMat.rows / 2;
				vis[q[0].y][q[0].x] = cur_round;
				bool found = false;
				while (head <= tail)
				{
					for (int k = 0; k < 4; k++)
					{
						int tx = q[head].x + dx[k], ty = q[head].y + dy[k];
						if (tx >= 0 && tx < handMat.cols && ty >= 0 && ty < handMat.rows && vis[ty][tx] != cur_round)
						{
							q[++tail].x = tx; q[tail].y = ty;
							vis[ty][tx] = cur_round;
							if (handMat.at<ushort>(ty, tx))
							{
								avgZ = handMat.at<ushort>(ty, tx);
								found = true;
								break;
							}
						}
					}
					if (found) break;
					head++;
					//cout << "Head: " << head << "\n";
				}
				if (!found) avgZ = 500.0;
			}
			
			//cout << "avgZ found : " << avgZ << "\n";
		}
		return true;
	}

	template <typename Dtype>
	ushort DeepHandModelGetHands19ChaDepthLayer<Dtype>::drawHistogram(const Mat &handMat, int opt)
	{
		ushort minPixel = numeric_limits<ushort>::max();
		ushort maxPixel = 0;
		for (int row = 0; row < handMat.rows; ++row)
		{
			for (int col = 0; col < handMat.cols; ++col)
			{
				ushort pixel = handMat.at<ushort>(row, col);
				if (pixel == 0)
				{
					continue;
				}
				minPixel = minPixel > pixel ? pixel : minPixel;
				maxPixel = maxPixel < pixel ? pixel : maxPixel;
			}
		}
		int len = maxPixel - minPixel + 1;
		if (len <= 0)
		{
			return 0;
		}
		vector<int> vect(len, 0);
		for (int row = 0; row < handMat.rows; ++row)
		{
			for (int col = 0; col < handMat.cols; ++col)
			{
				ushort pixel = handMat.at<ushort>(row, col);
				if (pixel == 0)
				{
					continue;
				}
				int val = pixel - minPixel;
				++vect[val];
			}
		}
		int maxVal = 0;
		for (auto it : vect)
		{
			maxVal = it > maxVal ? it : maxVal;
		}
		Mat graph(maxVal + 1, len, CV_8UC1, Scalar(0));

		int ind = 0;
		for (int i = 0; i < vect.size(); ++i)
		{
			int col = i;
			int row = maxVal - vect[i];
			if (vect[i] == maxVal) ind = i;
			graph.at<uchar>(row, col) = 255;
		}
		
		double sum = 0.0;
		for (int i = 0; i < vect.size(); ++i)
		{
			sum += vect[i];
		}

		vector<int> localmax(2 * len, 0);
		int local_max_num = 0;
		int window_w = len / 6;
		if (window_w < 10) window_w = len / 3;
		for (int i = 0; i < vect.size(); i++)
		{
			int val_ = vect[i];
			if (val_ < 10) continue;

			bool lmax = true;

			int left = max(i - window_w + 1, 0);
			int right = min(i + window_w - 1, (int)vect.size() - 1);

			for (int j = left; j < i; j++)
			{
				if (val_ < vect[j])
				{
					lmax = false;
					break;
				}
			}
			if (lmax)
			{
				for (int j = i; j <= right; j++)
				{
					if (val_ < vect[j])
					{
						lmax = false;
						break;
					}
				}
			}
			if (lmax)
			{
				localmax[2 * local_max_num + 0] = i;
				localmax[2 * local_max_num + 1] = val_;
				local_max_num++;
			}
		}
		// Accum sum
		double accum = 0.0;
		int windowSum = 0;
		for (int i = 0; i < vect.size();)
		{
			if (vect[i] == 0) {
				int j = 0;
				for (; j < kLastLen; ++j)
				{
					if (vect[i + j] > 5)
					{
						break;
					}
					windowSum += vect[i + j];
				}
				if (j >= kLastLen)
				{
					if (accum / sum > kRatioThreshold
						&& windowSum < kLastLen * 0.4)
					{
						return minPixel + i;
					}
				}
				windowSum = 0;
				i += j + 1;
			}
			else
			{
				accum += vect[i];
				++i;
			}
		}
		//cout << "threshold before= " << 0 << endl;
		// remove background
		if (1)
		{
			int threshold = 0;
			if (local_max_num >= 2 && len > 180)
			{
				int begin = localmax[2 * (local_max_num - 2)];
				if (local_max_num > 3 && len > 380)
					begin = localmax[2 * (local_max_num - 3)];
				int end = localmax[2 * (local_max_num - 1)];
				int len_l = end - begin + 1;
				if (len_l == 0)
					return threshold;

				vector<int> localmin(2 * len_l, 0);
				int local_min_num = 0;
				int window_m = len_l / 4;

				for (int i = begin; i < end; i++)
				{
					int val_ = vect[i];

					if (val_ < 1)
					{
						bool maskm = true;
						int bar_l = min(end, i + 4);
						for (int j = i; j < bar_l; j++)
						{
							if (vect[j] >1)
							{
								maskm = false;
								break;
							}
						}
						if (maskm)
						{
							int suml = 0;
							for (int k = bar_l; k < vect.size(); k++)
							{
								suml += vect[k];
							}
							if (suml < sum * 0.6 && suml > sum * 0.1)
							{
								threshold = minPixel + i;
								return threshold;
							}
							else
								continue;
						}
						else
							continue;
					}

					bool lmin = true;

					int left = max(i - window_m + 1, begin);
					int right = min(i + window_m - 1, end - 1);

					for (int j = left; j < i; j++)
					{
						if (val_ > vect[j])
						{
							lmin = false;
							break;
						}
					}
					if (lmin)
					{
						for (int j = i; j <= right; j++)
						{
							if (val_ > vect[j])
							{
								lmin = false;
								break;
							}
						}
					}
					if (lmin)
					{
						localmin[2 * local_min_num + 0] = i;
						localmin[2 * local_min_num + 1] = val_;
						local_min_num++;
					}
				}
				if (local_min_num > 0)
				{
					int min_loc = localmin[1];
					int ind_loc = 0;
					for (int i = 0; i < local_min_num; i++)
					{
						if (min_loc < localmin[2 * i + 1])
						{
							min_loc = localmin[2 * i + 1];
							ind_loc = i;
						}
					}

					int sumf = 0;
					for (int k = localmin[2 * ind_loc]; k < vect.size(); k++)
					{
						sumf += vect[k];
					}
					if (sumf < sum * 0.6 && sumf > sum * 0.1)
					{
						threshold = minPixel + localmin[2 * ind_loc];
						return threshold;
					}
					else
						return 0;
				}
				else
					return 0;
			}
		}

		return 0;
	}

	template <typename Dtype>
	cv::Mat DeepHandModelGetHands19ChaDepthLayer<Dtype>::convertDepthToRGB(const cv::Mat &depthImg)
	{
		int cols = depthImg.cols;
		int rows = depthImg.rows;
		std::vector<uint32_t>  histogram(kMaxDistance, 0);
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				uint16_t depth = depthImg.at<ushort>(row, col);
				if (depth >= kMaxDistance)
				{
					continue;
				}
				++histogram[depth];
			}
		}
		// Build a cumulative histogram for the indices in [1,0xFFFF]
		for (int i = 2; i < histogram.size(); ++i)
		{
			histogram[i] += histogram[i - 1];
		}
		cv::Mat rgbMat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				uint32_t depth = depthImg.at<ushort>(row, col);
				if (depth >= kMaxDistance)
				{
					continue;
				}
				if (depth != 0)
				{
					// 0-255 based on histogram location
					int val = histogram[depth] * 255 / histogram.back();
					rgbMat.at<cv::Vec3b>(row, col) = cv::Vec3b(255 - val, 0, val);
				}
				else
				{
					rgbMat.at<cv::Vec3b>(row, col) = cv::Vec3b(20, 5, 0);
				}
			}
		}
		return rgbMat;
	}

	template <typename Dtype>
	bool DeepHandModelGetHands19ChaDepthLayer<Dtype>::GetRidOfBackground(vector<double> vect, char imgpath[maxlen], Mat &handMat, Rect &extendRect, int opt)
	{
		Mat img = imread(imgpath, CV_LOAD_IMAGE_ANYDEPTH);
		Mat colorImg = convertDepthToRGB(img);
		circle(colorImg, Point((int)vect[0], (int)vect[1]), 3, Scalar(255, 0, 0), 2);
		circle(colorImg, Point((int)vect[2], (int)vect[1]), 3, Scalar(0, 255, 0), 2);
		circle(colorImg, Point((int)vect[0], (int)vect[3]), 3, Scalar(255, 0, 255), 2);
		circle(colorImg, Point((int)vect[2], (int)vect[3]), 3, Scalar(255, 255, 255), 2);
		Rect rect((int)vect[0], (int)vect[1], (int)(vect[2] - vect[0]), (int)(vect[3] - vect[1]));
		rectangle(colorImg, rect, Scalar(255, 255, 255));
		int x = vect[0];
		int y = vect[1];
		int w = vect[2] - vect[0];
		int h = vect[3] - vect[1];
		int xCenter = x + w / 2;
		int yCenter = y + h / 2;
		int newEdge = w > h ? w : h;
		w = h = newEdge;
		x = xCenter - w / 2;
		y = yCenter - h / 2;
		vect[0] = x;
		vect[1] = y;
		vect[2] = w;
		vect[3] = h;
		extendRect = Rect((int)vect[0], (int)vect[1], (int)(vect[2]), (int)(vect[3]));
		rectangle(colorImg, extendRect, Scalar(255, 0, 255));
		circle(colorImg, Point(mmcp_x, mmcp_y), 6, Scalar(0, 255, 0), -3);
		circle(colorImg, Point(extendRect.x + extendRect.width / 2.0, extendRect.y + extendRect.height / 2.0), 3, Scalar(255, 255, 255), -1);
		handMat = Mat::zeros(Size(extendRect.width, extendRect.height), CV_16UC1);
		for (int row = extendRect.y; row < extendRect.y + extendRect.height; row++)
		{
			for (int col = extendRect.x; col < extendRect.x + extendRect.width; col++)
			{
				if (row >= 0 && row < 480 && col >= 0 && col < 640)
				{
					handMat.at<ushort>(row - extendRect.y, col - extendRect.x) = img.at<ushort>(row, col);
				}
			}
		}

		if (opt == 1) //simple depth thresholding; assumption: hand is the closest object to the camera
		{
			for (int row = 0; row < handMat.rows; row++)
			{
				for (int col = 0; col < handMat.cols; col++)
				{
					int cur_d = handMat.at<ushort>(row, col);
					if (cur_d < avgZ - depth_threshold_ || cur_d > avgZ + depth_threshold_)
					{
						handMat.at<ushort>(row, col) = 0;
					}
				}
			}
		}
		Mat colorHandMat = convertDepthToRGB(handMat);
		//------ Plot gray mask, draw histogram and eliminate the cluttered background 
		ushort threshold = drawHistogram(handMat, opt);
		//cout << "threshold = " << threshold << endl;

		ushort minPixel = USHRT_MAX;
		ushort maxPixel = 0;
		for (int row = 0; row < handMat.rows; ++row)
		{
			for (int col = 0; col < handMat.cols; ++col)
			{
				ushort pixel = handMat.at<ushort>(row, col);
				if (pixel < 1)
				{
					continue;
				}
				minPixel = minPixel > pixel ? pixel : minPixel;
				maxPixel = maxPixel < pixel ? pixel : maxPixel;
			}
		}

		//Gray-scale cropped hand
		double range = maxPixel - minPixel;
		Mat onlyHand(handMat.rows, handMat.cols, CV_8UC1, Scalar(0));
		for (int row = 0; row < handMat.rows; ++row)
		{
			for (int col = 0; col < handMat.cols; ++col)
			{
				ushort pixel = handMat.at<ushort>(row, col);
				if (pixel < 1)
				{
					continue;
				}
				double diff = pixel - minPixel;
				double val = 255.0 * diff / range;
				onlyHand.at<uchar>(row, col) = static_cast<uchar>(val);
			}
		}
		Mat mask;
		double bar = cv::threshold(onlyHand, mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		Mat handMatCopy;
		handMat.copyTo(handMatCopy);
		for (int row = 0; row < mask.rows; ++row)
		{
			for (int col = 0; col < mask.cols; ++col)
			{
				uchar pixel = mask.at<uchar>(row, col);
				if (pixel < 1)
				{
					continue;
				}
				handMatCopy.at<ushort>(row, col) = 0;
			}
		}
		Mat rgbHandMatCopy = convertDepthToRGB(handMatCopy);
		if (threshold > 0)
		{
			for (int row = 0; row < handMat.rows; ++row)
			{
				for (int col = 0; col < handMat.cols; ++col)
				{
					ushort pixel = handMat.at<ushort>(row, col);
					if (pixel > threshold)
					{
						handMat.at<ushort>(row, col) = 0;
					}
				}
			}
		}

		//Outlier
		if (threshold > 0)
		{
			for (int row = 0; row < handMat.rows; ++row)
			{
				for (int col = 0; col < handMat.cols; ++col)
				{
					ushort pixel = handMat.at<ushort>(row, col);
					if (pixel > threshold)
					{
						handMat.at<ushort>(row, col) = 0;
					}
				}
			}
		}


		//------ Get rid of background
		Mat getRidBackground = convertDepthToRGB(handMat);
		return true;
	}

	template <typename Dtype>
	void DeepHandModelGetHands19ChaDepthLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bbx_x1_data = bottom[0]->cpu_data();  
		const Dtype* bbx_y1_data = bottom[1]->cpu_data();
		const Dtype* bbx_x2_data = bottom[2]->cpu_data();
		const Dtype* bbx_y2_data = bottom[3]->cpu_data();
		const Dtype* mmcp_x_data = bottom[4]->cpu_data();
		const Dtype* mmcp_y_data = bottom[5]->cpu_data();
		const Dtype* index_data = bottom[6]->cpu_data();

		Dtype* depth_data = top[0]->mutable_cpu_data();
		Dtype* ext_bbx_x1_data = top[1]->mutable_cpu_data();
		Dtype* ext_bbx_y1_data = top[2]->mutable_cpu_data();
		Dtype* ext_bbx_x2_data = top[3]->mutable_cpu_data();
		Dtype* ext_bbx_y2_data = top[4]->mutable_cpu_data();
		Dtype* avgX_data = top[5]->mutable_cpu_data();
		Dtype* avgY_data = top[6]->mutable_cpu_data();
		Dtype* avgZ_data = top[7]->mutable_cpu_data();
		Dtype* avgU_data = top[8]->mutable_cpu_data();
		Dtype* avgV_data = top[9]->mutable_cpu_data();

		Dtype* raw_depth_data;
		if (o_add_depth_) raw_depth_data = top[10]->mutable_cpu_data();

		for (int t = 0; t < batSize; t++)
		{
			double bbx_x1 = bbx_x1_data[t];
			double bbx_y1 = bbx_y1_data[t];
			double bbx_x2 = bbx_x2_data[t];
			double bbx_y2 = bbx_y2_data[t];
			mmcp_x = mmcp_x_data[t];
			mmcp_y = mmcp_y_data[t];
			bool is_train = (this->phase_ == TRAIN);
			int img_id = index_data[t];
			bool is_synth = (img_id >= 175951);
			char imgpath[maxlen], imgpath_real[maxlen];
			if (is_synth)
			{
				sprintf(imgpath_real, "%s%s%08d.png", file_prefix_.c_str(), img_prefix_.c_str(), img_id - 175951);
				sprintf(imgpath, "%s%s%08d.png", file_prefix_.c_str(), img_prefix_.c_str(), img_id); //synth
			}
			else
			{
				sprintf(imgpath_real, "%s%s%08d.png", file_prefix_.c_str(), img_prefix_.c_str(), img_id);
				sprintf(imgpath, "%s%s%08d.png", file_prefix_.c_str(), img_prefix_.c_str(), img_id + 175951); //synth
			}

			


			vector<double> vect;
			vect.push_back(bbx_x1);
			vect.push_back(bbx_y1);
			vect.push_back(bbx_x2);
			vect.push_back(bbx_y2);

			//SHARED VARIABLE DECLARATION
			Rect extendrect_real;
			bool success, success_real;
			bool flag, flag_real;
			double scale_real;

			//if (!is_synth) //Real depth image; use previous version 
			{

				success_real = GetRidOfBackground(vect, imgpath_real, handMat, extendrect_real, 0);

				//cout << t << " " << 1 << "\n";

				//cout << "a" << "\n";
				//cout << extendrect.x << " " << extendrect.y << " " << extendrect.width << " " << extendrect.height << "\n";
				avgX_real = avgY_real = 0.0;
				avgZ_real = 500.0;
				cur_round++;
				flag_real = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX_real, avgY_real, avgZ_real, avgU_real, avgV_real, 0);
				//avgX, avgY, avgU, avgV are not used
				//cout << "b" << "\n";
				//cout << t << " " << 2 << "\n";

				scale_real = mean_depth / avgZ_real * mean_scale; //(img_id < 175951 ? mean_scale : 263.0);

				//explicitly reset
				avgU_real = extendrect_real.x + extendrect_real.width / 2.0;
				avgV_real = extendrect_real.y + extendrect_real.height / 2.0;
				vect[0] = avgU_real - scale_real / 2.0;
				vect[1] = avgV_real - scale_real / 2.0;
				vect[2] = avgU_real + scale_real / 2.0;
				vect[3] = avgV_real + scale_real / 2.0;

				//narrow the bbx USE CENTER AVGZ TO FILTER OUT
				avgZ = avgZ_real; //FOR ELIMINATING OUTLIERS
				success_real = GetRidOfBackground(vect, imgpath_real, handMat, extendrect_real, 1);
				//if (!success_real) continue;
				//cout << "c" << "\n";
				//After depth thresholding -> recalculate centroid
				
				//cout << t << " " << 3 << "\n";

				//True centroid
				cur_round++;
				flag_real = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX_real, avgY_real, avgZ_real, avgU_real, avgV_real, 1);
				//cout << t << " " << 4 << "\n";
			}
			
			if (is_synth) //After doing real
			{
				//READ SYNTH IMAGE
				success = GetRidOfBackground(vect, imgpath, handMat, extendrect_real, 0);
				cur_round++;
				//cout << t << " " << 5 << "\n";

				//WE'LL JUST USE THE CENTER AVGZ
				flag = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX, avgY, avgZ, avgU, avgV, 0);
				//USE CENTER AVGZ TO REMOVE OUTLIER
				
				//cout << t << " " << 6 << "\n";
				success = GetRidOfBackground(vect, imgpath, handMat, extendrect_real, 1);

				//cout << t << " " << 7 << "\n";
				//RECALCULATE TRUE CENTROID OF SYNTH
				cur_round++;
				flag = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX, avgY, avgZ, avgU, avgV, 1);
				//cout << t << " " << 8 << "\n";
			}

			//cout << t << " " << 9 << "\n";
			//OUTPUT ORIGINAL AND CROPPED DEPTH
			int Tid = t * 1 * crop_size_ * crop_size_;
			Mat norm_depth = Mat::zeros(Size(handMat.rows, handMat.cols), CV_8UC1);
			if (o_add_depth_)
			{
				for (int row = 0; row < 480; row++)
				{
					for (int col = 0; col < 640; col++)
					{
						int Aid = t * 1 * 480 * 640;
						raw_depth_data[Aid + row * 640 + col] = 0.0;
					}
				}
			}
			//cout << t << " " << 10 << "\n";
			//cout << "d" << "\n";
			for (int row = 0; row < handMat.rows; row++)
			{
				for (int col = 0; col < handMat.cols; col++)
				{
					int cur_d = handMat.at<ushort>(row, col);
					if (cur_d != 0)
					{
						int cur_col = (-(double(cur_d - (is_synth ? avgZ: avgZ_real)) / double(100.0)) + 1.0) / 2.0 * 255;
						norm_depth.at<uchar>(row, col) = cur_col;

						if (o_add_depth_)
						{
							int Aid = t * 1 * 480 * 640;
							int raw_row = extendrect_real.y + row;
							int raw_col = extendrect_real.x + col;
							if (raw_row >= 0 && raw_row < 480 && raw_col >= 0 && raw_col < 640)
							{
								raw_depth_data[Aid + raw_row * 640 + raw_col] = (-(double(cur_d - (is_synth ? avgZ: avgZ_real)) / double(100.0)) + 1.0) / 2.0;
								//[0, 1]
							}
						}
					}
				}
			}
			//cout << t << " " << 11 << "\n";
			//cout << "e" << "\n";
			//cout << "done\n";
			resize(norm_depth, norm_depth, Size(crop_size_, crop_size_));
			for (int row = 0; row < crop_size_; row++)
			{
				for (int col = 0; col < crop_size_; col++)
				{
					depth_data[Tid + row * crop_size_ + col] = norm_depth.at<uchar>(row, col) / 255.0;
				}
			}
			//cout << t << " " << 12 << "\n";
			//cout << "f" << "\n";
			Tid = t;
			ext_bbx_x1_data[Tid] = extendrect_real.x;
			ext_bbx_y1_data[Tid] = extendrect_real.y;
			ext_bbx_x2_data[Tid] = extendrect_real.x + extendrect_real.width; 
			ext_bbx_y2_data[Tid] = extendrect_real.y + extendrect_real.height;
			avgX_data[Tid] = (is_synth ? avgX: avgX_real);
			avgY_data[Tid] = (is_synth ? avgY: avgY_real);
			avgZ_data[Tid] = (is_synth ? avgZ: avgZ_real);
			avgU_data[Tid] = (is_synth ? avgU: avgU_real);
			avgV_data[Tid] = (is_synth ? avgV: avgV_real);
			//cout << t << " " << 13 << "\n";
		}
	}

	template <typename Dtype>
	void DeepHandModelGetHands19ChaDepthLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGetHands19ChaDepthLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGetHands19ChaDepthLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGetHands19ChaDepth);
}

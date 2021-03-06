#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <ctime>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "convertDepthToRGB.h"
#include "HandModel.h"
#define maxlen 111
#define maxq 11111111
#define before 0
#define after 1
#define mean_depth 500.0
//#define mean_scale 263.0
#define mean_scale 210.0

#define crop_size_ 256
const int dx[4] = {-1, 0, 0, 1}; 
const int dy[4] = {0, -1, 1, 0};

//#define use_mean_scale
//#define DEBUG

//#define save_mmcp
#define save_jt_annot
#define save_bbx
#define crop_depth
#define save_crop

#define train_num 175951
#define JointNum_challenge 21

float gt_joint_3d_global[train_num][JointNum_challenge * 3];

#define train_annotation_file "../Task 1/training_joint_annotation.txt"
#define wait_key
#define use_my_calc_centroid
#define run_on_train
#ifdef run_on_train
#define rough_bbx_train_file "../Task 1/training_bbs.txt"
#define save_bbx_prefix "../Task 1/training/bbx/"
#define save_mmcp_prefix "../Task 1/training/mmcp/"
#define save_mmcp_3d_prefix "../Task 1/training/mmcp3d/"
#define save_jt_annot_prefix "../Task 1/training/gt3d/"
#define save_crop_prefix "../Task 1/training_images_crop/"
#else
#define rough_bbx_test_file "../Task 1/test_bbs.txt"
#define save_bbx_prefix "../Task 1/test/bbx/"
#define save_crop_prefix "../Task 1/test_images_crop/"
#endif
using namespace std;
using namespace cv;

const int kLastLen = 10;
const float kRatioThreshold = 0.2;
const float kCubicHalfLen = 150.0;

const float u0 = 315.944855;
const float v0 = 245.287079;
const float fx = 475.065948;
const float fy = 475.065857;
float avgX, avgY, avgZ, avgU, avgV;
float mmcp_x, mmcp_y, mmcp_z;

int vis[480][640];
int cur_round = 0;
int index = 0;

#define start_id 0
#define end_id 175952
struct queue_type
{
	int x, y;
}q[maxq];
void reproject2DTo3D(float &x, float &y, const float z)
{
	x = z * (x - u0) / fx;
	y = z * (y - v0) / fy;
}

void Project3DTo2D(float &x, float &y, const float z)
{
	x = x / z * fx + u0;
	y = y / z * fy + v0;
}


bool calcBoxCenter(const cv::Mat &handMat,
	int x, int y, float &avgX, float &avgY, float &avgZ, float &avgU, float &avgV, int opt) 
{
	//Calculation of center of mass (centroid)

	float accumX, accumY, accumZ, count;
	accumX = accumY = accumZ = count = 0.0;
	//imshow("handmat", handMat);
	//waitKey(0);
#ifdef use_wy_calc_centroid
	//average of 3D in world space
	for (int row = 0; row < handMat.rows; ++row) 
	{
		for (int col = 0; col < handMat.cols; ++col) 
		{
			ushort pixel = handMat.at<ushort>(row, col);
			if (pixel < 1) 
			{
				continue;
			}
			float x3D = static_cast<float>(x + col);
			float y3D = static_cast<float>(y + row);
			float z = static_cast<float>(pixel);
			reproject2DTo3D(x3D, y3D, z);
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
	avgX = accumX / count;
	avgY = accumY / count;
	avgZ = accumZ / count;

	Project3DTo2D(avgX, avgY, avgZ);
	avgU = avgX;
	avgV = avgY;
#else
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
			float x3D = static_cast<float>(x + col);
			float y3D = static_cast<float>(y + row);
			//directly accumulate depth
			float z = static_cast<float>(pixel);

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
	if (opt == 1)
	{
		reproject2DTo3D(avgX, avgY, avgZ);
	}
	else
	{
		avgZ = handMat.at<ushort>(int(handMat.rows / 2), int(handMat.cols / 2));
		if (avgZ == 0) //BFS to find nearest non-zero pixel
		{
			//avgZ or avgZ_real will be reset to 0.0 again so WE NEED TO SET IT TO 500 IF WE DO NOT FOUND ANY NEARBY NON-ZERO PIXEL
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
			}
			if (!found) avgZ = 500.0;
		}
	}
	//convert average_u average_v average_d back to world space
	//reproject2DTo3D(avgX, avgY, avgZ);
	//cout << "avgX" << " " << avgX << " avgY " << avgY << " avgZ " << avgZ << "\n";
	//avgZ = handMat.at<ushort>(int(avgY - y), int(avgX - x));
	//cout << "avgX" << " " << avgX << " avgY " << avgY << " avgZ " << avgZ << "\n";
	//cout << "====================\n\n";

	//Use originally provided center_x, center_y from bounding box annotation to do depth thresholding
	
	//cout << "Center point depth is " << avgZ << "===================================\n\n\n";
#endif
	return true;
}



ushort drawHistogram(const Mat &handMat, int opt) 
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
	/*
	#ifdef DEBUG
	if (opt == after)
	{
	imshow("histogram", graph);
	#ifdef wait_key
	cv::waitKey(0);
	#else
	cv::waitKey(1);
	#endif
	}
	#endif
	*/
	//imshow("histogram", graph);
	//waitKey(0);
	float sum = 0.0;
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
	/*cout << "localmax = " << local_max_num << ", max[0] = " << localmax[0] << endl;
	cout << "max = " << maxPixel << ",min = " << minPixel << ",high = " << ind + minPixel << endl;*/
	// Accum sum
	float accum = 0.0;
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

bool GetRidOfBackground(vector<float> vect, string imgPath, string imgName, Mat &handMat, Rect &extendRect, int opt) 
{
	//------ vect : (bbx_x1, bbx_y1, bbx_x2, bbx_y2)
	//------ visualize cropped depth image in 1.2x expanded bounding box
	//always reload from disk
	Mat img = imread(imgPath, CV_LOAD_IMAGE_ANYDEPTH);
	Mat colorImg = convertDepthToRGB(img);
	circle(colorImg, Point((int)vect[0], (int)vect[1]), 3, Scalar(255, 0, 0), 2);
	circle(colorImg, Point((int)vect[2], (int)vect[1]), 3, Scalar(0, 255, 0), 2);
	circle(colorImg, Point((int)vect[0], (int)vect[3]), 3, Scalar(255, 0, 255), 2);
	circle(colorImg, Point((int)vect[2], (int)vect[3]), 3, Scalar(255, 255, 255), 2);
	Rect rect((int)vect[0], (int)vect[1], (int)(vect[2] - vect[0]), (int)(vect[3] - vect[1]));
	rectangle(colorImg, rect, Scalar(255, 255, 255));
	//printf("%d %d %d %d\n", (int)vect[0], (int)vect[1], (int)vect[2], (int)vect[3]);

	//------ Deal with truncation. Modify the extended rectangular to fit in the raw depth image
	int x = vect[0];
	int y = vect[1];
	int w = vect[2] - vect[0];
	int h = vect[3] - vect[1];
	int xCenter = x + w / 2;
	int yCenter = y + h / 2;
	int newEdge = w > h ? w : h;
	//printf("x center %d y center %d\n", xCenter, yCenter);
	//printf("newedge %d\n", newEdge);
	w = h = newEdge;
	x = xCenter - w / 2;
	y = yCenter - h / 2;
	vect[0] = x;
	vect[1] = y;
	vect[2] = w;
	vect[3] = h;
	//printf("%d %d %d %d\n", (int)vect[0], (int)vect[1], (int)vect[2], (int)vect[3]);

	extendRect = Rect((int)vect[0], (int)vect[1], (int)(vect[2]), (int)(vect[3]));

	/*extendRect.x = extendRect.x < 0 ? 0 : extendRect.x;
	extendRect.y = extendRect.y < 0 ? 0 : extendRect.y;
	if (extendRect.x + extendRect.width >= colorImg.cols) 
	{
		extendRect.x = colorImg.cols - 1 - extendRect.width;
	}
	if (extendRect.y + extendRect.height >= colorImg.rows) 
	{
		extendRect.y = colorImg.rows - 1 - extendRect.height;
	}*/

	rectangle(colorImg, extendRect, Scalar(255, 0, 255));

	
	circle(colorImg, Point(mmcp_x, mmcp_y), 7, Scalar(0, 255, 0), 2);
	//printf("extend rect x %d width %d y %d height %d\n", extendRect.x, extendRect.width, extendRect.y, extendRect.height);
	circle(colorImg,Point(extendRect.x + extendRect.width / 2.0, extendRect.y + extendRect.height / 2.0), 3, Scalar(255, 255, 255), -1);
	
	//show color img
	//if (opt == after)
	if (index - 1 >= start_id)
	{
		//imshow("color", colorImg);
		//waitKey(0);

	}
	
#ifdef DEBUG
	if (opt == after) 
	{
		imshow("1", colorImg);
#ifdef wait_key
		cv::waitKey(0);
#else
		cv::waitKey(1);
#endif
	}
#endif


	/*if (extendRect.x < 0 || extendRect.y < 0
		|| extendRect.x >= 640 || extendRect.y >= 480
		|| extendRect.width < 0 || extendRect.height < 0
		|| extendRect.x + extendRect.width >= 640 || extendRect.y + extendRect.height >= 480) 
	{
		cout << "Image " << imgName << " bbox error" << endl;
		return false;
	}*/

	//Get RGB of cropped rectangle (image read from disk)
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
	//img(extendRect).copyTo(handMat);

	if (opt == after)
	{
		for (int row = 0; row < handMat.rows; row++)
		{
			for (int col = 0; col < handMat.cols; col++)
			{
				int cur_d = handMat.at<ushort>(row, col);
				if (cur_d < avgZ - 100 || cur_d > avgZ + 100)
				{
					handMat.at<ushort>(row, col) = 0;
				}
			}
		}
	}



	Mat colorHandMat = convertDepthToRGB(handMat);
	//show color mat
	if (opt == after)
	{
		//imshow("colormat", colorHandMat);
		//waitKey(0);
	}
	/*
	#ifdef DEBUG
	if (opt == after)
	{
	imshow("hand", colorHandMat);
	#ifdef wait_key
	cv::waitKey(0);
	#else
	cv::waitKey(1);
	#endif
	}
	#endif
	*/

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
	float range = maxPixel - minPixel;
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
			float diff = pixel - minPixel;
			float val = 255.0 * diff / range;
			onlyHand.at<uchar>(row, col) = static_cast<uchar>(val);
		}
	}

	
	/*
	#ifdef DEBUG
	if (opt == after)
	{
	imshow("Gray", onlyHand);
	#ifdef wait_key
	cv::waitKey(0);
	#else
	cv::waitKey(1);
	#endif
	}
	#endif
	*/
	Mat mask;

	double bar = cv::threshold(onlyHand, mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//cout << "Bar = " << bar << endl;

	/*
	#ifdef DEBUG
	if (opt == after)
	{
	imshow("mask", mask);
	#ifdef wait_key
	cv::waitKey(0);
	#else
	cv::waitKey(1);
	#endif
	}
	#endif
	*/
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
	/*
	#ifdef DEBUG
	if (opt == after)
	{
	imshow("maskHand", rgbHandMatCopy);
	#ifdef wait_key
	cv::waitKey(0);
	#else
	cv::waitKey(1);
	#endif
	}
	#endif
	*/

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
	if (opt == after)
	{
		//imshow("getRidBackground", getRidBackground);
		//waitKey(0);
	}
#ifdef DEBUG
	if (opt == after) 
	{
		imshow("getRidBackground", getRidBackground);
#ifdef wait_key
		cv::waitKey(0);
#else
		cv::waitKey(1);
#endif
	}
#endif
	
	return true;
}


Mat handMat;
void Process()
{
	srand(time(0));
#ifdef run_on_train
	ifstream readFile(rough_bbx_train_file);
#else
	ifstream readFile(rough_bbx_test_file);
#endif

	//read original annotation
#ifdef run_on_train
	FILE *fin_train_annotation = fopen(train_annotation_file, "r");

	char s[maxlen];

	//Note that the joint number is 1-based
	for (int i = 0; i < train_num; i++) 
	{
		if (i % 100 == 0) cout << "Reading " << i << "\n";
		fscanf(fin_train_annotation, "%s", s);
		//cout << s << "\n";
		for (int j = 0; j < JointNum_challenge; j++) 
		{
			for (int k = 0; k < 3; k++) 
			{
				fscanf(fin_train_annotation, "%f", &gt_joint_3d_global[i][j * 3 + k]);
				//cout << gt_joint_3d_global[i][j * 3 + k] << " ";
			}
		}
		//cout << "\n";
	}
	cout << "Reading original 3D ground truth annotation on training set Done!!!\n";
#endif


	string line;
	string suffix = ".png";
	
	while (getline(readFile, line))
	{
		if (line.empty())
		{
			break;
		}
		index++;
		cur_round++;
		int rindex = line.find(suffix) + suffix.length();
		string imgname = line.substr(0, rindex);
		//------parse image name to index
		int imgid = -1;
		//e.g. image_D00000001.png
		if (imgname.length() == 19) 
		{
			imgid = 0;
			for (int j = 7; j < 15; j++) imgid = imgid * 10 + imgname[j] - '0';
		}
		else continue; //invalid IND

		vector<float> vect;

		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t') 
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t') 
		{
			++rindex;
		}
		//------ First float bbx_x1
		vect.push_back(stof(line.substr(0, rindex)));
		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t') 
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t') 
		{
			++rindex;
		}
		//------ Second float bbx_y1
		vect.push_back(stof(line.substr(0, rindex)));
		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t') 
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t')
		{
			++rindex;
		}
		//------ Third float width
		vect.push_back(stof(line.substr(0, rindex)));
		line = line.substr(rindex);
		while (line.front() == ' ' || line.front() == '\t')
		{
			line = line.substr(1);
		}
		rindex = 0;
		while (line[rindex] != ' ' && line[rindex] != '\t' && line[rindex] != '\0')
		{
			++rindex;
		}
		//------ Fourth float height
		vect.push_back(stof(line.substr(0, rindex)));
		char buffer[50];
		sprintf(buffer, "%08d", index);
#ifdef run_on_train
		string imgpath = "../Task 1/training_images/" + imgname;
#else
		string imgpath = "../Task 1/test_images/" + imgname;
#endif

		if ((index - 1) % 100 == 0) 
		{
			cout << "processed " << index << " imgs" << endl;
		}

		if (index - 1 < start_id)
		{
			continue;
		}

		if (index - 1 > end_id)
		{
			break;
		}

#ifdef save_bbx
		//save original bbx provided by the author
		char save_bbx_file[maxlen];
		sprintf(save_bbx_file, "%s%d%s", save_bbx_prefix, imgid, ".txt");
		FILE *fout_bbx = fopen(save_bbx_file, "w");
		fprintf(fout_bbx, "%12.6f %12.6f %12.6f %12.6f", (float)((int)vect[0]), (float)((int)vect[1]), (float)((int)vect[2]), (float)((int)vect[3]));
		fclose(fout_bbx);
#endif

#ifdef run_on_train
		//get Middle MCP
		mmcp_x = gt_joint_3d_global[index - 1][mmcp * 3];
		mmcp_y = gt_joint_3d_global[index - 1][mmcp * 3 + 1];
		mmcp_z = gt_joint_3d_global[index - 1][mmcp * 3 + 2];
		//printf("%12.6f %12.6f\n", mmcp_x, mmcp_y);
		Project3DTo2D(mmcp_x, mmcp_y, gt_joint_3d_global[index - 1][mmcp * 3 + 2]);
		//printf("%12.6f %12.6f %12.6f %12.6f\n", vect[0], vect[1], vect[2], vect[3]);
		//printf("MMCP %12.5f %12.6f\n", mmcp_x, mmcp_y);
#ifdef save_mmcp
		//save middle MCP x,y
		char save_mmcp_file[maxlen];
		sprintf(save_mmcp_file, "%s%d%s", save_mmcp_prefix, imgid, ".txt");
		FILE *fout_mmcp = fopen(save_mmcp_file, "w");
		fprintf(fout_mmcp, "%12.6f %12.6f\n", mmcp_x, mmcp_y);
		fclose(fout_mmcp);

		sprintf(save_mmcp_file, "%s%d%s", save_mmcp_3d_prefix, imgid, ".txt");
		fout_mmcp = fopen(save_mmcp_file, "w");
		fprintf(fout_mmcp, "%12.6f %12.6f %12.6f\n", gt_joint_3d_global[index - 1][mmcp * 3], gt_joint_3d_global[index - 1][mmcp * 3 + 1], gt_joint_3d_global[index - 1][mmcp * 3 + 2]);
		fclose(fout_mmcp);

#endif

#ifdef save_jt_annot
		char save_jt_annot_file[maxlen];
		sprintf(save_jt_annot_file, "%s%d%s", save_jt_annot_prefix, imgid, ".txt");
		FILE *fout_jt_annot = fopen(save_jt_annot_file, "w");
		for (int j = 0; j < JointNum_challenge; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				fprintf(fout_jt_annot, "%12.6f ", gt_joint_3d_global[index - 1][j * 3 + k]);
			}
			fprintf(fout_jt_annot, "\n");
		}
		fclose(fout_jt_annot);
#endif
#endif

#ifdef crop_depth
		//Get initial extend rectangular

		//Crop from true synthetic image 

		//W/ real suffix: corresponding real
		//W/O any suffix: synthetic 
		string imgpath_real = "../Task 1/training_images/" + imgname;
		imgpath = "../Task 1/training_images_synth/" + imgname;

		Rect extendrect, extendrect_real;

		//Rough extend rect make it square 
		bool success_real = GetRidOfBackground(vect, imgpath_real, imgname, handMat, extendrect_real, before);

		if (!success_real) continue;


		//------ Recalculate centroid given subtracted image (TAKE CENTER POINT)
		float avgX_real = 0.0, avgY_real = 0.0, avgZ_real = 0.0, avgU_real = 0.0, avgV_real = 0.0;
		avgX_real = avgY_real = avgZ_real = 0.0;
		avgZ_real = 500.0;
		cur_round++;
		bool flag_real = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX_real, avgY_real, avgZ_real, avgU_real, avgV_real, 0);
		if (flag_real == false) 
		{
			cout << "Image " << imgname << " calc box center error" << endl;
			//continue;
		}


		//------Get the cropped image again
		//define new bbx_x1 bbx_y1 width height
		//new scale is the reciprocal of centroid depth

		float scale_real;
#ifdef use_mean_scale
		scale_real = mean_scale;
#else
		scale_real = mean_depth / avgZ_real * 210.0;//mean_scale;
#endif

		//imshow("handmat", handMat);
		//waitKey(0);
		//now is bbx_x1 bbx_y1 bbx_x2 bbx_y2 not bbx_x1 bbx_y1 width height
		
		


		//explicitly reset
		avgU_real = extendrect_real.x + extendrect_real.width / 2.0;
		avgV_real = extendrect_real.y + extendrect_real.height / 2.0;
		vect[0] = avgU_real - scale_real / 2.0;
		vect[1] = avgV_real - scale_real / 2.0;
		vect[2] = avgU_real + scale_real / 2.0;
		vect[3] = avgV_real + scale_real / 2.0;

		//1. DO REAL
		imgpath_real = "../Task 1/training_images/" + imgname;
		avgZ = avgZ_real; //forcibly set avgZ so that ouotlier can be removed (global variable)
		success_real = GetRidOfBackground(vect, imgpath_real, imgname, handMat, extendrect_real, after);
		if (!success_real) continue;

		//RECALCULATE THE CENTROID OF REAL
		cur_round++;
		flag_real = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX_real, avgY_real, avgZ_real, avgU_real, avgV_real, 1);
		if (flag_real == false)
		{
			cout << "Image " << imgname << " calc box center error" << endl;
			//continue;
		}
		Mat handMat_real = handMat.clone();
		Mat norm_depth_real = Mat::zeros(Size(handMat.rows, handMat.cols), CV_8UC1);
		for (int row = 0; row < handMat.rows; row++)
		{
			for (int col = 0; col < handMat.cols; col++)
			{
				int cur_d = handMat.at<ushort>(row, col);
				if (cur_d != 0)
				{
					int cur_col = (-(double(cur_d - avgZ_real) / double(100.0)) + 1.0) / 2.0 * 255;
					norm_depth_real.at<uchar>(row, col) = cur_col;
				}
			}
		}

		resize(norm_depth_real, norm_depth_real, Size(crop_size_, crop_size_));



		//2. DO SYNTH
		//imgpath = "../Task 1/training_images_synth/" + imgname;
		//READ SYNTH IMAGE
		//bool success = GetRidOfBackground(vect, imgpath, imgname, handMat, extendrect_real, before);
		//if (!success) continue;

		//WE'LL JUST USE THE CENTER AVGZ
		//cur_round++;
		//bool flag = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX, avgY, avgZ, avgU, avgV, 0);
		//if (flag == false)
		//{
		//	cout << "Image " << imgname << " calc box center error" << endl;
			//continue;
		//}

		//USE CENTER AVGZ TO REMOVE OUTLIER
		//success = GetRidOfBackground(vect, imgpath, imgname, handMat, extendrect_real, after);
		//if (!success) continue;

		//RECALCULATE TRUE CENTROID OF SYNTH
		//cur_round++;
		//flag = calcBoxCenter(handMat, extendrect_real.x, extendrect_real.y, avgX, avgY, avgZ, avgU, avgV, 1);

		//Mat norm_depth = Mat::zeros(Size(handMat.rows, handMat.cols), CV_8UC1);
		//for (int row = 0; row < handMat.rows; row++)
		//{
		//	for (int col = 0; col < handMat.cols; col++)
		//	{
		//		int cur_d = handMat.at<ushort>(row, col);
		//		if (cur_d != 0)
		//		{
		//			if (cur_d >= 250 && cur_d <= 1500) //already reading a 640x480 cropped hand depth image w/ raw depth values
		//			{
		//				int cur_col = (-(double(cur_d - avgZ) / double(100.0)) + 1.0) / 2.0 * 255;
		//				norm_depth.at<uchar>(row, col) = cur_col;
		//			}
		//		}
		//	}
		//}
		
		//resize(norm_depth, norm_depth, Size(crop_size_, crop_size_));
		if (index - 1 >= start_id)
		{
			/*imshow("norm_depth_real", norm_depth_real);
			waitKey(0);
			imshow("norm_depth_synth", norm_depth);
			waitKey(0);

			imshow("handMat_real", handMat_real);
			waitKey(0);
			imshow("handMat_synth", handMat);
			waitKey(0);*/

			//printf("Real  avgX %12.6f avgY %12.6f avgZ %12.6F\n", avgX_real, avgY_real, avgZ_real);
			//printf("Synth avgX %12.6f avgY %12.6f avgZ %12.6F\n", avgX, avgY, avgZ);

			//imshow("handmat", handMat);
			//waitKey(0);
#ifdef save_crop
			//char save_crop_name[maxlen];
			//sprintf(save_crop_name, "%s%d%s", save_crop_prefix, index - 1 + train_num, ".png");
			//imwrite(save_crop_name, handMat);

			//REAL DEPTH
			char save_crop_real_name[maxlen];
			sprintf(save_crop_real_name, "%s%d%s", save_crop_prefix, index - 1, ".png");
			imwrite(save_crop_real_name, handMat_real);

			//sprintf(save_crop_name, "%s%d%s", "../Task 1/training_images_crop_synth/", index - 1, ".png");
			//imwrite(save_crop_name, handMat);
#endif
			//imshow("norm_depth", norm_depth);
			//waitKey(0);

		}
		/*cur_round++;
		//get real COM avgU avgV
		flag = calcBoxCenter(handMat, extendrect.x, extendrect.y, avgX, avgY, avgZ, avgU, avgV, 1);
		Mat show_com = convertDepthToRGB(handMat);
		circle(show_com, Point(avgU - extendrect.x, avgV - extendrect.y), 3, Scalar(0, 255, 255), -2);
		//imshow("showcom", show_com);
		
#ifdef run_on_train
		//COM -> MMCP
		mmcp_x = gt_joint_3d_global[index - 1][mmcp * 3];
		mmcp_y = gt_joint_3d_global[index - 1][mmcp * 3 + 1];
		mmcp_z = gt_joint_3d_global[index - 1][mmcp * 3 + 2];
		//printf("COM %12.6f %12.6f %12.6f -> MMCP %12.6f %12.6f %12.6f\n", avgX, avgY, avgZ, mmcp_x, mmcp_y, mmcp_z);
#endif*/
#endif
	}
}
int main()
{
	Process();
	return 0;
}


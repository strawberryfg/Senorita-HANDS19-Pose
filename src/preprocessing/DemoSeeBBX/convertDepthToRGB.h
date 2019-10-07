#pragma once
#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

// Depth unit of depth is 0.1mm, here is 10m
const int kMaxDistance = 100000;

cv::Mat convertDepthToRGB(const cv::Mat &depthImg) 
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
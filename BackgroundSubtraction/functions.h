#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

void runningAverage(cv::Mat &, cv::Mat, double);
std::vector<cv::Point> getMaximumContour(std::vector<std::vector<cv::Point>>, std::vector<cv::Vec4i>);
void segmentImage(cv::Mat, cv::Mat, cv::Mat &, std::vector<cv::Point> &);

#endif
#ifndef ENTROPY_H
#define ENTROPY_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

float entropy(cv::Mat, cv::Size, int);

// myEntropy calculates relative occurrence of different symbols within given input sequence using histogram
// cv::Mat myEntropy(cv::Mat, int);
#ifndef KEYFRAMESANALYSIS_H
#define KEYFRAMESANALYSIS_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <fstream>

bool getFilesList(const std::string &, std::vector<std::string> *, const bool);
float calcEntropy(cv::Mat, cv::Size, int);
cv::Mat myEntropy(cv::Mat, int);
std::vector<unsigned long> getPeaksPositions(std::vector<float> *);

#endif
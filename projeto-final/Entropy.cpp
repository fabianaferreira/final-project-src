#include "Entropy.h"

using namespace std;
using namespace cv;

float entropy(Mat seq, Size size, int index)
{
	int cnt = 0;
	float entr = 0;
	float total_size = size.height * size.width; //total size of all symbols in an image

	for (int i = 0; i < index; i++)
	{
		float sym_occur = seq.at<float>(0, i); //the number of times a sybmol has occured
		if (sym_occur > 0) //log of zero goes to infinity
		{
			cnt++;
			entr += (sym_occur / total_size)*(log2(total_size / sym_occur));
		}
	}
	return entr;
}

Mat myEntropy(Mat seq, int histSize)
{

	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	/// Compute the histograms:
	calcHist(&seq, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	return hist;
}
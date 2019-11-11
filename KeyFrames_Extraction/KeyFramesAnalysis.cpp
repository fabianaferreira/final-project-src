#include "KeyFramesAnalysis.h"

using namespace std;
using namespace cv;

bool getFilesList(const string &path, vector<string> *files, const bool showHiddenDirs = false)
{
    DIR *dpdf;
    struct dirent *epdf;
    dpdf = opendir(path.c_str());
    if (dpdf != NULL)
    {
        while ((epdf = readdir(dpdf)) != NULL)
        {
            if (showHiddenDirs ? (epdf->d_type == DT_DIR && string(epdf->d_name) != ".." && string(epdf->d_name) != ".") : (epdf->d_type == DT_DIR && strstr(epdf->d_name, "..") == NULL && strstr(epdf->d_name, ".") == NULL))
            {
                getFilesList(path + "/" + epdf->d_name, files, showHiddenDirs);
            }
            if (epdf->d_type == DT_REG)
            {
                // string extension = getFileExtension(string(epdf->d_name));
                // if (extension.compare(EXTENSION) == 0)
                files->push_back(path + "/" + epdf->d_name);
            }
        }
        closedir(dpdf);
        return true;
    }
    return false;
}

float calcEntropy(Mat seq, Size size, int index)
{
    int cnt = 0;
    float entr = 0;
    float total_size = size.height * size.width; //total size of all symbols in an image

    for (int i = 0; i < index; i++)
    {
        float sym_occur = seq.at<float>(0, i); //the number of times a sybmol has occured
        if (sym_occur > 0)                     //log of zero goes to infinity
        {
            cnt++;
            entr += (sym_occur / total_size) * (log2(total_size / sym_occur));
        }
    }
    return entr;
}

Mat myEntropy(Mat seq, int histSize)
{

    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    Mat hist;

    /// Compute the histograms
    calcHist(&seq, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    return hist;
}

vector<unsigned long> getPeaksPositions(vector<float> *entropy)
{
    vector<unsigned long> peaks;
    for (int i = 1; i < entropy->size() - 1; ++i)
    {
        float left = entropy->at(i - 1);
        float cent = entropy->at(i);
        float right = entropy->at(i + 1);

        if ((left < cent && right <= cent) || (left > cent && right >= cent))
        {
            //Pmin || Pmax
            peaks.push_back(i);
            peaks.push_back(entropy->at(i));
        }
    }

    return peaks;
}
#include "functions.h"

using namespace std;

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
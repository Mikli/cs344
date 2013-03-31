//#include <cvaux.h>
//#include <highgui.h>
//#include <cxcore.hpp>
//#include <cv.hpp>
//#include <highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include "cuda_runtime.h"

//The caller becomes responsible for the returned pointer. This
//is done in the interest of keeping this code as simple as possible.
//In production code this is a bad idea - we should use RAII
//to ensure the memory is freed.  DO NOT COPY THIS AND USE IN PRODUCTION
//CODE!!!
using namespace cv;
using namespace std;
void loadImageHDR(const std::string &filename,
                  float **imagePtr,
                  size_t *numRows, size_t *numCols)
{
  cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);

  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  if (image.channels() != 3) {
    std::cerr << "Image must be color!" << std::endl;
    exit(1);
  }

  if (!image.isContinuous()) {
    std::cerr << "Image isn't continuous!" << std::endl;
    exit(1);
  }

  int type = image.type();
  *imagePtr = new float[image.rows * image.cols * image.channels()];

  float *cvPtr = image.ptr<float>(0);
  for (size_t i = 0; i < image.rows * image.cols * image.channels(); ++i)
    (*imagePtr)[i] = cvPtr[i];

  *numRows = image.rows;
  *numCols = image.cols;
}

void loadImageRGBA(const std::string &filename,
                   uchar4 **imagePtr,
                   size_t *numRows, size_t *numCols)
{
  cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  if (image.channels() != 3) {
    std::cerr << "Image must be color!" << std::endl;
    exit(1);
  }

  if (!image.isContinuous()) {
    std::cerr << "Image isn't continuous!" << std::endl;
    exit(1);
  }

  cv::Mat imageRGBA;
  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  *imagePtr = new uchar4[image.rows * image.cols];

  unsigned char *cvPtr = imageRGBA.ptr<unsigned char>(0);
  for (size_t i = 0; i < image.rows * image.cols; ++i) {
    (*imagePtr)[i].x = cvPtr[4 * i + 0];
    (*imagePtr)[i].y = cvPtr[4 * i + 1];
    (*imagePtr)[i].z = cvPtr[4 * i + 2];
    (*imagePtr)[i].w = cvPtr[4 * i + 3];
  }

  *numRows = image.rows;
  *numCols = image.cols;
}

void saveImageRGBA(const uchar4* const image,
                   const size_t numRows, const size_t numCols,
                   const std::string &output_file)
{
  int sizes[2];
  sizes[0] = numRows;
  sizes[1] = numCols;
  size_t steps = 0;
  cv::Mat imageRGBA(numRows, numCols, CV_8UC4, (void *)image, steps);
  //cv::Mat imageRGBA(2, sizes, CV_8UC4, (void *)image,);
  cv::Mat imageOutputBGR;
  cv::cvtColor(imageRGBA, imageOutputBGR, CV_RGBA2BGR);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);
}

//output an exr file
//assumed to already be BGR
void saveImageHDR(const float* const image,
                  const size_t numRows, const size_t numCols,
                  const std::string &output_file)
{
  int sizes[2];
  sizes[0] = numRows;
  sizes[1] = numCols;
	size_t steps = 0;
  //cv::Mat imageHDR(2, sizes, CV_32FC3, (void *)image);
	cv::Mat imageHDR(numRows, numCols, CV_32FC3, (void *)image, steps);	

  imageHDR = imageHDR * 255;

  cv::imwrite(output_file.c_str(), imageHDR);
}

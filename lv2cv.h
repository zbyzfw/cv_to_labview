//
// Created by dahe on 2021/9/24.
//

#pragma once
#include "opencv2/opencv.hpp"
#ifdef DLL_IMPLEMENT
#define DLL_API _declspec(dllimport)
#else
#define DLL_API _declspec(dllexport)
#endif
using namespace cv;

extern "C" DLL_API void add2(int rows, int cols, unsigned __int8 data);
extern "C" DLL_API int ImgdatatoLabview(unsigned __int8 imgdata);
extern "C" DLL_API int getimagesize(int rows, int cols);

//void add2(int rows, int cols, unsigned __int8 *data){}

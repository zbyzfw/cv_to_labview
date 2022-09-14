//
// Created by dahe on 2021/9/24.
//

//#include "lv2cv.h"

#include <opencv2\opencv.hpp>

using namespace cv;

#define WINDOW_NAME1 "[原始图片]"

#define WINDOW_NAME2 "[效果窗口]"



//Mat g_srcImage; Mat g_templateImage; Mat g_resultImage;
//
//int g_nMatchMethod;
//
//int g_nMaxTrackbarnum = 5;
//
//void on_Matching(int, void*);

extern "C" __declspec(dllexport) void add2(int rows, int cols, unsigned __int8 *data)

{
   Mat image_src(rows,cols, CV_8U, &data[0]);

   /* Insert code here */
   Mat temp;
   boxFilter(image_src, temp, -1, Size(5, 5));
   Canny(temp, image_src, 150, 100, 3);

};
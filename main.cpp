#include <iostream>
#include <numeric>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.1415926
//弧度转角度
#define r2a(x) ((x)*180/PI)
//角度转弧度
#define a2r(x) ((x)*PI/180)

using namespace cv;
using namespace std;

double getSeconds(chrono::time_point<chrono::system_clock> &start,
                  chrono::time_point<chrono::system_clock> &end) {
  auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
  return double(duration.count()) / 1000000;
}

bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
  //Number of key points
  int N = key_point.size();
  //构造矩阵X
  cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
  for (int i = 0; i < n + 1; i++)
  {
    for (int j = 0; j < n + 1; j++)
    {
      for (int k = 0; k < N; k++)
      {
        X.at<double>(i, j) = X.at<double>(i, j) +
            std::pow(key_point[k].x, i + j);
      }
    }
  }
  //构造矩阵Y
  cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
  for (int i = 0; i < n + 1; i++)
  {
    for (int k = 0; k < N; k++)
    {
      Y.at<double>(i, 0) = Y.at<double>(i, 0) +
          std::pow(key_point[k].x, i) * key_point[k].y;
    }
  }
  A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
  //求解矩阵A
  cv::solve(X, Y, A, cv::DECOMP_LU);
  return true;
}

vector<int> findPeaks(vector<double> num,int count)
{
  vector<int> sign;
//  mypoints sign2;
  for(int i = 1;i<count;i++)
  {
    /*相邻值做差：
     *小于0，赋-1
     *大于0，赋1
     *等于0，赋0
     */
    double diff = num[i] - num[i-1];
    if(diff>0)
    {
      sign.push_back(1);
    }
    else if(diff<0)
    {
      sign.push_back(-1);
    }
    else
    {
      sign.push_back(0);
    }
  }
  //再对sign相邻位做差
  //保存极大值和极小值的位置
  vector<int> indMax;
  vector<int> indMin;

  for(int j = 1;j<sign.size();j++)
  {
    int diff = sign[j]-sign[j-1];
    if(diff<0)
    {
      indMax.push_back(j);
    }
    else if(diff>0)
    {
      indMin.push_back(j);
    }
  }
//  cout<<"max:"<<endl;
//  for(int m = 0;m<indMax.size();m++)
//  {
//    cout<<num[indMax[m]]<<"  ";
//  }
//  cout<<endl;
//  cout<<"min:"<<endl;
//  for(int n = 0;n<indMin.size();n++)
//  {
//    cout<<num[indMin[n]]<<"  ";
//  }
  return indMax;
}

//int main()
//
//{
//
//  int a[] = {1,2,10,2,4,1,8,10,23,0};
//
//  findPeaks(a,10);
//
//  return 0;
//
//}


int main() {
    std::cout << "start" << std::endl;
    auto start = chrono::system_clock::now(); // 开始时间
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point>> best_contours;
    vector<int> center_y(2);
    float rotated;
//    std::string image_path = "E:/cvproject2/cv_start/lay/img1 (8).bmp";
    std::string image_path = "E:/cvproject2/cv_start/09.png";
    //convexHull();
    Point org(10, 90);
    Mat img = imread(image_path,0);
    transpose(img,img);
    flip(img,img, -1);
    adaptiveThreshold(img,img,255,ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,21,-2);
    findContours(img,contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE,Point(0, 0));
    // if (contours.size() == 0) {return;}
    Mat drawImg = Mat::zeros(img.size(), CV_8UC3);
//    cout << img.size[0] << endl;
//    cout << center_y[0] << endl;
    drawImg.setTo(cv::Scalar(0, 0, 0));
    //腐蚀操作
//    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
//    Mat dstImage;
//    erode(srcImage, dstImage, element);
    for (int i = 0; i < contours.size()-1; i++) {
      RotatedRect rect = minAreaRect(contours[i]);
      if((rect.size.width>1000 || rect.size.height>1000) && (rect.size.width<60 || rect.size.height<60)){
        if(!center_y[0]){
          best_contours.push_back(contours[i]);
          center_y[0] = rect.center.y;
          if(rect.size.width<rect.size.height){
            rotated = 90-rect.angle;
          }else{
            rotated = rect.angle;
          }

        }else if(!center_y[1]){
          if(abs(rect.center.y-center_y[0])>150){
            best_contours.push_back(contours[i]);
            center_y[1] = rect.center.y;
          }
        }
      }
    // Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
    };
    cout<<"contours size: "<<best_contours.size()<<" -- center_y size: "<<center_y.size()<<endl;
//    for(int i = 0; i < contours.size()-1; i++){
//      if(!center_y[0]){
//        cout << center_y.size() << endl;
//        cout <<"center:"<< minAreaRect(contours[i]).center.y << endl;
//      }else if(!center_y[1]){
//        if(abs(minAreaRect(contours[i]).center.y-center_y[0])>150){
//          best_contours.push_back(contours[i]);
//          center_y[1] = minAreaRect(contours[i]).center.y;
//        }
//      }
//    };

//    drawContours(drawImg, best_contours,-1, Scalar(255,0,0), 2, 8, hierarchy, 0, Point(0, 0));

//    cv::putText(img,"当前时间",org,FONT_HERSHEY_COMPLEX_SMALL, 3,
//                Scalar(1, 1, 255), 5, 8);
    // 曲线拟合
    //输入拟合点
    std::vector<cv::Point> points = best_contours[0];
    cv::Mat A;
    polynomial_curve_fit(points, 9, A);
//    std::cout << "A = " << A <<drawImg.size[1]<< std::endl;
    std::vector<cv::Point> points_fitted;
    std::vector<double> cycle_y;
    for (int x = 0; x < drawImg.size[1]; x++)
    {
      double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
          A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3) + A.at<double>(4, 0)*std::pow(x, 4) + A.at<double>(5, 0)*std::pow(x, 5)
          + A.at<double>(6, 0)*std::pow(x, 6) + A.at<double>(7, 0)*std::pow(x, 7) + A.at<double>(8, 0)*std::pow(x, 8) + A.at<double>(9, 0)*std::pow(x, 9);
//      double y_ = A.at<double>(1, 0) +
//      A.at<double>(2, 0)*x + A.at<double>(3, 0)*std::pow(x, 2) + A.at<double>(4, 0)*std::pow(x, 3) + A.at<double>(5, 0)*std::pow(x, 4)
//      + A.at<double>(6, 0)*std::pow(x, 5) + A.at<double>(7, 0)*std::pow(x, 6) + A.at<double>(8, 0)*std::pow(x, 7) + A.at<double>(9, 0)*std::pow(x, 8);
      points_fitted.push_back(cv::Point(x, y));
      cycle_y.push_back(y);
//      if(x)
//      cycle
    }
//    cv::polylines(drawImg, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
//    cv::imshow("image", image);
//    cv::polylines(drawImg, points_fitted, false, cv::Scalar(255, 255, 255), 1, 8, 0);

//    double temp[1024];
//    float *buffer = new float[sizeof(temp)];
//    if (!cycle_y.empty())
//    {
//      memcpy(buffer, &temp[0], cycle_y.size()*sizeof(double));
//    }
//    for(auto i:temp)cout<<i<<endl;
    vector<int> result;
    vector<int> ind = findPeaks(cycle_y,1024);
    if(ind.size() >3){
      for(int i=1;i<ind.size()-2;i++){
//        cout<<"index:"<<ind[i+1]-ind[i]<<endl;
        result.push_back(ind[i+1]-ind[i]);
      }
    }
    double sumValue = accumulate(std::begin(result), std::end(result), 0.0);
    double meanValue = sumValue / result.size();
    //    String value = "aa";
    string str = "lay:";
    string res = str + to_string(meanValue);
    putText(drawImg,res,org,FONT_HERSHEY_COMPLEX_SMALL, 3,
                            Scalar(1, 1, 255), 5, 8);
    auto end = chrono::system_clock::now();
    double time = getSeconds(start,end);
    cout <<"lay length: "<< meanValue/cos(a2r(rotated)) << endl;
    cout <<"angle: "<< rotated << endl;
    cout << "time= " << time << "s" << endl;
    cout << "diameter:"<< abs(center_y[1]-center_y[0])* cos(a2r(rotated)) << endl;
//    imshow("baizhi",drawImg);
//    waitKey(10000);
    return 0;
}

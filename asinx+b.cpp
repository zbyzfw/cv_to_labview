#pragma execution_character_set("utf-8")
//
// Created by dahe on 2023/5/8.
//
// 导入opencv头文件
#include <opencv2/opencv.hpp>
#include <iostream>
//
using namespace cv;
using namespace std;


// 定义三角函数的形式为y = a * sin(b * x + c) + d
// 将其转化为多项式形式为y = a * (sin(c) * cos(b * x) + cos(c) * sin(b * x)) + d
// 令p = a * sin(c), q = a * cos(c), r = d, 则y = p * cos(b * x) + q * sin(b * x) + r
// 令u = cos(b * x), v = sin(b * x), 则y = p * u + q * v + r
// 用最小二乘法求解p, q, r的值，再反推出a, b, c, d的值
// yt = a*sin(b*(xcos10-ysin10)+c)+d
// yt = a*sin(b*xcos10-b*ysin10+c)cos10+a*cos(b*xcos 10-b*ysin10+c)sin 10+d



void PrintMat(Mat A) {
  printf("printMat:\n");
  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < A.cols; j++)
      printf("%f ", A.at<double>(i, j));
    printf("\n");
  }
  printf("\n");
}

int main()
{
  // 生成一些离散的三角函数点
  double a = 2.0; // 振幅
  double b = 0.5; // 角频率
  double c = 0.3; // 相位
  double d = 1.0; // 偏移量
  double t = 0.1745; // 旋转角度
  int n = 20; // 点的个数
  cout<<a<<b<<c<<d<<n<<endl;
  vector<double> x(n); // 横坐标
  vector<double> y(n); // 纵坐标
  RNG rng(0); // 随机数生成器，用于添加噪声
  for (int i = 0; i < n; i++)
  {
    x[i] = i;
    y[i] = a * sin(b * x[i] + c) + d + rng.gaussian(0.1); // 添加高斯噪声
  }

  // 构造A矩阵和b向量，A的每一行为[cos(b * x), sin(b * x), 1]
  Mat A(n, 3, CV_64F);
  Mat _b(n, 1, CV_64F);

  for (int i = 0; i < n; i++)
  {
    A.at<double>(i, 0) = cos(b * x[i]);
    A.at<double>(i, 1) = sin(b * x[i]);
    A.at<double>(i, 2) = 1;
    _b.at<double>(i, 0) = y[i];
  }
  PrintMat(A);
  PrintMat(_b);
  // 求解超定方程Ax=b，得到x向量，即[p, q, r]
  Mat xvec;
  solve(A, b, xvec, DECOMP_SVD);

  // 输出拟合结果
  cout << "原始参数：" << endl;
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "c = " << c << endl;
  cout << "d = " << d << endl;
  cout << "拟合参数：" << endl;
  double p = xvec.at<double>(0, 0);
  double q = xvec.at<double>(1, 0);
  double r = xvec.at<double>(2, 0);
  cout << "p = " << p << endl;
  cout << "q = " << q << endl;
  cout << "r = " << r << endl;

  // 反推出a, b, c, d的值
  double a_fit = sqrt(p * p + q * q); // a的拟合值
  double b_fit = b; // b的拟合值，已知
  double c_fit = atan2(q, p); // c的拟合值，注意atan2的用法
  double d_fit = r; // d的拟合值
  cout << "a_fit = " << a_fit << endl;
  cout << "b_fit = " << b_fit << endl;
  cout << "c_fit = " << c_fit << endl;
  cout << "d_fit = " << d_fit << endl;

}
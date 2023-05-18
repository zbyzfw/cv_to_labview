#pragma execution_character_set("utf-8")
//
// Created by dahe on 2023/5/8.
//
// ����opencvͷ�ļ�
#include <opencv2/opencv.hpp>
#include <iostream>
//
using namespace cv;
using namespace std;


// �������Ǻ�������ʽΪy = a * sin(b * x + c) + d
// ����ת��Ϊ����ʽ��ʽΪy = a * (sin(c) * cos(b * x) + cos(c) * sin(b * x)) + d
// ��p = a * sin(c), q = a * cos(c), r = d, ��y = p * cos(b * x) + q * sin(b * x) + r
// ��u = cos(b * x), v = sin(b * x), ��y = p * u + q * v + r
// ����С���˷����p, q, r��ֵ���ٷ��Ƴ�a, b, c, d��ֵ
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
  // ����һЩ��ɢ�����Ǻ�����
  double a = 2.0; // ���
  double b = 0.5; // ��Ƶ��
  double c = 0.3; // ��λ
  double d = 1.0; // ƫ����
  double t = 0.1745; // ��ת�Ƕ�
  int n = 20; // ��ĸ���
  cout<<a<<b<<c<<d<<n<<endl;
  vector<double> x(n); // ������
  vector<double> y(n); // ������
  RNG rng(0); // ������������������������
  for (int i = 0; i < n; i++)
  {
    x[i] = i;
    y[i] = a * sin(b * x[i] + c) + d + rng.gaussian(0.1); // ��Ӹ�˹����
  }

  // ����A�����b������A��ÿһ��Ϊ[cos(b * x), sin(b * x), 1]
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
  // ��ⳬ������Ax=b���õ�x��������[p, q, r]
  Mat xvec;
  solve(A, b, xvec, DECOMP_SVD);

  // �����Ͻ��
  cout << "ԭʼ������" << endl;
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "c = " << c << endl;
  cout << "d = " << d << endl;
  cout << "��ϲ�����" << endl;
  double p = xvec.at<double>(0, 0);
  double q = xvec.at<double>(1, 0);
  double r = xvec.at<double>(2, 0);
  cout << "p = " << p << endl;
  cout << "q = " << q << endl;
  cout << "r = " << r << endl;

  // ���Ƴ�a, b, c, d��ֵ
  double a_fit = sqrt(p * p + q * q); // a�����ֵ
  double b_fit = b; // b�����ֵ����֪
  double c_fit = atan2(q, p); // c�����ֵ��ע��atan2���÷�
  double d_fit = r; // d�����ֵ
  cout << "a_fit = " << a_fit << endl;
  cout << "b_fit = " << b_fit << endl;
  cout << "c_fit = " << c_fit << endl;
  cout << "d_fit = " << d_fit << endl;

}
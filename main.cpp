#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;

int main() {
    std::cout << "Hello, OpenCV!" << std::endl;
    std::string image_path = "D:/039_003.png";
    //convexHull();
    Point org(10, 90);

    Mat img = imread(image_path,IMREAD_COLOR);
    cv::putText(img,"当前时间",org,FONT_HERSHEY_COMPLEX_SMALL, 3,
                Scalar(1, 1, 255), 5, 8);
    imshow("baizhi",img);
    waitKey(10000);
    return 0;
}

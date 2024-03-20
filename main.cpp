#include <iostream>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void grayscale(Mat& InputArray, Mat& OutputArray){
    OutputArray = Mat(InputArray.rows, InputArray.cols, CV_8UC3);
    for(int i = 0; i < InputArray.rows; i++){
        for(int j = 0; j < InputArray.cols; j++){
            double pxl = 0.299 * InputArray.at<Vec3b>(i, j)[2] + 0.587 * InputArray.at<Vec3b>(i, j)[1] + 0.114 * InputArray.at<Vec3b>(i, j)[0];
            OutputArray.at<Vec3b>(i, j)[0] = pxl;
            OutputArray.at<Vec3b>(i, j)[1] = pxl;
            OutputArray.at<Vec3b>(i, j)[2] = pxl;
        }
    }
}

void sepiascale(Mat& InputArray, Mat& OutputArray){
    OutputArray = Mat(InputArray.rows, InputArray.cols, CV_8UC3);
    for(int i = 0; i < InputArray.rows; i++){
        for(int j = 0; j < InputArray.cols; j++){
            double tr, tg, tb;
            tb = 0.272 * InputArray.at<Vec3b>(i, j)[2] + 0.534 * InputArray.at<Vec3b>(i, j)[1] + 0.131 * InputArray.at<Vec3b>(i, j)[0];
            tg = 0.349 * InputArray.at<Vec3b>(i, j)[2] + 0.686 * InputArray.at<Vec3b>(i, j)[1] + 0.168 * InputArray.at<Vec3b>(i, j)[0];
            tr = 0.393 * InputArray.at<Vec3b>(i, j)[2] + 0.769 * InputArray.at<Vec3b>(i, j)[1] + 0.189 * InputArray.at<Vec3b>(i, j)[0];
            if (tr > 255) tr = 255;
            if (tg > 255) tg = 255;
            if (tb > 255) tb = 255;
            OutputArray.at<Vec3b>(i, j)[0] = tb;
            OutputArray.at<Vec3b>(i, j)[1] = tg;
            OutputArray.at<Vec3b>(i, j)[2] = tr;
        }
    }
}

void negativescale(Mat& InputArray, Mat& OutputArray){
    OutputArray = Mat(InputArray.rows, InputArray.cols, CV_8UC3);
    for(int i = 0; i < InputArray.rows; i++){
        for(int j = 0; j < InputArray.cols; j++){
            OutputArray.at<Vec3b>(i, j)[0] = 255 - InputArray.at<Vec3b>(i, j)[0];
            OutputArray.at<Vec3b>(i, j)[1] = 255 - InputArray.at<Vec3b>(i, j)[1];
            OutputArray.at<Vec3b>(i, j)[2] = 255 - InputArray.at<Vec3b>(i, j)[2];
        }
    }
}

void contourscale(Mat& InputArray, Mat& OutputArray){
    Mat temp;
    GaussianBlur(InputArray, temp, Size(0, 0), 2);
    cvtColor(temp, temp, COLOR_BGR2GRAY);
    OutputArray = Mat(InputArray.rows, InputArray.cols, CV_8U);
    for(int i = 1; i < temp.rows - 1; i++){
        for(int j = 1; j < temp.cols - 1; j++){
            float gx = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i, j + 1) + temp.at<uchar>(i - 1, j + 1) - temp.at<uchar>(i + 1, j - 1) - 2 * temp.at<uchar>(i, j - 1) - temp.at<uchar>(i - 1, j - 1);
            float gy = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i + 1, j) + temp.at<uchar>(i + 1, j - 1) - temp.at<uchar>(i - 1, j - 1) - 2 * temp.at<uchar>(i - 1, j) - temp.at<uchar>(i - 1, j + 1);
            OutputArray.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
        }
    }
}

int main()
{
    Mat img = imread("../Parallel_Computing_Basics-Practice_7-8_semester/image.jpg", IMREAD_COLOR);
    if (img.empty()){
        cout << "Изображение не загружено" << endl;
        return -1;
    }
    namedWindow("Original", WINDOW_NORMAL);
    imshow("Original", img);
    Mat gray, sepia, negative, contour;

#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
        {
            grayscale(img, gray);
        }
#pragma omp section
        {
            sepiascale(img, sepia);
        }
#pragma omp section
        {
            negativescale(img, negative);
        }
#pragma omp section
        {
            contourscale(img, contour);
        }
    }
    namedWindow("Gray", WINDOW_NORMAL);
    imshow("Gray", gray);
    namedWindow("Sepia", WINDOW_NORMAL);
    imshow("Sepia", sepia);
    namedWindow("Negative", WINDOW_NORMAL);
    imshow("Negative", negative);
    namedWindow("Contour", WINDOW_NORMAL);
    imshow("Contour", contour);

    waitKey(0);
    return 0;
}

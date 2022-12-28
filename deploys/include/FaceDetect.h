//
// Created by 60180 on 2022/12/27.
//
#include <opencv2/opencv.hpp>
#include <iostream>

#ifndef TEST_FACEDETECT_H
#define TEST_FACEDETECT_H
struct Object{
    cv::Rect rect;
    float score;
};
struct net_config{
    float conf_threshold;
    std::string model_path;
    int input_height;
    int input_width;
};



class FaceDetect {
public:
    explicit FaceDetect(const net_config& config);

    void detect(cv::Mat&image, std::vector<Object>&result);
private:
    int input_width;
    int input_height;
    float conf_threshold;
    cv::dnn::Net net;
    void post_process(cv::Mat &heatmap,cv::Mat&heatmap_dilate,const cv::Mat& out) const;//成员变量不可以被修改
    void image_process(cv::Mat & image);
    const float mean[3] = {0.4914, 0.4822, 0.4465};
    const float stds[3] = {0.2023, 0.1994, 0.2010};
};


#endif //TEST_FACEDETECT_H

//
// Created by 60180 on 2022/12/27.
//

#include "FaceDetect.h"

FaceDetect::FaceDetect(const net_config& config) {
    this->input_height = config.input_height;
    this->input_width = config.input_width;
    this->conf_threshold = config.conf_threshold;
    std::cout<<"...load onnx model..."<<std::endl;
    this->net = cv::dnn::readNetFromONNX(config.model_path);
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void FaceDetect::detect(cv::Mat &src_image,std::vector<Object>&result) {
    cv::Mat image;
    cv::resize(src_image,image,cv::Size(input_width,input_height));
    image_process(image);
    cv::Mat blob = cv::dnn::blobFromImage(image);
    net.setInput(blob);
    cv::Mat out;
    net.forward(out, "score");
    cv::Mat heatmap(image.rows / 4, image.cols / 4,CV_8UC1,cv::Scalar(0,0,0));
    cv::Mat heatmap_dilate(image.rows / 4, image.cols / 4,CV_8UC1,cv::Scalar(0,0,0));
    post_process(heatmap,heatmap_dilate,out);

    const uchar *heatmap_data = heatmap.data;
    const uchar *heatmap_dilate_data = heatmap_dilate.data;
    const float *data = (float *) out.data;
    for (int h = 0; h < heatmap_dilate.rows; ++h) {
        for (int w = 0; w < heatmap_dilate.cols; ++w) {
            if (*heatmap_data==*heatmap_dilate_data){
                float pixel = *heatmap_dilate_data/255.;
                if(pixel>conf_threshold){
                    float p2 = data[1];
                    float p3 = data[2];
                    int cx = (w + p2) * 4;
                    int cy = (h + p3) * 4;

                    float p4 = data[3];
                    float p5 = data[4];
                    float bw = exp(p4) * 4;
                    float bh = exp(p5) * 4;
                    int x = cx-0.5*bw;
                    int y = cy-0.5*bh;
                    result.push_back(Object{cv::Rect(x,y,int(bw),int(bh)),pixel});
                }
            }
            heatmap_data++;
            heatmap_dilate_data++;
            data += out.cols;
        }
    }
}

void FaceDetect::post_process(cv::Mat &heatmap, cv::Mat &heatmap_dilate,const cv::Mat& out) const {
    const float *data = (float *) out.data;
    cv::Mat src(input_height / 4, input_width / 4, CV_32FC1,cv::Scalar(0.,0.,0.));
    for (int h = 0; h < input_height / 4; ++h) {
        for (int w = 0; w < input_width / 4; ++w) {
            int index = h * input_width / 4 + w;
            if (data[0] > conf_threshold) {
                float p1 = data[0];
                src.at<float>(h, w) = p1 * 255;
            }
            data += out.cols;
        }

    }
    src.convertTo(heatmap, CV_8UC1);
    cv::imshow("heatmap",heatmap);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(heatmap,heatmap_dilate,element);
}

void FaceDetect::image_process(cv::Mat &image) {
    std::vector<cv::Mat> rgbChannels(3);
    split(image, rgbChannels);
    for (int i = 0; i < rgbChannels.size(); i++)
    {
        rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / (255.0*stds[i]), (0.0 - mean[i]) / stds[i]);
    }
    merge(rgbChannels, image);
}

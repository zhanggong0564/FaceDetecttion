#include "FaceDetect.h"


void draw_image(const cv::Mat &image, float srcHeightScale, float srcWidthScale, std::vector<Object> &results);

int main(){
    std::string image_path = "/mnt/e/workspace/FaceDetection/deploys/images/0_Parade_marchingband_1_483.jpg";
    std::string onnx_path = "/mnt/e/workspace/FaceDetection/tools/resnet18_1_3_800_800_static.onnx";
    net_config config{
        0.5,
        onnx_path,
        800,
        800
    };
    FaceDetect face_detect(config);
    cv::Mat image = cv::imread(image_path);
//    cv::resize(image,image,cv::Size(800,800));
    float srcHeightScale = (float)image.rows / (float)config.input_height;
    float srcWidthScale = (float)image.cols / (float)config.input_width;
    std::vector<Object> results;
    auto start = static_cast<double>(cv::getTickCount());
    face_detect.detect(image, results);
    auto speed = ((double )cv::getTickCount()- start)/cv::getTickFrequency();
    std::cout<<"speed time: "<<speed<<std::endl;
    draw_image(image, srcHeightScale, srcWidthScale, results);
    cv::imshow("image", image);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}

void draw_image(const cv::Mat &image, float srcHeightScale, float srcWidthScale, std::vector<Object> &results) {
    for (auto & result : results) {
        result.rect.x *=srcWidthScale;
        result.rect.y *=srcHeightScale;
        result.rect.width *=srcWidthScale;
        result.rect.height*=srcHeightScale;
        cv::rectangle(image,result.rect,cv::Scalar(0, 255, 0));
    }
}

#include<iostream>
#include<opencv2/opencv.hpp>
#include <cmath>

using namespace std;
const float mean[3] = {0.4914, 0.4822, 0.4465};
const float stds[3] = {0.2023, 0.1994, 0.2010};

void image_process(cv::Mat & image){
    vector<cv::Mat> rgbChannels(3);
    split(image, rgbChannels);
    for (int i = 0; i < rgbChannels.size(); i++)
    {
        rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / (255.0*stds[i]), (0.0 - mean[i]) / stds[i]);
    }
    merge(rgbChannels, image);
}

int main() {
    string imagepath = "/mnt/e/workspace/FaceDetection/deploys/images/0_Parade_marchingband_1_483.jpg";
    string onnx_path = "/mnt/e/workspace/FaceDetection/tools/resnet18_1_3_800_800_static.onnx";
    cv::Mat image = cv::imread(imagepath);
    cv::resize(image,image,cv::Size(800,800));
    image_process(image);
    cout << image.size << endl;
    cv::Mat blob = cv::dnn::blobFromImage(image);

    cv::dnn::Net model = cv::dnn::readNetFromONNX(onnx_path);
    cout << "read onnx model ...." << endl;
    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    model.setInput(blob);
    cv::Mat out;
    model.forward(out, "score");
    auto *data = (float *) out.data;
//    cv::Mat heatmap(image.rows / 4, image.cols / 4, CV_8UC1);
    cv::Mat heatmap(image.rows / 4, image.cols / 4,CV_8UC1,cv::Scalar(0,0,0));
    cv::Mat src(image.rows / 4, image.cols / 4, CV_32FC1,cv::Scalar(0.,0.,0.));
    int nums = 0;
    for (int h = 0; h < image.rows / 4; ++h) {
        for (int w = 0; w < image.cols / 4; ++w) {
            int index = h * image.cols / 4 + w;

            if (data[0] > 0.59) {
                nums++;
                float p1 = data[0];
                cout << p1 << endl;
                src.at<float>(h, w) = p1 * 255;
//                float p2 = data[1];
//                float p3 = data[2];
//                int cx = (w + p2) * 4;
//                int cy = (h + p3) * 4;
//                cv::circle(image, cv::Point(cx, cy), 2, cv::Scalar(0, 0, 255));
//                float p4 = data[3];
//                float p5 = data[4];
//                float bw = p4 * 4;
//                float bh = p5 * 4;
//                int x1 = cx - 0.5 * bw;
//                int y1 = cy - 0.5 * bh;
//                int x2 = cx + 0.5 * bw;
//                int y2 = cy + 0.5 * bh;
//                cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
            }
            data += out.cols;
        }

    }
    src.convertTo(heatmap, CV_8UC1);
    cv::imshow("heatmap",heatmap);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat heatmap_dilate;
    cv::dilate(heatmap,heatmap_dilate,element);
    uchar *heatmap_data = heatmap.data;
    uchar *heatmap_dilate_data = heatmap_dilate.data;
    int count = 0;
    cout<<"nms"<<endl;
//    delete data;
    data = (float *) out.data;
    for (int h = 0; h < heatmap_dilate.rows; ++h) {
        for (int w = 0; w < heatmap_dilate.cols; ++w) {
            if (*heatmap_data==*heatmap_dilate_data){
                float pixel = *heatmap_dilate_data/255.;
//                dst_heatmap.at<float>(h,w)=pixel;
                if(pixel>0.5){
                    float p2 = data[1];
                    float p3 = data[2];
                    int cx = (w + p2) * 4;
                    int cy = (h + p3) * 4;
                    cv::circle(image, cv::Point(cx, cy), 2, cv::Scalar(0, 0, 255));
                    float p4 = data[3];
                    float p5 = data[4];
                    float bw = exp(p4) * 4;
                    float bh = exp(p5) * 4;
                    int x1 = cx - 0.5 * bw;
                    int y1 = cy - 0.5 * bh;
                    int x2 = cx + 0.5 * bw;
                    int y2 = cy + 0.5 * bh;
                    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
                }
            }
            heatmap_data++;
            heatmap_dilate_data++;
            data += out.cols;
        }
    }
    cout<<count<<endl;

    cv::imshow("image", image);
//    cv::imshow("image1", heatmap_dilate);
    cv::waitKey();
    cv::destroyAllWindows();
    cout << nums << endl;
    cout << "finish" << endl;
    return 0;
}
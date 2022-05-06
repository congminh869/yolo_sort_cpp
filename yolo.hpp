#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"
#include "common.hpp"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

namespace WrappingYolo{

    using namespace std;

    #define INPUT_H Yolo::INPUT_H
    #define INPUT_W Yolo::INPUT_W
    #define CLASS_NUM Yolo::CLASS_NUM
    #define OUTPUT_SIZE Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1 // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    #define INPUT_BLOB_NAME "data"
    #define OUTPUT_BLOB_NAME "prob"    

    /*void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);
    const char* type_name(Type type);*/

    typedef struct Box // if you use struct Box => then you use Box => std::vector<struct Box>
    {
        int class_id;
        int left;
        int top;
        int bottom;
        int right;
        float confidence; 
    }Box;

    typedef std::vector<Box> BoxArray; //typedef tương đối giống với khai báo biến và hàm, tuy nhiên mục đích của nó là để định nghĩa một tên khác cho một kiểu dữ liệu 

    class Infer{
    public:
        Infer(std::string engine_file);
        ~Infer();

        int doInference(cv::Mat src, BoxArray &result, cv::Mat &dts);

    private:
        IExecutionContext* context;
        IRuntime* runtime;
        ICudaEngine* engine;
        cudaStream_t stream;
        float* buffers[2];

        float prob[BATCH_SIZE * OUTPUT_SIZE];
        int inputIndex;
        int outputIndex;  
        uint8_t* img_host = nullptr;
        uint8_t* img_device = nullptr;

        Logger gLogger;
    };
}; // namespace Yolo

#endif // YOLO_HPP

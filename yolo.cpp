#include "yolo.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

namespace WrappingYolo{
    using namespace cv;
    using namespace std;


    static int get_width(int x, float gw, int divisor = 8) {
        return int(ceil((x * gw) / divisor)) * divisor;
    }

    static int get_depth(int x, float gd) {
        if (x == 1) return 1;
        int r = round(x * gd);
        if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
            --r;
        }
        return std::max<int>(r, 1);
}

    Infer::Infer(std::string engine_file) // reconstruction fuction , class in yolo.hpp
    {
        //copy same in src tensorrtx-yolov5
        //Read file engine
        std::ifstream file(engine_file, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engine_file << " error!" << std::endl;
            return;
        }
        char *trtModelStream = nullptr;
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();

        //static float prob[BATCH_SIZE * OUTPUT_SIZE];
        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;
        assert(engine->getNbBindings() == 2);

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

        // Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));
        // prepare input data cache in pinned memory 
        CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
        // prepare input data cache in device memory
        CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    }

    Infer::~Infer()//when not use Inter -> delete class
    {
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(img_device));
        CUDA_CHECK(cudaFreeHost(img_host));
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }

    int Infer::doInference(cv::Mat src,BoxArray &result, cv::Mat &dts)
    {
        float* buffer_idx = (float*)buffers[inputIndex];
        if (src.empty())
            return -1;

        size_t  size_image = src.cols * src.rows * 3;
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;
        //copy data to pinned memory
        memcpy(img_host,src.data,size_image);
        //copy data to device memory
        CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
        preprocess_kernel_img(img_device, src.cols, src.rows, buffer_idx, INPUT_W, INPUT_H, stream); 
        buffer_idx += size_image_dst;

        // Run inference
        auto start = std::chrono::system_clock::now();
        context->enqueue(BATCH_SIZE, (void**)buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<Yolo::Detection> res;
        nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
        dts = src.clone();
        int ret = 0;
        for (size_t j = 0; j < res.size(); j++) 
        {
            cv::Rect r = get_rect(dts, res[j].bbox);

            Box box;
            box.class_id = (int)res[j].class_id;
            box.left = r.x;
            box.top = r.y;
            box.bottom = r.y + r.height;
            box.right = r.x + r.width;
            box.confidence = res[j].conf;
            result.push_back(box);

            //if((int)res[j].class_id == 0)
            //{
                cv::rectangle(dts, r, cv::Scalar(0x27, 0xC1, 0x36), 1);
                ret++;
            //}
            
            //cv::putText(dts, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        //cv::imwrite("_" + file_names[f - fcount + 1 + b], img);

        return ret;
    }
};
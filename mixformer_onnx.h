//
// Create by Daniel Lee on 2023/9/22
//

#ifndef MIXFORMER_H
#define MIXFORMER_H

#include <vector> 
#include <map>
#include <memory>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

# define USE_CUDA

struct DrBBox {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct DrOBB {
    DrBBox box;
    float score;
    int class_id;
};

struct Config {
    std::vector<float> window;

    float template_factor = 2.0;
    float search_factor = 4.0; // 5.0
    float template_size = 128; //192
    float search_size = 256; // 384
    float stride = 16;
    int feat_sz = 16; // 24
};
enum
{
    CHW = 0, HWC = 1
};


class Mixformer {

public:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    // CPU MemoryInfo
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
     // hardcode input node names
    unsigned int num_inputs = 2;
    std::vector<const char *> input_node_names = {
        "z",
        "x"
    };
    // init dynamic input dims
    std::vector<std::vector<int64_t>> input_node_dims = {
        {1, 3, 128, 128}, // z  (b=1,c,h,w)
        {1, 3, 256, 256} // x
    }; 
    std::vector<float> input_value_handler_z;
    std::vector<float> input_value_handler_x;

    // hardcode output node names
    unsigned int num_outputs = 3;
    std::vector<const char *> output_node_names = {
        "size_map",
        "score_map", 
        "offset_map"
    };

    const char *onnx_path = nullptr;
    
    const char *log_id = nullptr;

public:     
    explicit Mixformer(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~Mixformer(); //override

    void init(const cv::Mat &img, DrOBB bbox);
    
    const DrOBB &track(const cv::Mat &img);
    
    // state  dynamic
    DrBBox state;
    
    // config static
    Config cfg; 

protected:
    const unsigned int num_threads; // initialize at runtime.

private:

    std::vector<Ort::Value>  transform(const cv::Mat &mat_z, const cv::Mat &mat_x);

    void map_box_back(DrBBox &pred_box, float resize_factor);

    void clip_box(DrBBox &box, int height, int wight, int margin);

    void cal_bbox(std::vector<Ort::Value> &output_tensors, DrBBox &pred_box, float &max_score, float resize_factor);

    void sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor);

public:
    
    std::vector<float> hann(int sz);

    Ort::Value create_tensor(const cv::Mat &mat, const std::vector<int64_t> &tensor_dims,
                            const Ort::MemoryInfo &memory_info_handler,
                            std::vector<float> &tensor_value_handler,
                            unsigned int data_format) throw(std::runtime_error);

    cv::Mat normalize(const cv::Mat &mat, float mean, float scale);

    cv::Mat normalize(const cv::Mat &mat, const float mean[3], const float scale[3]);

    void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);

    void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);

    void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3]);

private:
    const float means[3]  = {0.406*255, 0.485*255, 0.456*255}; // BGR
    const float norms[3] = {1/(0.225*255), 1/(0.229*255), 1/(0.224*255)}; // BGR
    // const float mean_vals[3] = {103.53f, 116.28f, 123.675f}; // BGR
    // const float scale_vals[3] = {0.017429f, 0.017507f, 0.017125f};

    Ort::Value *x = nullptr;
    Ort::Value *z = nullptr;

    cv::Mat z_patch;

    DrOBB object_box;
};

#endif 

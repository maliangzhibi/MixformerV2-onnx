//
// Create by Daniel Lee on 2023/9/22
//

#ifndef MIXFORMER_TRT_H
#define MIXFORMER_TRT_H

#include <iostream>
#include <fstream>
#include <vector> 
#include <map>
#include <memory>
#include <assert.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <half.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "cuda_fp16.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id

# define USE_CUDA

using namespace nvinfer1;

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
    // std::vector<float> window;
    float template_factor = 2.0;
    float search_factor = 4.5; // 5.0
    float template_size = 112; //192
    float search_size = 224; // 384
    float stride = 16;
    int feat_sz = 14; // 24
    int update_interval = 200;
};
enum
{
    CHW = 0, HWC = 1
};


class MixformerTRT {

public:
    Logger gLogger;

    const char* INPUT_BLOB_IMGT_NAME = "img_t";
    const char* INPUT_BLOB_IMGOT_NAME = "img_ot";
    const char* INPUT_BLOB_IMGSEARCH_NAME = "img_search";

    const char* OUTPUT_BLOB_PREDBOXES_NAME = "pred_boxes"; 
    const char* OUTPUT_BLOB_PREDSCORES_NAME = "pred_scores"; 

    char *trt_model_stream = nullptr;
    
    size_t size{0};

    // define the TensorRT runtime, engine, context,stream
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;

    cudaStream_t stream;

    //  // hardcode input node names
    // unsigned int num_inputs = 3;
    // std::vector<const char *> input_node_names = {
    //     "img_t",
    //     "img_ot",
    //     "img_search"
    // };
    // init dynamic input dims
    std::vector<std::vector<int64_t>> input_node_dims = {
        {1, 3, 112, 112}, // z  (b=1,c,h,w)
        {1, 3, 112, 112}, // z  (b=1,c,h,w)
        {1, 3, 224, 224} // x
    };

    half_float::half *input_imt = nullptr;
    half_float::half *input_imot = nullptr;
    half_float::half *input_imsearch = nullptr;
    half_float::half *output_pred_boxes = nullptr;
    half_float::half *output_pred_scores = nullptr;


public:
    MixformerTRT(std::string &engine_name);

    ~MixformerTRT(); //override

    void init(const cv::Mat &img, DrOBB bbox);    
    
    const DrOBB &track(const cv::Mat &img);
    
    // state  dynamic
    DrBBox state;
    
    // config static
    Config cfg; 

// protected:
//     const unsigned int num_threads; // initialize at runtime.

private:

    void transform(const cv::Mat &mat_z, const cv::Mat &mat_oz, const cv::Mat &mat_x);

    void map_box_back(DrBBox &pred_box, float resize_factor);

    void clip_box(DrBBox &box, int height, int wight, int margin);

    void cal_bbox(half_float::half *boxes_ptr, half_float::half * scores_ptr, DrBBox &pred_box, float &max_score, float resize_factor);

    void sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor);

    float fp16_to_float(half_float::half value);

public:

    cv::Mat normalize(const cv::Mat &mat, float mean, float scale);

    cv::Mat normalize(const cv::Mat &mat, const float mean[3], const float scale[3]);

    void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);

    void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);

    void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3]);

    void deserialize_engine(std::string &engine_name);

    void infer(
        half_float::half  *input_imt,
        half_float::half  *input_imot,
        half_float::half  *input_imsearch,
        half_float::half  *output_pred_boxes,
        half_float::half  *output_pred_scores,
        cv::Size input_imt_shape,
        cv::Size input_imsearch_shape);
    
    half_float::half* blob_from_image_half(cv::Mat& img);

private:
    const float means[3]  = {0.406*255, 0.485*255, 0.456*255}; // BGR
    const float norms[3] = {1/(0.225*255), 1/(0.229*255), 1/(0.224*255)}; // BGR
    float max_pred_score = -1.f;
    float max_score_decay = 1.f;

    cv::Mat z_patch; // template
    cv::Mat oz_patch; // online_template
    cv::Mat max_oz_patch; // online max template

    DrOBB object_box;
    int frame_id = 0;

    int output_pred_boxes_size = 1;
    int output_pred_scores_size = 1;

    // 数据尺度的定义
    static const int INPUT_IMT_W = 112;
    static const int INPUT_IMOT_W = 112;
    static const int INPUT_IMSEARCH_W = 224;

    static const int INPUT_IMT_H = 112;
    static const int INPUT_IMOT_H = 112;
    static const int INPUT_IMSEARCH_H = 224;

    // half_float::half* blob_imt = nullptr;
    // half_float::half* blob_imot = nullptr;
    // half_float::half* blob_imsearch = nullptr;
};

#endif 

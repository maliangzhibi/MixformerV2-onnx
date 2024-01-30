// 
// Create by Daniel Lee on 2023/9/22
// 
#include <cstdlib>
#include <string>
#include <cmath>
#include "mixformer_trt.h"

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#ifdef TIME
    struct timeval tv;
    uint64_t time_last;
    double time_ms;
#endif

MixformerTRT::MixformerTRT(std::string &engine_name)
{
    // deserialize engine
    this->deserialize_engine(engine_name);
    
    // // 分配内存给输入和输出
    // this->input_imt = new half_float::half[cfg.template_size * cfg.template_size * 3];
    // this->input_imot = new half_float::half[cfg.template_size * cfg.template_size * 3];
    // this->input_imsearch = new half_float::half[cfg.search_size * cfg.search_size * 3];

    auto out_dims_0 = this->engine->getBindingDimensions(3);
    for(int j=0; j < out_dims_0.nbDims; j++)
    {
        this->output_pred_boxes_size *= out_dims_0.d[j];
    }
    // this->output_pred_boxes = new half_float::half[this->output_pred_boxes_size];

    auto out_dims_1 = this->engine->getBindingDimensions(4);
    for(int j=0; j < out_dims_1.nbDims; j++)
    {
        this->output_pred_scores_size *= out_dims_1.d[j];
    }
    // this->output_pred_scores = new half_float::half[this->output_pred_scores_size];
    // std::cout << "output_pred_bbox_size: " << this->output_pred_boxes_size << std::endl;
    // std::cout << "output_pred_bbox_score: " << this->output_pred_scores_size << std::endl;
    this->output_pred_boxes = new float[this->output_pred_boxes_size];
    this->output_pred_scores = new float[this->output_pred_scores_size];
    
}

MixformerTRT::~MixformerTRT(){
    delete context;
    delete engine;
    delete runtime;
    delete[] trt_model_stream;
    delete[] this->output_pred_boxes;
    delete[] this->output_pred_scores;
    cudaStreamDestroy(stream);
}

void MixformerTRT::deserialize_engine(std::string &engine_name){
    // create a model using the API directly and serialize it to a stream
    // char *trt_model_stream{nullptr};
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {  
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        this->trt_model_stream = new char[this->size];
        assert(this->trt_model_stream);
        file.read(trt_model_stream, this->size);
        file.close();
    }

    this->runtime = createInferRuntime(this->gLogger);    
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trt_model_stream,
                                                        this->size);    
    assert(this->engine != nullptr);

    this->context = this->engine->createExecutionContext();
    assert(context != nullptr);
    // delete[] trt_model_stream;
}

void MixformerTRT::infer(
    float *input_imt,
    float *input_imot,
    float *input_imsearch,
    float *output_pred_boxes,
    float *output_pred_scores,
    cv::Size input_imt_shape,
    cv::Size input_imot_shape,
    cv::Size input_imsearch_shape)
{
    assert(engine->getNbBindings() == 5);
    void* buffers[5];
    
    const int inputImgtIndex = engine->getBindingIndex(INPUT_BLOB_IMGT_NAME);
    std::cout << ">>>debug infer start. " << (engine->getBindingDataType(inputImgtIndex) == nvinfer1::DataType::kFLOAT) << std::endl;
    assert(engine->getBindingDataType(inputImgtIndex) == nvinfer1::DataType::kFLOAT);
    const int inputImgotIndex = engine->getBindingIndex(INPUT_BLOB_IMGOT_NAME);
    assert(engine->getBindingDataType(inputImgotIndex) == nvinfer1::DataType::kFLOAT);
    const int inputImgsearchIndex = engine->getBindingIndex(INPUT_BLOB_IMGSEARCH_NAME);
    assert(engine->getBindingDataType(inputImgsearchIndex) == nvinfer1::DataType::kFLOAT);

    const int outputPredboxesIndex = engine->getBindingIndex(OUTPUT_BLOB_PREDBOXES_NAME);
    assert(engine->getBindingDataType(outputPredboxesIndex) == nvinfer1::DataType::kFLOAT);
    const int outputPredscoresIndex = engine->getBindingIndex(OUTPUT_BLOB_PREDSCORES_NAME);
    assert(engine->getBindingDataType(outputPredscoresIndex) == nvinfer1::DataType::kFLOAT);

    int mBatchSize = engine->getMaxBatchSize();
    
    // create gpu buffer on devices
    CHECK(cudaMalloc(&buffers[inputImgtIndex], 3 * input_imt_shape.height * input_imt_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[inputImgotIndex], 3 * input_imot_shape.height * input_imot_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[inputImgsearchIndex], 3 * input_imsearch_shape.height * input_imsearch_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputPredboxesIndex], this->output_pred_boxes_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputPredscoresIndex], this->output_pred_scores_size * sizeof(float)));
   
    // create stream
    CHECK(cudaStreamCreate(&stream));
       
    // DMA input batch  data to device, infer on the batch asynchronously,  and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputImgtIndex], input_imt, 3 * input_imt_shape.height * input_imt_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputImgotIndex], input_imot, 3 * input_imot_shape.height * input_imot_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputImgsearchIndex], input_imsearch, 3 * input_imsearch_shape.height * input_imsearch_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    // inference
    context->enqueue(mBatchSize, buffers, stream, nullptr);
    
    CHECK(cudaMemcpyAsync(output_pred_boxes, buffers[outputPredboxesIndex], this->output_pred_boxes_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_pred_scores, buffers[outputPredscoresIndex], this->output_pred_scores_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    // release buffers
    CHECK(cudaFree(buffers[inputImgtIndex]));
    CHECK(cudaFree(buffers[inputImgotIndex]));
    CHECK(cudaFree(buffers[inputImgsearchIndex]));
    CHECK(cudaFree(buffers[outputPredboxesIndex]));
    CHECK(cudaFree(buffers[outputPredscoresIndex]));
    std::cout << ">>>debug infer end. "  << std::endl;
}

// put z and x into transform
// void  MixformerTRT::transform(const cv::Mat &mat_z, const cv::Mat &mat_oz, const cv::Mat &mat_x)
void  MixformerTRT::transform(cv::Mat &mat_z, cv::Mat &mat_oz, cv::Mat &mat_x)
{
    this->blob_from_image_half(mat_z, mat_oz, mat_x);
}

void MixformerTRT::blob_from_image_half(cv::Mat& img, cv::Mat &imgot, cv::Mat &imgx) {
    cv::Mat imt_t;
    cv::Mat imot_t;
    cv::Mat imx_t;
    // cv::imshow("BGR", img);
    // cv::waitKey(500);
    cvtColor(img, imt_t, cv::COLOR_BGR2RGB);
    // cv::imshow("RGB", imt_t);
    // cv::waitKey(500);
    cvtColor(imgot, imot_t, cv::COLOR_BGR2RGB);
    cvtColor(imgx, imx_t, cv::COLOR_BGR2RGB);

    // img = imt_t.clone();
    // imgot = imot_t.clone();
    // imgx = imx_t.clone();

    // 需及时释放
    this->input_imt = new float[img.total() * 3]; // Use __fp16 data type for blob array
    this->input_imot = new float[imgot.total() * 3]; // Use __fp16 data type for blob array
    this->input_imsearch = new float[imgx.total() * 3]; // Use __fp16 data type for blob array

    half_norm(imt_t, this->input_imt);
    half_norm(imot_t, this->input_imot);
    half_norm(imx_t, this->input_imsearch);
}

void MixformerTRT::half_norm(const cv::Mat &img, float* input_data)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;

    // std::vector<float> meanb = {0.485, 0.456, 0.406}; //RGB
    // std::vector<float> std_var = {0.229, 0.224, 0.225};

    cv::Mat img_cp;
    img_cp = img.clone();
    
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < img_h; h++) {
            for (size_t w = 0; w < img_w; w++) {
                input_data[c * img_w * img_h + h * img_w + w] = 
                    cv::saturate_cast<float>((((float)img_cp.at<cv::Vec3b>(h, w)[c]) - mean_vals[c]) * norm_vals[c]);
                    // cv::saturate_cast<half_float::half>((((float)img_cp.at<cv::Vec3b>(h, w)[c])/255.0f - meanb[c]) / std_var[c]);
            }
        }
    }

}

// void MixformerTRT::preprocessor_img(cv::Mat &img, half_float::half* output_data)
// void MixformerTRT::preprocessor_img(cv::Mat &img)
// {
//     std::cout << ">>>debug img type 0: " << (img.type()==CV_16FC3) << std::endl;
//     // Convert BGR to RGB
//     cv::Mat img_rgb;
//     cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
//     // img = img_rgb.clone();
//     // cv::imshow("deb", img);
//     // cv::waitKey(500);
//     std::cout << ">>>debug img type1: " << (img.type()==CV_16FC3) << std::endl;
//     this->normalize_inplace(img_rgb, means_fp16, norms_fp16);
//     std::cout << ">>>debug img type2: " << (img.type()==CV_16FC3) << std::endl;
//     img = img_rgb;
// }

// void MixformerTRT::normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3])
// {
//     std::cout<< ">>>debug>>> normalize_inplace0." << std::endl;
//     if (mat_inplace.type() != CV_16FC3){
//         std::cout<< ">>>debug>>> normalize_inplace0." << std::endl;
//         cv::imshow("nin", mat_inplace);
//         cv::waitKey(5000);
//         mat_inplace.convertTo(mat_inplace, CV_16FC3);
//         std::cout<< ">>>debug>>> normalize_inplace1." << std::endl;
//     }
    
//     for (unsigned int i = 0; i < mat_inplace.rows; ++i)
//     {
//         cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
//         for (unsigned int j = 0; j < mat_inplace.cols; ++j)
//         {
//             p[j][0] = (p[j][0] - mean[0]) * scale[0];
//             p[j][1] = (p[j][1] - mean[1]) * scale[1];
//             p[j][2] = (p[j][2] - mean[2]) * scale[2];
//         }
//     }
// }

// void MixformerTRT::convertMatToFP16(cv::Mat& mat, float* fp16Data) {
//     // 检查输入图像的数据类型是否为 CV_16FC3
//     if (mat.type() != CV_16FC3) {
//         std::cerr << "Input image data type is not CV_16FC3." << std::endl;
//         return; // 返回错误或采取适当的处理
//     }

//     // 获取图像的高度和宽度
//     int height = mat.rows;
//     int width = mat.cols;

//     // 遍历图像并将数据转换为 FP16 存储在 fp16Data 中
//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x < width; ++x) {
//             cv::Vec3f pixel = mat.at<cv::Vec3f>(y, x);
//             for (int c = 0; c < 3; ++c) {
//                 // 将 CV_16FC3 中的每个通道数据转换为 FP16 存储
//                 fp16Data[(y * width + x) * 3 + c] = float(pixel[c]);
//             }
//         }
//     }
// }

void MixformerTRT::blob_from_image_half(cv::Mat& img, float* input_blob_half) {
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < img_h; h++) {
            for (size_t w = 0; w < img_w; w++) {
                input_blob_half[c * img_w * img_h + h * img_w + w] = float(img.at<cv::Vec3b>(h, w)[c]);
                    // cv::saturate_cast<half_float::half>((((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std_var[c]);
                    // cv::saturate_cast<half_float::half>((float)img.at<cv::Vec3b>(h, w)[c]);
                // std::cout << input_blob_half[c * img_w * img_h + h * img_w + w] << std::endl;
            }
        }
    }  
}

void MixformerTRT::init(const cv::Mat &img, DrOBB bbox)
{
    // get subwindow
    cv::Mat zt_patch; 
    cv::Mat ozt_patch;   
    float resize_factor = 1.f;
    this->sample_target(img, zt_patch, bbox.box, this->cfg.template_factor, this->cfg.template_size, resize_factor);
    this->sample_target(img, ozt_patch, bbox.box, this->cfg.template_factor, this->cfg.template_size, resize_factor);
    // cv::Mat oz_patch = z_patch.clone(); 
    this->z_patch = zt_patch;
    this->oz_patch = ozt_patch;
    this->state = bbox.box;
}

const DrOBB &MixformerTRT::track(const cv::Mat &img)
{
    // if (img.empty()) return;
    // get subwindow
    cv::Mat x_patch;
    this->frame_id += 1;
    float resize_factor = 1.f;
    this->sample_target(img, x_patch, this->state, this->cfg.search_factor, this->cfg.search_size, resize_factor);
    
    // preprocess input tensor
    this->transform(this->z_patch, this->oz_patch, x_patch);

    // inference score， size  and offsets
    cv::Size input_imt_shape = this->z_patch.size();
    cv::Size input_imot_shape = this->oz_patch.size();
    cv::Size input_imsearch_shape = x_patch.size();
    
    // cv::Mat inputMatT = input_zx.at(0);
    // cv::Mat inputMatOt = input_zx.at(1);
    // cv::Mat inputMatSearch = input_zx.at(2);
    
    // // 获取指向图像数据的指针
    // float* inputPtrT = reinterpret_cast<float*>(inputMatT.data);
    // float* inputPtrOt = reinterpret_cast<float*>(inputMatOt.data);
    // float* inputPtrSearch = reinterpret_cast<float*>(inputMatSearch.data);    
    // std::cout << ">>>debug img type 0: " << input_imt_shape.height << std::endl;
    this->infer(input_imt, input_imot, input_imsearch, 
              output_pred_boxes, output_pred_scores, 
              input_imt_shape, input_imot_shape, 
              input_imsearch_shape);
    
    delete[] this->input_imt;
    delete[] this->input_imot;
    delete[] this->input_imsearch;
    // std::cout << "+++++++++debug++++++++++" << std::endl;
    // std::cout << "开始跟踪2: " << std::endl;
    // std::cout << ">>>debug img type 1: "  << std::endl;
    DrBBox pred_box;
    float pred_score;
    // std::cout << "after infer cx cy w h: "
    //         << output_pred_boxes[0]<< " " 
    //         << output_pred_boxes[1] << " "
    //         << output_pred_boxes[2] << " "
    //         << output_pred_boxes[3] << std::endl;
    this->cal_bbox(output_pred_boxes, output_pred_scores, pred_box, pred_score, resize_factor);
    
    this->map_box_back(pred_box, resize_factor);
    this->clip_box(pred_box, img.rows, img.cols, 10);
    
    object_box.box = pred_box;
    object_box.class_id = 0;
    object_box.score = pred_score;

    this->state = object_box.box;

    this->max_pred_score = this->max_pred_score * this->max_score_decay;
    // update template
    if (pred_score > 0.5 && pred_score > this->max_pred_score)
    {
      this->sample_target(img, this->max_oz_patch, this->state, this->cfg.template_factor, this->cfg.template_size, resize_factor);
      this->max_pred_score = pred_score;
    }

    if (this->frame_id % this->cfg.update_interval == 0)
    {
      this->oz_patch = this->max_oz_patch;
      this->max_pred_score = -1.0;
      this->max_oz_patch = this->oz_patch;
    }
    
    return object_box;
}

// calculate bbox
void MixformerTRT::cal_bbox(float *boxes_ptr, float * scores_ptr, DrBBox &pred_box, float &max_score, float resize_factor) {
    auto cx = boxes_ptr[0];
    auto cy = boxes_ptr[1];
    auto w = boxes_ptr[2];
    auto h = boxes_ptr[3];
    std::cout << "cal_bbox cx cy w h "<< cx << " " << cy << " " << w << " " << h << std::endl;
    cx = cx * this->cfg.search_size / resize_factor;
    cy = cy * this->cfg.search_size / resize_factor;
    w = w * this->cfg.search_size / resize_factor;
    h = h * this->cfg.search_size / resize_factor;
    
    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;

    max_score = scores_ptr[0];
}

void MixformerTRT::map_box_back(DrBBox &pred_box, float resize_factor) {
    float cx_prev = this->state.x0 + 0.5 * (this->state.x1 - this->state.x0);
    float cy_prev = this->state.y0 + 0.5 * (this->state.y1 - this->state.y0);

    float half_side = 0.5 * this->cfg.search_size / resize_factor;

    float w = pred_box.x1 - pred_box.x0;
    float h = pred_box.y1 - pred_box.y0;
    float cx = pred_box.x0 + 0.5 * w;
    float cy = pred_box.y0 + 0.5 * h;

    float cx_real = cx + (cx_prev - half_side);
    float cy_real = cy + (cy_prev - half_side);

    pred_box.x0 = cx_real - 0.5 * w;
    pred_box.y0 = cy_real - 0.5 * h;
    pred_box.x1 = cx_real + 0.5 * w;
    pred_box.y1 = cy_real + 0.5 * h;
}

void MixformerTRT::clip_box(DrBBox &box, int height, int wight, int margin) {
    box.x0 = std::min(std::max(0, int(box.x0)), wight - margin);
    box.y0 = std::min(std::max(0, int(box.y0)), height - margin);
    box.x1 = std::min(std::max(margin, int(box.x1)), wight);
    box.y1 = std::min(std::max(margin, int(box.y1)), height);
}

void MixformerTRT::sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor) {
    /* Extracts a square crop centrered at target_bb box, of are search_area_factor^2 times target_bb area

    args:
        im: Img image
        target_bb - target box [x0, y0, x1, y1]
        search_area_factor - Ratio of crop size to target size
        output_sz - Size
    
    */
   int x = target_bb.x0;
   int y = target_bb.y0;
   int w = target_bb.x1 - target_bb.x0;
   int h = target_bb.y1 - target_bb.y0;
   int crop_sz = std::ceil(std::sqrt(w *h) * search_area_factor);

   float cx = x + 0.5 * w;
   float cy = y + 0.5 * h;
   int x1 = std::round(cx - crop_sz * 0.5);
   int y1 = std::round(cy - crop_sz * 0.5);

   int x2 = x1 + crop_sz;
   int y2 = y1 + crop_sz;

   int x1_pad = std::max(0, -x1);
   int x2_pad = std::max(x2 - im.cols +1, 0);
   
   int y1_pad = std::max(0, -y1);
   int y2_pad = std::max(y2- im.rows + 1, 0);

   // Crop target
   cv::Rect roi_rect(x1+x1_pad, y1+y1_pad, (x2-x2_pad)-(x1+x1_pad), (y2-y2_pad)-(y1+y1_pad));
   cv::Mat roi = im(roi_rect);

   // Pad
   cv::copyMakeBorder(roi, croped, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

   // Resize
   cv::resize(croped, croped, cv::Size(output_sz, output_sz));

   resize_factor = output_sz * 1.f / crop_sz;
}

// cv::Mat MixformerTRT::normalize(const cv::Mat &mat, float mean, float scale)
// {
//   cv::Mat matf;
//   if (mat.type() != CV_32FC3) mat.convertTo(matf, CV_32FC3);
//   else matf = mat; // reference
//   return (matf - mean) * scale;
// }

// cv::Mat MixformerTRT::normalize(const cv::Mat &mat, const float mean[3], const float scale[3])
// {
//   cv::Mat mat_copy;
//   if (mat.type() != CV_32FC3) mat.convertTo(mat_copy, CV_32FC3);
//   else mat_copy = mat.clone();
//   for (unsigned int i = 0; i < mat_copy.rows; ++i)
//   {
//     cv::Vec3f *p = mat_copy.ptr<cv::Vec3f>(i);
//     for (unsigned int j = 0; j < mat_copy.cols; ++j)
//     {
//       p[j][0] = (p[j][0] - mean[0]) * scale[0];
//       p[j][1] = (p[j][1] - mean[1]) * scale[1];
//       p[j][2] = (p[j][2] - mean[2]) * scale[2];
//     }
//   }
//   return mat_copy;
// }

// void MixformerTRT::normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale)
// {
//   outmat = this->normalize(inmat, mean, scale);
// }

// void MixformerTRT::normalize_inplace(cv::Mat &mat_inplace, float mean, float scale)
// {
//   if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
//   this->normalize(mat_inplace, mat_inplace, mean, scale);
// }

// void MixformerTRT::normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3])
// {
//   if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
//   for (unsigned int i = 0; i < mat_inplace.rows; ++i)
//   {
//     cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
//     for (unsigned int j = 0; j < mat_inplace.cols; ++j)
//     {
//       p[j][0] = (p[j][0] - mean[0]) * scale[0];
//       p[j][1] = (p[j][1] - mean[1]) * scale[1];
//       p[j][2] = (p[j][2] - mean[2]) * scale[2];
//     }
//   }
// }

// float MixformerTRT::fp16_to_float(float value)
// {
//     return static_cast<float>(value);
// }








// 
// Create by Daniel Lee on 2023/9/22
// 
#include <cstdlib>
#include <string>
#include <cmath>
#include "mixformer_onnx.h"

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#ifdef TIME
    struct timeval tv;
    uint64_t time_last;
    double time_ms;
#endif


// static std::vector<float> hann(int sz) {
std::vector<float> Mixformer::hann(int sz) {
    std::vector<float> hann1d(sz);
    std::vector<float> hann2d(sz * sz);
    for (int i = 1; i < sz + 1; i++) {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (sz+1) );
        hann1d[i-1] = w;
    }
    for (int i = 0; i < sz; i++) {
        for (int j=0; j < sz; j++) {
            hann2d[i*sz + j] = hann1d[i] * hann1d[j];
        }
    }
   return hann2d;
}

// put z and x into transform
std::vector<Ort::Value>  Mixformer::transform(const cv::Mat &mat_z, const cv::Mat &mat_x)
{
    cv::Mat z = mat_z.clone();
    cv::Mat x = mat_x.clone();

    cv::cvtColor(z, z, cv::COLOR_BGR2RGB);
    cv::cvtColor(x, x, cv::COLOR_BGR2RGB);
    // z.convertTo(z, CV_32FC3, 1. / 255.f, 0.f);
    // x.convertTo(x, CV_32FC3, 1. / 255.f, 0.f);
    
    this->normalize_inplace(z, means, norms); // float32
    this->normalize_inplace(x, means, norms); // float32

    std::vector<Ort::Value> input_tensors;

    input_tensors.emplace_back(this->create_tensor(
        z, input_node_dims.at(0), memory_info_handler,
        input_value_handler_z, CHW));

    input_tensors.emplace_back(this->create_tensor(
        x, input_node_dims.at(1), memory_info_handler,
        input_value_handler_x, CHW));

    return input_tensors;
}

void Mixformer::init(const cv::Mat &img, DrOBB bbox)
{
    // get subwindow
    cv::Mat z_patch;
    float resize_factor = 1.f;
    this->sample_target(img, z_patch, bbox.box, this->cfg.template_factor, this->cfg.template_size, resize_factor);
    this->z_patch = z_patch;
    this->state = bbox.box;
}

const DrOBB &Mixformer::track(const cv::Mat &img)
{
    // if (img.empty()) return;
    // get subwindow
    cv::Mat x_patch;
    float resize_factor = 1.f;
    this->sample_target(img, x_patch, this->state, this->cfg.search_factor, this->cfg.search_size, resize_factor);

    // preprocess input tensor
    std::vector<Ort::Value> input_tensor_xz = this->transform(this->z_patch, x_patch);
    // std::cout << "开始跟踪1: " << input_tensor_xz.size()<< std::endl;
    // inference score， size  and offsets
    std::vector<Ort::Value> output_tensors = ort_session->Run(
        Ort::RunOptions{nullptr}, input_node_names.data(),
        input_tensor_xz.data(), 2, output_node_names.data(), 3
    );
    // std::cout << "开始跟踪2: " << std::endl;
    DrBBox pred_box;
    float max_score;
    this->cal_bbox(output_tensors, pred_box, max_score, resize_factor);
    
    this->map_box_back(pred_box, resize_factor);
    this->clip_box(pred_box, img.rows, img.cols, 10);
    
    object_box.box = pred_box;
    object_box.class_id = 0;
    object_box.score = max_score;

    this->state = object_box.box;

    return object_box;
}

// calculate bbox
void Mixformer::cal_bbox(std::vector<Ort::Value> &output_tensors, DrBBox &pred_box, float &max_score, float resize_factor) {
    Ort::Value &sizes_tensor = output_tensors.at(0); // (1,1,16,16)
    Ort::Value &scores_tensor = output_tensors.at(1); // (1,2,16,16)
    Ort::Value &offsets_tensor = output_tensors.at(2); // (1,2,16,16)

    auto scores_dims = scores_tensor.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    auto sizes_dims = sizes_tensor.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    auto offsets_dims = offsets_tensor.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    
    const unsigned int scores_b = scores_dims.at(0);
    const unsigned int scores_c = scores_dims.at(1);
    const unsigned int scores_height = scores_dims.at(2);
    const unsigned int scores_width = scores_dims.at(3);

    int scores_elem_size = scores_height * scores_width;

    auto scores_ptr = scores_tensor.GetTensorData<float>();
    // Add hann window, get max value and index.
    std::cout << "开始跟踪: " << scores_b << " " << scores_c << " " << scores_height << " " << scores_width << " " << scores_elem_size << std::endl;
    float max_value = this->cfg.window[0] * scores_ptr[0];
    int max_idx_y = 0; int max_idx_x = 0; int max_idx = 0;
    float tmp_score = 0.f;
    for (int i = 0; i < scores_elem_size; i++) {
        tmp_score = this->cfg.window[i] * scores_ptr[i];
        if (tmp_score > max_value) {
            max_idx = i;
            max_value = tmp_score;
            // std::cout << "max_value: " << max_value << std::endl;
        }
    }
    
    max_idx_y = max_idx / scores_width;
    max_idx_x = max_idx % scores_width;
    
    auto sizes_ptr = sizes_tensor.GetTensorData<float>();
    auto offsets_ptr = offsets_tensor.GetTensorData<float>();

    float cx = (max_idx_x + offsets_ptr[max_idx_y * offsets_dims.at(3) + max_idx_x]) * 1.f / this->cfg.feat_sz;
    float cy = (max_idx_y + offsets_ptr[offsets_dims.at(3) * offsets_dims.at(2) + max_idx_y * offsets_dims.at(3) + max_idx_x]) *1.f / this->cfg.feat_sz;

    float w = sizes_ptr[max_idx_y * sizes_dims.at(3) + max_idx_x];
    float h = sizes_ptr[sizes_dims.at(3) * sizes_dims.at(2) + max_idx_y * sizes_dims.at(3) + max_idx_x];
  
    cx = cx * this->cfg.search_size / resize_factor;
    cy = cy * this->cfg.search_size / resize_factor;
    w = w * this->cfg.search_size / resize_factor;
    h = h * this->cfg.search_size / resize_factor;
    
    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;    

    max_score = max_value;
}

void Mixformer::map_box_back(DrBBox &pred_box, float resize_factor) {
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

void Mixformer::clip_box(DrBBox &box, int height, int wight, int margin) {
    box.x0 = std::min(std::max(0, int(box.x0)), wight - margin);
    box.y0 = std::min(std::max(0, int(box.y0)), height - margin);
    box.x1 = std::min(std::max(margin, int(box.x1)), wight);
    box.y1 = std::min(std::max(margin, int(box.y1)), height);
}

void Mixformer::sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor) {
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

Ort::Value Mixformer::create_tensor(const cv::Mat &mat, 
        const std::vector<int64_t> &tensor_dims, 
        const Ort::MemoryInfo &memory_info_handler,
        std::vector<float> &tensor_value_handler,
        unsigned int data_format = CHW)
throw(std::runtime_error)
{
  const unsigned int rows = mat.rows;
  const unsigned int cols = mat.cols;
  const unsigned int channels = mat.channels();

  cv::Mat mat_ref;
  if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
  else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

  if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
  if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

  // CXHXW
  if (data_format == CHW)
  {

    const unsigned int target_height = tensor_dims.at(2);
    const unsigned int target_width = tensor_dims.at(3);
    const unsigned int target_channel = tensor_dims.at(1);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;
    if (target_channel != channels) throw std::runtime_error("channel mismatch.");

    tensor_value_handler.resize(target_tensor_size);

    cv::Mat resize_mat_ref;
    if (target_height != rows || target_width != cols)
      cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
    else resize_mat_ref = mat_ref; // reference only. zero-time cost.

    std::vector<cv::Mat> mat_channels;
    cv::split(resize_mat_ref, mat_channels);
    // CXHXW
    for (unsigned int i = 0; i < channels; ++i)
      std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                  mat_channels.at(i).data,target_height * target_width * sizeof(float));

    return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                           target_tensor_size, tensor_dims.data(),
                                           tensor_dims.size());
  }

  // HXWXC
  const unsigned int target_height = tensor_dims.at(1);
  const unsigned int target_width = tensor_dims.at(2);
  const unsigned int target_channel = tensor_dims.at(3);
  const unsigned int target_tensor_size = target_channel * target_height * target_width;
  if (target_channel != channels) throw std::runtime_error("channel mismatch!");
  tensor_value_handler.resize(target_tensor_size);

  cv::Mat resize_mat_ref;
  if (target_height != rows || target_width != cols)
    cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
  else resize_mat_ref = mat_ref; // reference only. zero-time cost.

  std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

  return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                         target_tensor_size, tensor_dims.data(),
                                         tensor_dims.size());
}

cv::Mat Mixformer::normalize(const cv::Mat &mat, float mean, float scale)
{
  cv::Mat matf;
  if (mat.type() != CV_32FC3) mat.convertTo(matf, CV_32FC3);
  else matf = mat; // reference
  return (matf - mean) * scale;
}

cv::Mat Mixformer::normalize(const cv::Mat &mat, const float mean[3], const float scale[3])
{
  cv::Mat mat_copy;
  if (mat.type() != CV_32FC3) mat.convertTo(mat_copy, CV_32FC3);
  else mat_copy = mat.clone();
  for (unsigned int i = 0; i < mat_copy.rows; ++i)
  {
    cv::Vec3f *p = mat_copy.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_copy.cols; ++j)
    {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
  return mat_copy;
}

void Mixformer::normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale)
{
  outmat = this->normalize(inmat, mean, scale);
}

void Mixformer::normalize_inplace(cv::Mat &mat_inplace, float mean, float scale)
{
  if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
  this->normalize(mat_inplace, mat_inplace, mean, scale);
}

void Mixformer::normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3])
{
  if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
  for (unsigned int i = 0; i < mat_inplace.rows; ++i)
  {
    cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_inplace.cols; ++j)
    {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
}

Mixformer::Mixformer(const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
  onnx_path = _onnx_path.data();
#endif
  ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
  // 0. session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(4);
  // 1. session
  // GPU Compatibility.
#ifdef USE_CUDA
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);
  // Generate hann2d window.
  this->cfg.window = hann(this->cfg.feat_sz);
}

Mixformer::~Mixformer()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}









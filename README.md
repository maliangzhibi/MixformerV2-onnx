# MixformerV2 onnx c++ and TensorRT-py version
MixFormerV2: Efficient Fully Transformer Tracking.

[official pytorch](https://github.com/MCG-NJU/MixFormerV2.git)

Here, the mixformerv2 tracking algorithm with onnx and trt is provided, and the fps reaches about 500+fps on the 3080-laptop gpu.

At the same time, a pytrt and pyort version were also provided, which reached 430fps on the 3080-laptop gpu.
# 0. Download model
[mixformer_v2.onnx](https://www.123pan.com/s/6iArVv-FYAJ.html)

[mixformer_v2_sim.onnx](https://www.123pan.com/s/6iArVv-mcAJ.html)


# 1. How to build and run it?
Prerequisites: First, download the source code of [onnx](https://github.com/microsoft/onnxruntime) and compile it. For details, see lite.ai.toolkit. Put the header file into the onnxruntime folder and put the compiled .so file into the lib folder. The above two folders are located in Mixformerv2-onnx. However, the above steps are not required for tensorRT inference, you only need to configure TensorRT.
## modify your own CMakeList.txt
modify onnx path as yours

## build
```
$ mkdir build && cd build
$ cmake .. && make -j
```

## run
```
$ cd /home/code/Mixformerv2-onnx
$ ./mixformer-onnx [model_path] [videopath(file or camera)]
```

# 2. MixformerV2 TensorRT version inference in CPP and python
Assume that you have configured Tensorrt, use onnx2trt to convert the onnx model to engine on your GPU platform, and then start compilation and execution.

## cpp version 
build and run
```
$ cd Mixformerv2-onnx
$ mkdir build && cd build
$ cmake .. && make
& ./mixformer-trt ../model/mixformer_v2_sim.engine ../target.mp4
```
## python trt version
Modify the video path in Mixformerv2-onnx/mixformer-pytrt/mf_tracker_trt.py，and mkdir model file_dir, then download the onnx file and put onnx file into file_dir.
```
$ cd Mixformerv2-onnx
& python mixformer-pytrt/onnx2trt.py 
$ python mixformer-pytrt/mf_tracker_trt.py
```
Note: In addition to simplification when converting the onnx model, it is important to ensure that the shape of the data input to the engine model and the corresponding underlying data are continuous.

# Acknowledgments

Thanks for the [LightTrack-ncnn](https://github.com/Z-Xiong/LightTrack-ncnn.git) and [lite.ai.tookit](https://github.com/DefTruth/lite.ai.toolkit), which helps us to quickly implement our ideas.

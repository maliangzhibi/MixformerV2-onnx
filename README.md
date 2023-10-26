# MixformerV2 onnx c++ and TensorRT-py version
MixFormerV2: Efficient Fully Transformer Tracking.

[official pytorch](https://github.com/MCG-NJU/MixFormerV2.git)

Here, the c++ version of onnx mixformerv2 tracking algorithm is provided, and the fps reaches about 300fps on the 3080-laptop gpu.

At the same time, a pytrt version was also provided, which reached 430fps on the 3080-laptop gpu.
# 0. Download model
[mixformer_v2.onnx model](https://www.123pan.com/s/6iArVv-FYAJ.html)
[mixformer_v2_sim.onnx model](https://www.123pan.com/s/6iArVv-mcAJ.html)
[mixformer_v2_sim.engine model](https://www.123pan.com/s/6iArVv-ocAJ.html)
# 1. How to build and run it?

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

# 2. MixformerV2 TensorRT version inference in python
```
$ modify the video path in Mixformerv2-onnx/mixformer-pytrt/mf_tracker_trt.py
$ cd Mixformerv2-onnx
$ python mixformer-pytrt/mf_tracker_trt.py
```
Note: In addition to simplification when converting the onnx model, it is important to ensure that the shape of the data input to the engine model and the corresponding underlying data are continuous.

# Acknowledgments

Thanks for the [LightTrack-ncnn](https://github.com/Z-Xiong/LightTrack-ncnn.git) and [lite.ai.tookit](https://github.com/DefTruth/lite.ai.toolkit), which helps us to quickly implement our ideas.

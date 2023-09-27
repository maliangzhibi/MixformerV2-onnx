# MixformerV2的onnx c++版

# 0. Download model

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
$ cd build
$ ./Mixformer-onnx [videopath(file or camera)]
```

# Acknowledgments

Thanks for the [LightTrack-ncnn](https://github.com/Z-Xiong/LightTrack-ncnn.git) and [lite.ai.tookit](https://github.com/DefTruth/lite.ai.toolkit), which helps us to quickly implement our ideas.

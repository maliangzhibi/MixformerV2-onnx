import os
import cv2
import torch
import torch.utils.data
import random
import argparse
import numpy as np
import time
import torch.onnx
import onnx
import tensorrt as trt




def parse_args():
    parser = argparse.ArgumentParser(description='Test LightTrack')
    parser.add_argument('--arch', default='LightTrackM_Subnet', dest='arch', help='backbone architecture')
    parser.add_argument('--resume', default='../snapshot/ETTrack/checkpoint_e35.pth', type=str, help='pretrained model')
    parser.add_argument('--dataset', default='OTB100_pysot', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--video', type=str, help='test a video in benchmark')
    parser.add_argument('--stride', default=16, type=int, help='network stride')
    parser.add_argument('--even', type=int, default=1, help='1 corresponding to 256 ,0 corresponding to 255') 
    parser.add_argument('--path_name', type=str, default='NULL')
    args = parser.parse_args()

    return args


def get_data(bs, sz):
    image = torch.randn(bs, 3, sz, sz).cuda()
    return image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_engine(f_onnx, workspace=4, verbose=False, prefix="TensorRT", dynamic=False):
    """使用onnx_parser解析onnx模型, 然后编译得到engine进行推理

    Args:
        f_onnx (_type_): _description_
    """
    from pathlib import Path
    assert Path(f_onnx).exists(), f'NOt found ONNX file: {f_onnx}' 
    model = onnx.load(f_onnx)
    onnx.checker.check_model(model)

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_serverity = trt.Logger.Serverity.VERBOSE
    trt.init_libnvinfer_plugins(logger, namespace='')

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30
    # if bUseINT8Mode:
    #     config.int8_mode = bUseINT8Mode

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(f_onnx):
        raise RuntimeError(f'failed to load ONNX file: {f_onnx}')
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
        
    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)

    f = "model/mixformer_v2.engine"
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
    # with builder.build_serialized_network(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())

    return f, None
    
def main():
    export_engine(f_onnx='model/mixformer_v2.onnx')

if __name__ == '__main__':
    main()

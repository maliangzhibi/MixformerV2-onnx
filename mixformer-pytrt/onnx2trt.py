import contextlib
import torch
import torch.utils.data
import torch.onnx
import onnx
import tensorrt as trt
import inspect
import logging
import logging.config as log_config
import os
import platform
import time
from pathlib import Path


LOGGING_NAME = "mftrack"


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log_config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level,}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,}}})


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def get_data(bs, sz):
    image = torch.randn(bs, 3, sz, sz).cuda()
    return image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_engine(f_onnx, half=False, workspace=4, verbose=False, prefix="TensorRT"):
    """使用onnx_parser解析onnx模型, 编译得到engine进行推理

    Args:
        f_onnx (str): ONNX文件的路径
        half (bool, optional): 是否使用FP16模式. 默认为False.
        workspace (int, optional): 最大工作空间大小(GB). 默认为4.
        verbose (bool, optional): 是否启用详细日志. 默认为False.
        prefix (str, optional): 日志前缀. 默认为"TensorRT".
    """
    f = "model/mixformer_v2_sim.engine"

    assert Path(f_onnx).exists(), f'NOt found ONNX file: {f_onnx}' 
    model = onnx.load(f_onnx)
    onnx.checker.check_model(model)

    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_serverity = trt.Logger.Serverity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    
    print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        print(f"转换为FP16模型.")
        config.set_flag(trt.BuilderFlag.FP16)

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(f_onnx)):
        raise RuntimeError(f'failed to load ONNX file: {f_onnx}')
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
        
    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)
     
    engine = builder.build_serialized_network(network, config)
    with open(f, 'wb') as t:
        t.write(engine)
    
def main():
    export_engine(f_onnx=Path('model/mixformer_v2_sim.onnx'), half=False, verbose=False)
    # export_engine(f_onnx=Path('model/mixformer_v2_sim_fp16.onnx'), half=True, verbose=False)

if __name__ == '__main__':
    main()

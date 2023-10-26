import tensorrt as trt
from collections import OrderedDict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import logging
import logging.config
import time
import pkg_resources as pkg


TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

# # 检查注册的操作
# def get_plugin_names():
#     return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list] 
# print('检查注册算子>> ', get_plugin_names())

LOGGING_NAME="MixformerNvinfer"
LOGGER = logging.getLogger(LOGGING_NAME)

ENGINE_TYPE=['mixformer_v2', 'mixformer_v2_int32', 'mixformer_v2_sim']


class MixformerNvinfer:
    """Mixformer Nvinfer
    """
    def __init__(self, engine_name="mixformer_v2_sim") -> None:

        # 检查输入的engine_type
        assert engine_name in ENGINE_TYPE, "please check the engine_type whether is in ENGINE_TYPE=['bacbone_neck_x', \
            'backbone_neck_z', 'featfusor_head', 'mixformer_v2', 'mixformer_v2_int32', 'mixformer_v2_sim']."

        self.device = torch.device('cuda:0')
        
        # 根据输入engine类型得到对应类型的模型
        self.engine_path = os.path.join("model", engine_name + '.engine')
        if not os.path.exists(self.engine_path):
            LOGGER.info(f"Error ENGINE_NAME: {engine_name}")
            sys.exit(1)
        
        LOGGER.info(f"loading {self.engine_path} for TensorRT inference.")
        self.check_version(trt.__version__, '7.0.0', hard=True)

        # 定义绑定数据
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

        # 定义logger
        self.logger = trt.Logger(trt.Logger.INFO)

        # 反序列化engine文件
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        # runtime = trt.Runtime(self.logger)
        # with open(self.engine_path, "rb") as f:
        #     serialized_engine = f.read()
        # self.model = runtime.deserialize_cuda_engine(serialized_engine)
        
        # 创建上下文
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        # self.fp16 = False
        # self.dynamic = False
        # binding_names = [self.model.get_binding_name(i) for i in range(self.model.num_bindings)]
        # print(f">>>model numbindings: {self.model.num_bindings} {binding_names}")
        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            LOGGER.info(f"name: {name}")
            # input
            if self.model.binding_is_input(i): 
                # print(f">>>input name: {name} {self.model.get_binding_shape(i)} {dtype}")
                if -1 in tuple(self.model.get_binding_shape(i)):
                    dynamic = True
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0,1)[2]))
                if dtype == np.float16:
                    fp16 = True
            else: # output
                # print(f">>>output name: {name} {self.model.get_binding_shape(i)} {dtype}")
                self.output_names.append(name)

            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype)).to(self.device)

            # 绑定输入输出数据
            self.bindings[name] = self.Binding(name, dtype, shape, im, int(im.data_ptr()))

        # LOGGER.info(f"input and output's name and addr: {OrderedDict((n, d.ptr) for n, d in self.bindings.items())}")
        # 记录input_0，output_0, output_1的名称和地址
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['img_search'].shape[0]

    def infer(self, im, im_0, im_1):
        """输入图像进行trt推理

        Args:
            im (_type_): _description_
            augment (bool, optional): _description_. Defaults to False.
            visualize (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # print(f">>> binding_addrs keys: {self.binding_addrs.values()}") # ==
        # 将实际输入的图像地址取出，并赋值给binding_addr
        self.binding_addrs['img_t'] = int(im.data_ptr())
        self.binding_addrs['img_ot'] = int(im_0.data_ptr())
        self.binding_addrs['img_search'] = int(im_1.data_ptr())
        # print(f">>> binding_addrs keys1: {self.binding_addrs.values()}") # ==
        # 将绑定的地址地址传递给context，使用execute_v2进行推理，推理结果就会保存在对应的地址中，通过访问对应地址的数据就能得到输出
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
        
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


    def check_version(self, current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
        # Check version vs. required version
        current, minimum = (pkg.parse_version(x) for x in (current, minimum))
        result = (current == minimum) if pinned else (current >= minimum)  # bool
        s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
        if hard:
            assert result, s  # assert min requirements met
        if verbose and not result:
            LOGGER.warning(s)
        return result

# class MixformerNvinfer:
#     def __init__(self, engine_path=None) -> None:
#         assert engine_path is not None, "engine path is None"

#         # 1. 创建TensorRT的runtime
#         TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#         runtime = trt.Runtime(TRT_LOGGER)

#         # 2. 从文件中加载engine
#         with open(engine_path, "rb") as f:
#             engine_data = f.read()
#         self.engine = runtime.deserialize_cuda_engine(engine_data)
        
#         # 3. 创建执行上下文并进行推理
#         self.context = self.engine.create_execution_context()

#     def infer(self, img_t, img_ot, img_search):



if __name__=="__main__":
    det = MixformerNvinfer(ENGINE_TYPE[2])

    input= torch.rand((1, 3, 112, 112)).cuda()
    input0= torch.rand((1, 3, 112, 112)).cuda()
    input1= torch.rand((1, 3, 224, 224)).cuda()

    warmup_N = 100
    N = 1000
    for i in range(warmup_N):
        output = det.infer(input, input0, input1)
    
    start = time.time()
    for i in range(N):
        start_i = time.time()
        output = det.infer(input, input0, input1)
        # print(f">>>single infer time: {1 / (time.time() - start_i)} FPS")

    print(f">>>infer time: {1 / ((time.time() - start) / N)} FPS")

    print(f"output's length is {output[0].shape} {output[1].shape}")
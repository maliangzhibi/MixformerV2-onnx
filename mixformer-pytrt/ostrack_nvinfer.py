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


TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

# 检查注册的操作
def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list] 
print('检查注册算子>> ', get_plugin_names())

LOGGING_NAME="OSTrackNvinfer"
LOGGER = logging.getLogger(LOGGING_NAME)

ENGINE_TYPE=['ostrack']


class OSTrackNvinfer:
    """该类进行ET.Tracker
    """
    def __init__(self, engine_type="ostrack", engine_name="ostrack-256-ep300") -> None:

        # 检查输入的engine_type
        assert engine_type in ENGINE_TYPE, "please check the engine_type whether is in ENGINE_TYPE=['bacbone_neck_x', 'backbone_neck_z', 'featfusor_head']."

        self.device = 0
        
        # 根据输入engine类型得到对应类型的模型
        if engine_type == ENGINE_TYPE[0]:
            self.engine_path = os.path.join("model", engine_name + '.engine')
        else:
            LOGGER.info(f"Error ENGINE_TYPE: {engine_type}")
            sys.exit(1)

        LOGGER.info(f"loading {self.engine_path} for TensorRT inference.")
        
        # 定义绑定数据
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

        # 定义logger
        self.logger = trt.Logger(trt.Logger.INFO)

        # 反序列化engine文件
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            # print(">>> ",   f.read())
            self.model = runtime.deserialize_cuda_engine(f.read())
        
        # 创建上下文
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False

        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            LOGGER.info(name)
            # input
            if self.model.binding_is_input(i): 
                if -1 in tuple(self.model.get_binding_shape(i)):
                    dynamic = True
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0,1)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:# output
                self.output_names.append(name)

            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype)).to(self.device)

            # 绑定输入输出数据
            self.bindings[name] = self.Binding(name, dtype, shape, im, int(im.data_ptr()))

        LOGGER.info(f"input and output's name and addr: {OrderedDict((n, d.ptr) for n, d in self.bindings.items())}")
        # 记录input_0，output_0, output_1的名称和地址
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['x'].shape[0]

    def infer(self, im, im_1):
        """输入图像进行trt推理

        Args:
            im (_type_): _description_
            augment (bool, optional): _description_. Defaults to False.
            visualize (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # 通过bindings获取绑定的输入形状
        if self.dynamic and im.shape != self.bindings['z'].shape and im_1.shape != self.bindings['x'].shape:
            i = self.model.get_binding_index('z')
            j = self.model.get_binding_index('x')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.context.set_binding_shape(j, im_1.shape)  # reshape if dynamic

            self.bindings['z'] = self.bindings['x']._replace(shape=im.shape)
            self.bindings['x'] = self.bindings['x']._replace(shape=im_1.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

        s = self.bindings['z'].shape
        s1 = self.bindings['x'].shape

        # 判断实际输入的形状是否符合预先绑定的形状
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        assert im_1.shape == s1, f"input size {im_1.shape} {'>' if self.dynamic else 'not equal to'} max model size {s1}"

        # 将实际输入的图像地址取出，并赋值给binding_addr
        self.binding_addrs['z'] = int(im.data_ptr())
        self.binding_addrs['x'] = int(im_1.data_ptr())

        # 将绑定的地址地址传递给context，使用execute_v2进行推理，推理结果就会保存在对应的地址中，通过访问对应地址的数据就能得到输出
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
        
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


if __name__=="__main__":
    det = OSTrackNvinfer(ENGINE_TYPE[0])

    input= torch.rand((1, 3, 128, 128)).cuda()
    input1= torch.rand((1, 3, 256, 256)).cuda()

    start = time.time()
    N = 1000
    for i in range(N):
        start_i = time.time()
        output = det.infer(input, input1)
        print(f">>>single infer time: {1 / (time.time() - start_i)} FPS")

    print(f">>>infer time: {1 / ((time.time() - start) / N)} FPS")

    LOGGER.info(f"output's length is {len(output)} {output[0].shape} {output[1].shape} {output[0].device}")
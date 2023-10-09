import tensorrt as trt
from pathlib import Path
import struct
import numpy as np
import pycuda

verbose = True
IN_NAME1 = 'img_t'
IN_NAME2 = 'img_ot'
IN_NAME3 = 'img_search'
OUT_NAME1 = 'pred_boxes'
OUT_NAME2 = 'pred_scores'
IN_H1 = 112
IN_H2 = 112
IN_H3 = 224
IN_W1 = 112
IN_W2 = 112
IN_W3 = 224
BATCH_SIZE = 1

def read_wts(filename):
    # 读取权重文件
    weights = {}
    with open(filename, 'r') as f:
        # 读取权重数量
        num_weights = int(f.readline().strip())
        print(f'>>> num_weights: {num_weights}')
        for i in range(num_weights):
            # if i == 0:
            line = f.readline().strip()
            # print(f'>>> line: {line}')
            parts = line.split(' ')

            # 获取权重名称、大小
            name = parts[0]
            size = parts[1]
            # print(f">>>name and size: {name} {size}  {len(parts[2:])}")

            # 获取权重值
            values = [struct.unpack('!f', bytes.fromhex(x))[0] for x in parts[2:2+int(size)]]
            weights[name] = values
    print(f">>>wts name: {weights.keys()}")
    return weights

def reshape(network, input_tensor, new_shape):
    shuffle_layer = network.add_shuffle(input_tensor)
    shuffle_layer.reshape_dims = new_shape

    return shuffle_layer

def transpose(network, input_tensor, perm):
    # 创建一个shuffle层来实现transpose操作
    shuffle_layer = network.add_shuffle(input_tensor)

    # 假设你想将一个形状为(3, 32, 32)的tensor置换为(32, 32, 3)
    # 原始的顺序是[0, 1, 2]，新的顺序是[1, 2, 0]
    shuffle_layer.first_transpose = trt.Permutation(perm)
    return shuffle_layer

def layer_norm(network, weight, bias, input_tensor, exp):
    # 先进性标准化
    reduce_axes = 4 << 0  # 选择第一个维度
    reduce_layer = network.add_reduce(input_tensor, trt.ReduceOperation.AVG, reduce_axes, True)
    sub_tensor = network.add_elementwise(input_tensor, reduce_layer.get_output(0), trt.ElementWiseOperation.SUB)
    # 定义常数2.0
    constant_shape = input_tensor.shape
    constant_value = np.full(constant_shape, 2.0, dtype=np.float32)
    constant_weights = trt.Weights(constant_value.ravel())
    constant_layer = network.add_constant(constant_shape, constant_weights)
    # pow(sub_tensor, 2.0)
    pow_tensor = network.add_elementwise(sub_tensor.get_output(0), constant_layer.get_output(0), trt.ElementWiseOperation.POW)
    # reduce mean
    reduce_layer = network.add_reduce(pow_tensor.get_output(0), trt.ReduceOperation.AVG, reduce_axes, True)
    # add
    constant_shape = reduce_layer.get_output(0).shape
    constant_value = np.full(constant_shape, 9.999999974752427e-7, dtype=np.float32)
    constant_weights = trt.Weights(constant_value.ravel())
    constant_layer = network.add_constant(constant_shape, constant_weights)
    add_tensor = network.add_elementwise(reduce_layer.get_output(0), constant_layer.get_output(0), trt.ElementWiseOperation.SUM)    
    # sqrt
    sqrt_layer = network.add_unary(add_tensor.get_output(0), trt.UnaryOperation.SQRT)
    # div
    div_tensor = network.add_elementwise(sub_tensor.get_output(0), sqrt_layer.get_output(0), trt.ElementWiseOperation.DIV)
    # mul(div_tensor, norm1.weight)
    weight = np.array(weight).astype(np.float32)
    weight_shape = weight.shape
    weight = trt.Weights(weight.ravel())
    weight_constant = network.add_constant(weight_shape, weight)
    # 需要reshape成和div_tensor一样的形状
    weight_constant = reshape(network=network, input_tensor=weight_constant.get_output(0), new_shape=trt.Dims3(1, 1, weight_shape[0]))
    mul_tensor = network.add_elementwise(div_tensor.get_output(0), weight_constant.get_output(0), trt.ElementWiseOperation.PROD)
    # add
    bias = np.array(bias).astype(np.float32)
    bias_shape = bias.shape
    bias = trt.Weights(bias.ravel())
    bias_constant = network.add_constant(bias_shape, bias)
    # # 需要reshape成和div_tensor一样的形状
    bias_constant = reshape(network=network, input_tensor=bias_constant.get_output(0), new_shape=trt.Dims3(1, 1, bias_shape[0]))
    add_tensor = network.add_elementwise(mul_tensor.get_output(0), bias_constant.get_output(0), trt.ElementWiseOperation.SUM)
    
    return add_tensor

def matmul(network, input_tensor, weight):
    """实现矩阵乘的操作

    Args:
        network (_type_): _description_
        input_tensor (_type_): _description_
        weight (_type_): _description_
        bias (_type_): _description_
    """
    mm_layer = network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.NONE)
    return mm_layer


def mul(network, input_tensor, weight, weight_shape):
    """实现input_tensor与weight的乘法操作

    Args:
        network (_type_): _description_
        input_tensor (_type_): _description_
        weight (_type_): _description_
    """
    # mul(input_tensor, weight)
    weight = np.array(weight).astype(np.float32)
    weight = trt.Weights(weight.ravel())
    weight_constant = network.add_constant(weight_shape, weight)
    # input_tensor
    weight_constant = reshape(network=network, input_tensor=weight_constant.get_output(0), new_shape=weight_shape)
    mul_layer = network.add_elementwise(input_tensor, weight_constant.get_output(0), trt.ElementWiseOperation.PROD)
    return mul_layer


def attention(network, qkv_weight, qkv_bias, proj_weight, proj_bias, input_tensor):
    """对layernorm后的数据进行attention

    Args:
        network (_type_): _description_
        dim (_type_): _description_
        num_head (_type_): _description_
        attn_drop (_type_): _description_
        proj_drop (_type_): _description_
        qkv_weight (_type_): _description_
        qkv_bias (_type_): _description_
        proj_weight (_type_): _description_
        proj_bias (_type_): _description_
    """
    # print(f">>> input_tensor shape: {input_tensor.shape}")
    B, N, C = input_tensor.shape
    reshape_tensor_1 = reshape(network=network, input_tensor=input_tensor, new_shape=trt.Dims2(N, C))

    # linear 1
    # matmul操作
    qkv_weight = np.ascontiguousarray(np.array(qkv_weight).reshape(-1, 768).transpose(1, 0)).astype(np.float32)
    qkv_bias = np.array(qkv_bias).astype(np.float32)
    trt_weights_1 = trt.Weights(qkv_weight)  # 注意转置，因为TRT期望的权重布局与PyTorch不同
    trt_biases_1 = trt.Weights(qkv_bias)
    weight_tensor_1 = network.add_constant(shape=(768, 2304), weights=trt_weights_1)
    bias_tensor_1 = network.add_constant(shape=(1, 2304), weights=trt_biases_1)
    mm_layer_1 = network.add_matrix_multiply(reshape_tensor_1.get_output(0), trt.MatrixOperation.NONE, weight_tensor_1.get_output(0), trt.MatrixOperation.NONE)
    # add
    add_layer_1 = network.add_elementwise(mm_layer_1.get_output(0), bias_tensor_1.get_output(0), trt.ElementWiseOperation.SUM)
    
    # reshape
    fc_layer_1_shape = trt.Dims([B, N, 3, 12, C//12])
    fc_layer_1 = reshape(network=network, input_tensor=add_layer_1.get_output(0), new_shape=fc_layer_1_shape)
    transpose_1 = transpose(network=network, input_tensor=fc_layer_1.get_output(0), perm=[2, 0, 3, 1, 4])

    # unbind(0)
    q, k, v = split(network=network, input_tensor=transpose_1.get_output(0), axis=0)
    # squeeze
    q = reshape(network=network, input_tensor=q, new_shape=trt.Dims(q.shape[1:]))
    k = reshape(network=network, input_tensor=k, new_shape=trt.Dims(k.shape[1:]))
    v = reshape(network=network, input_tensor=v, new_shape=trt.Dims(v.shape[1:]))

    # 3 * split
    q_98_split, q_200_split = split2(network=network, input_tensor=q.get_output(0))
    k_98_split, k_200_split = split2(network=network, input_tensor=k.get_output(0))
    v_98_split, v_200_split = split2(network=network, input_tensor=v.get_output(0))

    k_98_split = transpose(network=network, input_tensor=k_98_split.get_output(0), perm=[0, 1, 3, 2])
    q98_k98_mm_layer = matmul(network=network, input_tensor=q_98_split.get_output(0), weight=k_98_split.get_output(0))
    qk98_mul_layer = mul(network=network, input_tensor=q98_k98_mm_layer.get_output(0), weight=0.125, weight_shape=trt.Dims([1, 1, 1, 1]))

    # softmax1
    qk98_softmax_layer = network.add_softmax(qk98_mul_layer.get_output(0))

    # k_transpose
    k_transpose = transpose(network=network, input_tensor=k.get_output(0), perm=[0, 1, 3, 2])
    q200ktranspose_mm_layer = matmul(network=network, input_tensor=q_200_split.get_output(0), weight=k_transpose.get_output(0))
    q200ktranspose_mul_layer = mul(network=network, input_tensor=q200ktranspose_mm_layer.get_output(0),
                                      weight=0.125, weight_shape=trt.Dims([1, 1, 1, 1]))
    # softmax2
    q200ktranspose_softmax_layer = network.add_softmax(q200ktranspose_mul_layer.get_output(0))

    # matmul(qk98softmax, v_98_split) transpose reshape
    qk98softmax_v98split_mm_layer = matmul(network=network, input_tensor=qk98_softmax_layer.get_output(0),
                                         weight=v_98_split.get_output(0))
    qk98v98_transpose_layer = transpose(network=network, input_tensor=qk98softmax_v98split_mm_layer.get_output(0), perm=[0, 2, 1, 3])
    dim = qk98v98_transpose_layer.get_output(0).shape
    qk98v98_reshape = reshape(network=network, input_tensor=qk98v98_transpose_layer.get_output(0), 
                            new_shape=trt.Dims([dim[0], dim[1], dim[2]*dim[3]]))
    
    # matmul(q200k98transpose_softmax, v)  transpose reshape
    q200_ktranspose_v_mm_layer = matmul(network=network, input_tensor=q200ktranspose_softmax_layer.get_output(0),
                                         weight=v.get_output(0))
    q200kv_transpose_layer = transpose(network=network, input_tensor=q200_ktranspose_v_mm_layer.get_output(0), perm=[0, 2, 1, 3])
    dim = q200kv_transpose_layer.get_output(0).shape
    q200kv_reshape = reshape(network=network, input_tensor=q200kv_transpose_layer.get_output(0), 
                             new_shape=trt.Dims([dim[0], dim[1], dim[2]*dim[3]]))
    
    # concat the qk98v98_reshape and q200kv_reshape
    qk98v98_q200kv_concat_layer = network.add_concatenation([qk98v98_reshape.get_output(0), q200kv_reshape.get_output(0)])
    qk98v98_q200kv_concat_layer.axis = 1

    # linear 2
    # matmul操作
    proj_weight = np.ascontiguousarray(np.array(proj_weight).reshape(-1, 768).transpose(1, 0)).astype(np.float32)
    proj_bias = np.array(proj_bias).astype(np.float32)
    trt_weights_2 = trt.Weights(proj_weight)  # 注意转置，因为TRT期望的权重布局与PyTorch不同
    trt_biases_2 = trt.Weights(proj_bias)
    weight_tensor_2 = network.add_constant(shape=(1, 768, 768), weights=trt_weights_2)
    bias_tensor_2 = network.add_constant(shape=(1, 1, 768), weights=trt_biases_2)

    mm_layer_2 = network.add_matrix_multiply(qk98v98_q200kv_concat_layer.get_output(0), trt.MatrixOperation.NONE, 
                                             weight_tensor_2.get_output(0), trt.MatrixOperation.NONE)
    # add
    add_layer_2 = network.add_elementwise(mm_layer_2.get_output(0), bias_tensor_2.get_output(0), trt.ElementWiseOperation.SUM)

    # print(f">>> fc shape: {mm_layer_2.get_output(0).shape} {add_layer_2.get_output(0).shape}")
    return add_layer_2
    

def split(network, input_tensor, axis=0):
    """将张量沿着指定的维度拆分成若干个张量,并返回一个元组(tuple)包含这些张量

    Args:
        input_tensor (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.
    """
    dim = input_tensor.shape
    depth = input_tensor.shape[axis]

    # 为每个depth切片创建一个slice layer
    slices = []
    # start = ()
    for i in range(depth):
        slice_layer = network.add_slice(input_tensor, start=(i, 0, 0, 0, 0), shape=(1, dim[1], dim[2], dim[3], dim[4]), stride=(1, 1, 1, 1, 1))
        slices.append(slice_layer.get_output(0))

    return slices

def split2(network, input_tensor):
    """将张量两个张量分别为(1, 12, 98, 64)和(1, 12, 98, 200),并返回对应的层。

    Args:
        input_tensor (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.
    """

    # 第一部分：从(1, 12, 0, 64)到(1, 12, 98, 64)
    first_slice_layer = network.add_slice(input_tensor, start=(0, 0, 0, 0), shape=(1, 12, 98, 64), stride=(1, 1, 1, 1))
    # 第二部分：从(1, 12, 98, 64)到(1, 12, 200, 64)
    second_slice_layer = network.add_slice(input_tensor, start=(0, 0, 98, 0), shape=(1, 12, 200, 64), stride=(1, 1, 1, 1))

    return first_slice_layer, second_slice_layer


def drop_path(network, input_tensor, drop_prob, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)

    Args:
        network (_type_): _description_
        input_tensor (_type_): _description_
        drop_prob (_type_): _description_
    """
    if drop_prob > 0. or training:
        return input_tensor
    
    # todo 可能的其他操作，未完成，该模型中不需要

def mlp(network, input_tensor, weight, bias):
    """_summary_

    Args:
        network (_type_): _description_
        input_tensor (_type_): _description_
        weight (_type_): _description_
        bias (_type_): _description_
    """
    # linear 1
    # matmul操作
    # B, N, C = input_tensor.sahpe
    # print(f">>>BNC: {B} {N} {C}")
    weight = np.ascontiguousarray(np.array(weight).reshape(-1, 768).transpose(1, 0)).astype(np.float32)
    weight_dim = weight.shape
    bias = np.array(bias).astype(np.float32)
    bias_dim = bias.shape
    trt_weights = trt.Weights(weight)  # 注意转置，因为TRT期望的权重布局与PyTorch不同
    trt_biases = trt.Weights(bias)
    weight_tensor = network.add_constant(shape=(1, weight_dim[0], weight_dim[1]), weights=trt_weights)
    bias_tensor = network.add_constant(shape=(1, 1, bias_dim[0]), weights=trt_biases)
    
    mm_layer = network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE, weight_tensor.get_output(0), trt.MatrixOperation.NONE)
    # add
    add_layer = network.add_elementwise(mm_layer.get_output(0), bias_tensor.get_output(0), trt.ElementWiseOperation.SUM)

    return add_layer
 
def gelu(network, input_tensor):
    """gelu激活函数

    Args:
         network (_type_): _description_
         input_tensor (_type_): _description_
    """
    # 假设你已经有一个输入ITensor named input_tensor
    # input_tensor = ...

    # 定义GELU所需的常数
    const_half = network.add_constant((1, 1, 1), trt.Weights(np.array([0.5], dtype=np.float32))).get_output(0)
    const_sqrt_2_over_pi = network.add_constant((1, 1, 1), trt.Weights(np.array([np.sqrt(2.0 / np.pi)], dtype=np.float32))).get_output(0)
    # const_0_044715 = network.add_constant((1, 1, 1), trt.Weights(np.array([0.044715], dtype=np.float32))).get_output(0)

    # 计算0.044715 * x^3
    x3 = network.add_elementwise(input_tensor, input_tensor, trt.ElementWiseOperation.PROD).get_output(0)
    x3 = network.add_elementwise(input_tensor, x3, trt.ElementWiseOperation.PROD).get_output(0)
    scaled_x3 = network.add_scale(x3, mode=trt.ScaleMode.UNIFORM, shift=np.array([0.], dtype=np.float32), scale=np.array([0.044715], dtype=np.float32)).get_output(0)
    
    # 计算sqrt(2/pi) * (x + 0.044715 * x^3)
    x_plus_scaled_x3 = network.add_elementwise(input_tensor, scaled_x3, trt.ElementWiseOperation.SUM).get_output(0)
    scaled_result = network.add_elementwise(x_plus_scaled_x3, const_sqrt_2_over_pi, trt.ElementWiseOperation.PROD).get_output(0)
    
    # 计算tanh(...)
    tanh_result = network.add_activation(scaled_result, trt.ActivationType.TANH).get_output(0)
    
    # 计算0.5 * x * (1 + tanh(...))
    one_plus_tanh = network.add_elementwise(tanh_result, const_half, trt.ElementWiseOperation.SUM).get_output(0)
    gelu_result_layer = network.add_elementwise(input_tensor, one_plus_tanh, trt.ElementWiseOperation.PROD)

    return gelu_result_layer


# def block_n(network, weights, block_num, input_x_tensor, H_t, W_t, H_s, W_s):
def block_n(network, weights, block_num, input_x_tensor):
    """计算一个block

    Args:
        network (_type_): _description_
        weights (_type_): _description_
        block_num (_type_): _description_
        input_x_tensor (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 参数设定
    norm1_weight = weights[f"blocks.{block_num}.norm1.weight"]
    norm1_bias = weights[f"blocks.{block_num}.norm1.bias"]
    
    attn_qkv_weight = weights[f"blocks.{block_num}.attn.qkv.weight"]
    attn_qkv_bias = weights[f"blocks.{block_num}.attn.qkv.bias"]

    attn_proj_weight = weights[f"blocks.{block_num}.attn.proj.weight"]
    attn_proj_bias = weights[f"blocks.{block_num}.attn.proj.bias"]

    norm2_weight = weights[f"blocks.{block_num}.norm2.weight"]
    norm2_bias = weights[f"blocks.{block_num}.norm2.bias"]

    mlp_fc1_weight = weights[f"blocks.{block_num}.mlp.fc1.weight"]
    mlp_fc1_bias = weights[f"blocks.{block_num}.mlp.fc1.bias"]

    mlp_fc2_weight = weights[f"blocks.{block_num}.mlp.fc2.weight"]
    mlp_fc2_bias = weights[f"blocks.{block_num}.mlp.fc2.bias"]
    # print(f">>> block param: {np.array(attn_qkv_weight).reshape(-1, 768).transpose(1, 0).shape} {np.array(attn_qkv_weight).shape}")
    
    # 网络搭建 
    norm1_layer = layer_norm(network=network, weight=norm1_weight, 
                                   bias=norm1_bias, input_tensor=input_x_tensor, 
                                   exp=2)
    attention_layer = attention(network=network, qkv_weight=attn_qkv_weight,
                                qkv_bias=attn_qkv_bias, proj_weight=attn_proj_weight,
                                proj_bias=attn_proj_bias, input_tensor=norm1_layer.get_output(0))
    
    inputx_attn_add_layer = network.add_elementwise(input_x_tensor, attention_layer.get_output(0), trt.ElementWiseOperation.SUM)
    
    norm2_layer = layer_norm(network=network, weight=norm2_weight, 
                                   bias=norm2_bias, input_tensor=inputx_attn_add_layer.get_output(0), 
                                   exp=2)
    # 计算MLP，其由linear+gelu+linear组成
    mlp_layer1 = mlp(network=network, input_tensor=norm2_layer.get_output(0), weight=mlp_fc1_weight, bias=mlp_fc1_bias)
    gelu_layer = gelu(network=network, input_tensor=mlp_layer1.get_output(0))
    mlp_layer2 = mlp(network=network, input_tensor=gelu_layer.get_output(0), weight=mlp_fc2_weight, bias=mlp_fc2_bias)

    # x + mlp
    x_mlp_add_layer = network.add_elementwise(inputx_attn_add_layer.get_output(0), mlp_layer2.get_output(0), trt.ElementWiseOperation.SUM)
    # print(f">>> mlp_layer2: {x_mlp_add_layer.get_output(0).shape}")
    return x_mlp_add_layer

def construct_network():
   # 读取权重
    wts_path = Path("model/mixformerv2.wts")
    weights = read_wts(wts_path)

    EXPLICIT_BATCH = 1 << (int)( 
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger() 
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config( 
    ) as config, builder.create_network(EXPLICIT_BATCH) as network: 
        # define input tensor 
        input_img_t = network.add_input( 
            name=IN_NAME1, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H1, IN_W1))
        input_img_ot = network.add_input( 
            name=IN_NAME2, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H2, IN_W2))
        input_img_search = network.add_input( 
            name=IN_NAME3, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H3, IN_W3))
        # print(f">>>input tensor: {type(input_img_ot)}")
        # define backbone
        # first layerDims2
        conv1_w = trt.Weights(np.array(weights['patch_embed.proj.weight']).astype(np.float32).reshape(768, 3, 16, 16))
        conv1_b = trt.Weights(np.array(weights['patch_embed.proj.bias']).astype(np.float32))
        
        conv1 = network.add_convolution_nd(input=input_img_search, num_output_maps=768, kernel_shape=trt.Dims2(16, 16), kernel=conv1_w, bias=conv1_b)
        conv1.stride_nd = trt.Dims2(16, 16)
        conv1.padding_nd = trt.Dims2(0, 0)
        conv1.dilation_nd = trt.Dims2(1, 1)
        conv1.num_groups = 1

        conv2 = network.add_convolution_nd(input=input_img_t, num_output_maps=768, kernel_shape=(16, 16), kernel=conv1_w, bias=conv1_b)
        conv2.stride_nd = trt.Dims2(16, 16)
        conv2.padding_nd = trt.Dims2(0, 0)
        conv2.dilation_nd = trt.Dims2(1, 1)
        conv2.num_groups = 1

        conv3 = network.add_convolution_nd(input=input_img_ot, num_output_maps=768, kernel_shape=(16, 16), kernel=conv1_w, bias=conv1_b)
        conv3.stride_nd = trt.Dims2(16, 16)
        conv3.padding_nd = trt.Dims2(0, 0)
        conv3.dilation_nd = trt.Dims2(1, 1)
        conv3.num_groups = 1

        # reshape1
        conv1_shape = trt.Dims3(1, 768, 196)
        conv2_shape = trt.Dims3(1, 768, 49)
        conv3_shape = trt.Dims3(1, 768, 49)
        reshape_1 = reshape(network=network, input_tensor=conv1.get_output(0), new_shape=conv1_shape)
        reshape_2 = reshape(network=network, input_tensor=conv2.get_output(0), new_shape=conv2_shape)
        reshape_3 = reshape(network=network, input_tensor=conv3.get_output(0), new_shape=conv3_shape)

        # transpose1
        perm = [0, 2, 1]
        transpose_1 = transpose(network=network, input_tensor=reshape_1.get_output(0), perm=perm)
        transpose_2 = transpose(network=network, input_tensor=reshape_2.get_output(0), perm=perm)
        transpose_3 = transpose(network=network, input_tensor=reshape_3.get_output(0), perm=perm)

        # add1 add2 add3
        pos_embed_s_input_shape = (1, 196, 768)
        pos_embed_t_input_shape = (1, 49, 768)
        
        pos_embed_s = trt.Weights(np.array(weights['pos_embed_s']).reshape(1, 196, 768).astype(np.float32))
        pos_embed_t = trt.Weights(np.array(weights['pos_embed_t']).reshape(1, 49, 768).astype(np.float32))

        pos_embed_s_constant = network.add_constant(pos_embed_s_input_shape, pos_embed_s)
        pos_embed_t_constant = network.add_constant(pos_embed_t_input_shape, pos_embed_t)

        add_layer_1 = network.add_elementwise(transpose_1.get_output(0), pos_embed_s_constant.get_output(0), trt.ElementWiseOperation.SUM)
        add_layer_2 = network.add_elementwise(transpose_2.get_output(0), pos_embed_t_constant.get_output(0), trt.ElementWiseOperation.SUM)
        add_layer_3 = network.add_elementwise(transpose_3.get_output(0), pos_embed_t_constant.get_output(0), trt.ElementWiseOperation.SUM)

        reg_tokens_input_shape = (1, 4, 768)
        reg_tokens =  trt.Weights(np.array(weights['reg_tokens']).reshape(1, 4, 768).astype(np.float32))
        reg_tokens_constant = network.add_constant(reg_tokens_input_shape, reg_tokens)

        pos_embed_reg_input_shape = (1, 4, 768)
        pos_embed_reg =  trt.Weights(np.array(weights['reg_tokens']).reshape(1, 4, 768).astype(np.float32))
        pos_embed_reg_constant = network.add_constant(pos_embed_reg_input_shape, pos_embed_reg)

        reg_tokens_add_pos_embed_reg = network.add_elementwise(reg_tokens_constant.get_output(0), 
                                                               pos_embed_reg_constant.get_output(0), trt.ElementWiseOperation.SUM)

        concat_layer_1 = network.add_concatenation([add_layer_2.get_output(0), add_layer_3.get_output(0), 
                                                    add_layer_1.get_output(0), reg_tokens_add_pos_embed_reg.get_output(0)])
        concat_layer_1.axis = 1

        # blocks
        block0_layer = block_n(network=network, weights=weights, block_num=0, input_x_tensor=concat_layer_1.get_output(0))
        block1_layer = block_n(network=network, weights=weights, block_num=1, input_x_tensor=block0_layer.get_output(0))
        block2_layer = block_n(network=network, weights=weights, block_num=2, input_x_tensor=block1_layer.get_output(0))
        block3_layer = block_n(network=network, weights=weights, block_num=3, input_x_tensor=block2_layer.get_output(0))

        # box_head

        # score head
        print(f">>>shape: {block3_layer.get_output(0).shape}")



if __name__=="__main__":
    construct_network()
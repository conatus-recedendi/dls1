import pickle
import os, sys

sys.path.append(os.pardir)
import numpy as np
from common.layer import (
    ConvolutionLayer,
    ReLULayer,
    PoolingLayer,
    AffineLayer,
    SoftmaxWithLoss,
    DropoutLayer,
)


# class DeepConvNet:
#     def __init__(
#         self,
#         input_dim=(1, 28, 28),
#         hidden_size=100,
#         output_size=10,
#         weight_init_std="he",
#         dropout_ratio=0,
#     ):
#         self.input_dim = input_dim

#         pre_node_nums = np.array(
#             [
#                 1 * 3 * 3,
#                 16 * 3 * 3,
#                 16 * 3 * 3,
#                 32 * 3 * 3,
#                 32 * 3 * 3,
#                 64 * 3 * 3,
#                 64 * 4 * 4,
#                 hidden_size,
#             ]
#         )
#         weight_init_scales = np.sqrt(
#             2.0 / pre_node_nums
#         )  # ReLU를 사용할 때의 권장 초깃값

#         conv_param_list = [
#             {"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
#             {"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
#             {"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
#             {"filter_num": 32, "filter_size": 3, "pad": 2, "stride": 1},
#             {"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
#             {"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
#         ]

#         pre_channel_num = input_dim[0]

#         self.params = {}
#         # 1 ~ 6
#         for idx, conv_param in enumerate(conv_param_list):
#             self.params[f"W{idx+1}"] = weight_init_scales[idx] * np.random.randn(
#                 conv_param["filter_num"],
#                 pre_channel_num,
#                 conv_param["filter_size"],
#                 conv_param["filter_size"],
#             )
#             self.params[f"b{idx+1}"] = np.zeros(conv_param["filter_num"])
#             pre_channel_num = conv_param["filter_num"]

#         self.params["W7"] = weight_init_scales[6] * np.random.randn(
#             pre_node_nums[6], hidden_size
#         )
#         self.params["b7"] = np.zeros(hidden_size)
#         self.params["W8"] = weight_init_scales[7] * np.random.randn(
#             hidden_size, output_size
#         )
#         self.params["b8"] = np.zeros(output_size)

#         self.layers = []

#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W1"],
#                 self.params["b1"],
#                 stride=conv_param_list[0]["stride"],
#                 pad=conv_param_list[0]["pad"],
#             )
#         )

#         self.layers.append(ReLULayer())

#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W2"],
#                 self.params["b2"],
#                 stride=conv_param_list[1]["stride"],
#                 pad=conv_param_list[1]["pad"],
#             )
#         )

#         self.layers.append(ReLULayer())
#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))

#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W3"],
#                 self.params["b3"],
#                 stride=conv_param_list[2]["stride"],
#                 pad=conv_param_list[2]["pad"],
#             )
#         )

#         self.layers.append(ReLULayer())

#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W4"],
#                 self.params["b4"],
#                 stride=conv_param_list[3]["stride"],
#                 pad=conv_param_list[3]["pad"],
#             )
#         )

#         self.layers.append(ReLULayer())

#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))

#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W5"],
#                 self.params["b5"],
#                 stride=conv_param_list[4]["stride"],
#                 pad=conv_param_list[4]["pad"],
#             )
#         )

#         self.layers.append(ReLULayer())

#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W6"],
#                 self.params["b6"],
#                 stride=conv_param_list[5]["stride"],
#                 pad=conv_param_list[5]["pad"],
#             )
#         )

#         self.layers.append(ReLULayer())

#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))

#         self.layers.append(AffineLayer(self.params["W7"], self.params["b7"]))

#         self.layers.append(ReLULayer())

#         self.layers.append(DropoutLayer(dropout_ratio))

#         self.layers.append(AffineLayer(self.params["W8"], self.params["b8"]))

#         self.layers.append(DropoutLayer(dropout_ratio))


#         self.last_layer = SoftmaxWithLoss()
# class DeepConvNet:
#     def __init__(
#         self,
#         input_dim=(1, 28, 28),
#         hidden_size=50,
#         output_size=10,
#         weight_init_std="he",
#         dropout_ratio=0,
#     ):
#         self.input_dim = input_dim

#         # Conv + FC 레이어별 pre_node 수 (He 초기화를 위해)
#         pre_node_nums = np.array(
#             [
#                 1 * 3 * 3,  # Conv1: 1x3x3 input
#                 16 * 3 * 3,  # Conv2: 16x3x3 input
#                 32 * 3 * 3,  # Conv3: 32x3x3 input
#                 64 * 3 * 3,  # Flattened: 64x3x3
#                 hidden_size,
#             ]
#         )
#         weight_init_scales = np.sqrt(2.0 / pre_node_nums)

#         # 줄인 conv 구조
#         conv_param_list = [
#             {"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
#             {"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
#             {"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
#         ]

#         pre_channel_num = input_dim[0]
#         self.params = {}

#         # Convolution 레이어
#         for idx, conv_param in enumerate(conv_param_list):
#             self.params[f"W{idx+1}"] = weight_init_scales[idx] * np.random.randn(
#                 conv_param["filter_num"],
#                 pre_channel_num,
#                 conv_param["filter_size"],
#                 conv_param["filter_size"],
#             )
#             self.params[f"b{idx+1}"] = np.zeros(conv_param["filter_num"])
#             pre_channel_num = conv_param["filter_num"]

#         # FC 레이어: 64 x 3 x 3 → hidden_size → output_size
#         self.params["W4"] = weight_init_scales[3] * np.random.randn(
#             64 * 3 * 3, hidden_size
#         )
#         self.params["b4"] = np.zeros(hidden_size)

#         self.params["W5"] = weight_init_scales[4] * np.random.randn(
#             hidden_size, output_size
#         )
#         self.params["b5"] = np.zeros(output_size)

#         self.layers = []

#         # Conv1 + ReLU + Pool
#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W1"],
#                 self.params["b1"],
#                 stride=conv_param_list[0]["stride"],
#                 pad=conv_param_list[0]["pad"],
#             )
#         )
#         self.layers.append(ReLULayer())
#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))

#         # Conv2 + ReLU + Pool
#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W2"],
#                 self.params["b2"],
#                 stride=conv_param_list[1]["stride"],
#                 pad=conv_param_list[1]["pad"],
#             )
#         )
#         self.layers.append(ReLULayer())
#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))

#         # Conv3 + ReLU + Pool
#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W3"],
#                 self.params["b3"],
#                 stride=conv_param_list[2]["stride"],
#                 pad=conv_param_list[2]["pad"],
#             )
#         )
#         self.layers.append(ReLULayer())
#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))

#         # Affine → ReLU → Dropout → Affine → Dropout
#         self.layers.append(AffineLayer(self.params["W4"], self.params["b4"]))
#         self.layers.append(ReLULayer())
#         self.layers.append(DropoutLayer(dropout_ratio))

#         self.layers.append(AffineLayer(self.params["W5"], self.params["b5"]))
#         self.layers.append(DropoutLayer(dropout_ratio))

#         # Softmax with Loss
#         self.last_layer = SoftmaxWithLoss()


# class DeepConvNet:
#     def __init__(
#         self,
#         input_dim=(1, 28, 28),
#         hidden_size=30,
#         output_size=10,
#         weight_init_std="he",
#         dropout_ratio=0,
#     ):
#         self.input_dim = input_dim

#         # Flatten input 크기 = 32 × 3 × 3 = 288
#         pre_node_nums = np.array(
#             [
#                 1 * 3 * 3,  # Conv1
#                 16 * 3 * 3,  # Conv2
#                 32 * 3 * 3,  # FC input (Flatten)
#                 hidden_size,  # FC hidden
#             ]
#         )
#         weight_init_scales = np.sqrt(2.0 / pre_node_nums)

#         # 간소화된 Conv 구조
#         conv_param_list = [
#             {"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
#             {"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
#         ]

#         pre_channel_num = input_dim[0]
#         self.params = {}

#         # Convolution Layer 파라미터
#         for idx, conv_param in enumerate(conv_param_list):
#             self.params[f"W{idx+1}"] = weight_init_scales[idx] * np.random.randn(
#                 conv_param["filter_num"],
#                 pre_channel_num,
#                 conv_param["filter_size"],
#                 conv_param["filter_size"],
#             )
#             self.params[f"b{idx+1}"] = np.zeros(conv_param["filter_num"])
#             pre_channel_num = conv_param["filter_num"]

#         # FC Layer 파라미터
#         self.params["W3"] = weight_init_scales[2] * np.random.randn(
#             32 * 3 * 3, hidden_size
#         )
#         self.params["b3"] = np.zeros(hidden_size)

#         self.params["W4"] = weight_init_scales[3] * np.random.randn(
#             hidden_size, output_size
#         )
#         self.params["b4"] = np.zeros(output_size)

#         self.layers = []

#         # Conv1 → ReLU → Pool
#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W1"],
#                 self.params["b1"],
#                 stride=conv_param_list[0]["stride"],
#                 pad=conv_param_list[0]["pad"],
#             )
#         )
#         self.layers.append(ReLULayer())
#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))  # → 14x14

#         # Conv2 → ReLU → Pool (추가)
#         self.layers.append(
#             ConvolutionLayer(
#                 self.params["W2"],
#                 self.params["b2"],
#                 stride=conv_param_list[1]["stride"],
#                 pad=conv_param_list[1]["pad"],
#             )
#         )
#         self.layers.append(ReLULayer())
#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))  # → 7x7
#         self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))  # → 3x3

#         # Affine → ReLU → Dropout
#         self.layers.append(AffineLayer(self.params["W3"], self.params["b3"]))
#         self.layers.append(ReLULayer())
#         self.layers.append(DropoutLayer(dropout_ratio))

#         # Affine → Dropout
#         self.layers.append(AffineLayer(self.params["W4"], self.params["b4"]))
#         self.layers.append(DropoutLayer(dropout_ratio))


#         self.last_layer = SoftmaxWithLoss()
class DeepConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        hidden_size=30,
        output_size=10,
        weight_init_std="he",
        dropout_ratio=0,
    ):
        self.input_dim = input_dim

        # Conv → Pool → Flatten → FC(30) → FC(10)
        pre_node_nums = np.array(
            [
                1 * 3 * 3,  # Conv1
                16 * 14 * 14,  # FC input = 1 Conv + 1 Pool
                hidden_size,
            ]
        )
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        conv_param = {"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1}

        self.params = {}

        # Convolution Layer (1개)
        self.params["W1"] = weight_init_scales[0] * np.random.randn(
            conv_param["filter_num"],
            input_dim[0],
            conv_param["filter_size"],
            conv_param["filter_size"],
        )
        self.params["b1"] = np.zeros(conv_param["filter_num"])

        # FC Layers
        self.params["W2"] = weight_init_scales[1] * np.random.randn(
            16 * 14 * 14, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)

        self.params["W3"] = weight_init_scales[2] * np.random.randn(
            hidden_size, output_size
        )
        self.params["b3"] = np.zeros(output_size)

        self.layers = []

        # Conv1 → ReLU → Pool
        self.layers.append(
            ConvolutionLayer(
                self.params["W1"],
                self.params["b1"],
                stride=conv_param["stride"],
                pad=conv_param["pad"],
            )
        )
        self.layers.append(ReLULayer())
        self.layers.append(PoolingLayer(pool_h=2, pool_w=2, stride=2))  # (28→14)

        # FC1 → ReLU → Dropout
        self.layers.append(AffineLayer(self.params["W2"], self.params["b2"]))
        self.layers.append(ReLULayer())
        self.layers.append(DropoutLayer(dropout_ratio))

        # FC2 → Dropout
        self.layers.append(AffineLayer(self.params["W3"], self.params["b3"]))
        self.layers.append(DropoutLayer(dropout_ratio))

        self.last_layer = SoftmaxWithLoss()
        # print weight parametr count
        # params_cnt = 0
        # for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
        #     params_cnt += np.prod(self.layers[layer_idx].W.shape)
        #     params_cnt += np.prod(self.layers[layer_idx].b.shape)
        params_cnt = 0
        for key, value in self.params.items():
            params_cnt += np.prod(value.shape)
        # print(f"Total params count: {params_cnt}")
        # print("Total params count: ", params_cnt)
        print("Total params count: ", params_cnt)

    def predict(self, x, train_flag=False):
        # print("predict")
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        return x

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers)
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        # for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
        #     grads["W" + str(i + 1)] = self.layers[layer_idx].dW
        #     grads["b" + str(i + 1)] = self.layers[layer_idx].db
        for i, layer_idx in enumerate((0, 3, 6)):
            grads["W" + str(i + 1)] = self.layers[layer_idx].dW
            grads["b" + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def loss(self, x, t):
        y = self.predict(x, train_flag=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx, train_flag=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def memory_usage(self):
        usage = {}
        total = {"params": 0, "gradients": 0, "activation": 0}
        for i, layer in enumerate(self.layers):
            layer_usage = layer.memory_usage()
            usage[f"Layer {i} ({layer.__class__.__name__})"] = layer_usage
            for key in total:
                total[key] += layer_usage[key]
        usage["Total"] = total
        return usage

    def training_time(self):
        time_usage = {}
        total_forward = 0
        total_backward = 0
        for i, layer in enumerate(self.layers):
            time_usage[f"Layer {i} ({layer.__class__.__name__})"] = {
                "forward": layer.forward_time,
                "backward": layer.backward_time,
            }
            total_forward += layer.forward_time
            total_backward += layer.backward_time
        time_usage["Total"] = {"forward": total_forward, "backward": total_backward}
        return time_usage

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(os.path.dirname(__file__) + "/" + file_name, "rb") as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        # for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
        #     self.layers[layer_idx].W = self.params["W" + str(i + 1)]
        #     self.layers[layer_idx].b = self.params["b" + str(i + 1)]

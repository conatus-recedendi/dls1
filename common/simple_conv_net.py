import os, sys

sys.path.append(os.pardir)
import numpy as np
from common.layer import (
    ConvolutionLayer,
    ReLULayer,
    PoolingLayer,
    AffineLayer,
    SoftmaxWithLoss,
)


class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        self.input_dim = input_dim
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]

        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) / filter_stride + 1

        pool_output_size = int(
            filter_num * (conv_output_size) / 2 * conv_output_size / 2
        )

        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)

        self.params["W2"] = weight_init_std * np.random.rand(
            pool_output_size, hidden_size
        )

        self.params["b2"] = np.zeros(hidden_size)

        self.params["W3"] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        self.layers = {}
        self.layers["Conv1"] = ConvolutionLayer(
            self.params["W1"], self.params["b1"], stride=filter_stride, pad=filter_pad
        )

        self.layers["Relu1"] = ReLULayer()
        self.layers["Pooling1"] = PoolingLayer(pool_h=2, pool_w=2, stride=2)

        self.layers["Affine1"] = AffineLayer(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = ReLULayer()

        self.layers["Affine2"] = AffineLayer(self.params["W3"], self.params["b3"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # print("predict")
        for key, layer in self.layers.items():
            # print("layer",/ x.shape)
            x = layer.forward(x)
        return x

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db
        return grads

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

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

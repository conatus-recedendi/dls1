import numpy as np
from collections import OrderedDict
from common.layer import (
    ReLULayer,
    SigmoidLayer,
    AffineLayer,
    SoftmaxWithLoss,
    BatchNormalization,
    DropoutLayer,
)


class MultiLayerNet:
    """완전연결 다층 신경망

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기
    """

    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
        batch_norm=False,
        use_dropout=False,
        dropout_ratio=0.15,
    ):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.batch_norm = batch_norm
        self.params = {}
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio

        self.init_weight(weight_init_std)

        activation_layer = {"sigmoid": SigmoidLayer, "relu": ReLULayer}
        self.layers = OrderedDict()

        for idx in range(self.hidden_layer_num):
            self.layers["Affine" + str(idx)] = AffineLayer(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )
            if self.batch_norm:
                self.params["gamma" + str(idx)] = np.ones(hidden_size_list[idx])
                self.params["beta" + str(idx)] = np.zeros(hidden_size_list[idx])
                self.layers["BatchNorm" + str(idx)] = BatchNormalization(
                    self.params["gamma" + str(idx)], self.params["beta" + str(idx)]
                )

            if self.use_dropout:
                self.layers["Dropout" + str(idx)] = DropoutLayer(self.dropout_ratio)
            self.layers["Activation_function" + str(idx)] = activation_layer[
                activation
            ]()

        self.layers["Affine" + str(self.hidden_layer_num)] = AffineLayer(
            self.params[f"W{self.hidden_layer_num}"],
            self.params[f"b{self.hidden_layer_num}"],
        )

        self.last_layer = SoftmaxWithLoss()

    def init_weight(self, weight_init_std):
        params_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(0, len(params_list) - 1):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / params_list[idx])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / params_list[idx])
            self.params["W" + str(idx)] = scale * np.random.randn(
                params_list[idx], params_list[idx + 1]
            )
            self.params["b" + str(idx)] = np.zeros(params_list[idx + 1])

    def predict(self, x, train_flg=False):
        for key in self.layers.keys():
            layer = self.layers[key]

            if "BatchNorm" in key or "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(self.hidden_layer_num):
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        pass

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(self.hidden_layer_num + 1):

            grads["W" + str(idx)] = (
                self.layers["Affine" + str(idx)].dW
                + self.weight_decay_lambda * self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db

            if self.batch_norm and idx != self.hidden_layer_num:
                grads["gamma" + str(idx)] = self.layers["BatchNorm" + str(idx)].dgamma
                grads["beta" + str(idx)] = self.layers["BatchNorm" + str(idx)].dbeta

        return grads

    def memory_usage(self):
        usage = {}
        total = {"params": 0, "gradients": 0, "activation": 0}
        for i, layer in enumerate(self.layers.values()):
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
        for i, layer in enumerate(self.layers.values()):
            time_usage[f"Layer {i} ({layer.__class__.__name__})"] = {
                "forward": layer.forward_time,
                "backward": layer.backward_time,
            }
            total_forward += layer.forward_time
            total_backward += layer.backward_time
        time_usage["Total"] = {"forward": total_forward, "backward": total_backward}
        return time_usage

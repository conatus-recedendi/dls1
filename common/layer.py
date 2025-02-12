from common.np import *  # import numpy as np
from common.config import GPU
import time
from common.functions import cross_entropy_error, softmax, sigmoid
from common.utils import im2col, col2im


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x, y):
        start = time.perf_counter()
        self.x = x
        self.y = y
        self.forward_time += time.perf_counter() - start
        return x * y

    def backward(self, dout):
        start = time.perf_counter()
        # dout = dL/dz
        # dL / dx = dL / dz * dz / dx = dout * y
        dx = dout * self.y
        dy = dout * self.x
        self.backward_time += time.perf_counter() - start
        return dx, dy

    def memory_usage(self):
        # ReLU는 파라미터나 기울기를 갖지 않으므로 activation만 계산
        mem_params = (
            self.x.nbytes + self.y.nbytes
            if self.x is not None and self.y is not None
            else 0
        )
        mem_gradients = 0
        mem_activation = 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class AddLayer:
    def __init__(self):
        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x, y):
        start = time.perf_counter()
        self.forward_time += time.perf_counter() - start
        return x + y

    def backward(self, dout):
        start = time.perf_counter()
        dx = dout * 1
        dy = dout * 1
        self.backward_time += time.perf_counter() - start
        return dx, dy

    def memory_usage(self):
        # ReLU는 파라미터나 기울기를 갖지 않으므로 activation만 계산
        mem_params = 0
        mem_gradients = 0
        mem_activation = 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class ReLULayer:
    def __init__(self):
        self.mask = None
        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x):
        start = time.perf_counter()
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        self.forward_time += time.perf_counter() - start
        return out

    def backward(self, dout):
        start = time.perf_counter()
        # dL / dx = dL / dz * dz / dx = dout * mask
        dout[self.mask] = 0
        dx = dout
        self.backward_time += time.perf_counter() - start

        return dx

    def memory_usage(self):
        # ReLU는 파라미터나 기울기를 갖지 않으므로 activation만 계산
        mem_params = 0
        mem_gradients = 0
        mem_activation = self.mask.nbytes if self.mask is not None else 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class SigmoidLayer:
    def __init__(self):
        self.out = None
        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x):
        start = time.perf_counter()
        out = sigmoid(x)
        self.out = out
        self.forward_time += time.perf_counter() - start
        return out

    def backward(self, dout):
        start = time.perf_counter()

        # - 1 / (1 + exp(-x))^2 * (-exp(-x)) = exp(-x) / (1 + exp(-1))^2
        # = out * (1 - out)

        # = out^2 * ( 1 / (out) - 1 ) = - out^2 + out = out (* ( 1- out))
        # dL / dx = dL / dz * dz / dx =
        dx = self.out * (1 - self.out) * dout
        self.backward_time += time.perf_counter() - start

        return dx

    def memory_usage(self):
        mem_params = 0
        mem_gradients = 0
        mem_activation = self.out.nbytes if self.out is not None else 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.original_x_shape = None

        self.x = None
        self.dW = None
        self.db = None

        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x):

        start = time.perf_counter()
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        # (2, 2) o
        out = np.dot(self.x, self.W) + self.b
        self.forward_time += time.perf_counter() - start
        return out

    def backward(self, dout):
        start = time.perf_counter()
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)

        self.backward_time += time.perf_counter() - start
        return dx

    def memory_usage(self):
        mem_params = self.W.nbytes + self.b.nbytes
        mem_gradients = self.dW.nbytes + self.db.nbytes
        mem_activation = self.x.nbytes if self.x is not None else 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x, t):
        start = time.perf_counter()
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, t)
        self.forward_time += time.perf_counter() - start
        return self.loss

    def backward(self, dout=1):
        start = time.perf_counter()
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        self.backward_time += time.perf_counter() - start
        return dx

    def memory_usage(self):

        mem_params = (
            self.y.nbytes + self.t.nbytes
            if self.y is not None and self.t is not None
            else 0
        )
        mem_gradients = 0
        mem_activation = self.mask.nbytes if self.mask is not None else 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta

        self.momentum = momentum
        self.input_size = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x, train_flg=True):
        start = time.perf_counter()
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        self.forward_time += time.perf_counter() - start

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        start = time.perf_counter()
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        self.backward_time += time.perf_counter() - start
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

    def memory_usage(self):
        mem_params = (
            self.gamma.nbytes
            + self.beta.nbytes
            + self.running_mean.nbytes
            + self.running_var.nbytes
            + self.std.nbytes
            + self.xc.nbytes
            + self.xn.nbytes
            if self.gamma is not None and self.beta is not None
            else 0
        )
        mem_gradients = self.dgamma.nbytes + self.dbeta.nbytes
        mem_activation = 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class PoolingLayer:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
        self.forward_time = 0
        self.backward_time = 0

    # (1, C, H, W) -> (1, C, 1 + (H + 2 * pad - pool_h) / stride , 1+ (W + 2 * pad + pool_w) / stride)
    # -> (1, C, H, W)
    def forward(self, x):
        start = time.perf_counter()
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)

        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        self.forward_time += time.perf_counter() - start
        return out

    def backward(self, dout):
        start = time.perf_counter()
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        self.backward_time += time.perf_counter() - start

        return dx

    def memory_usage(self):
        # ReLU는 파라미터나 기울기를 갖지 않으므로 activation만 계산
        mem_params = (
            self.x.nbytes + self.arg_max.nbytes
            if self.x is not None and self.arg_max is not None
            else 0
        )
        mem_gradients = 0
        mem_activation = 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class ConvolutionLayer:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x):
        start = time.perf_counter()
        N, C, H, W = x.shape
        FN, C, FH, FW = self.W.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        self.forward_time += time.perf_counter() - start

        return out

    def backward(self, dout):
        start = time.perf_counter()
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        self.backward_time += time.perf_counter() - start
        return dx

    def memory_usage(self):
        mem_params = (
            self.W.nbytes
            + self.b.nbytes
            + self.col.nbytes
            + self.col_W.nbytes
            + self.x.nbytes
        )
        mem_gradients = self.dW.nbytes + self.db.nbytes
        mem_activation = 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }


class DropoutLayer:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.forward_time = 0
        self.backward_time = 0

    def forward(self, x, train_flg=True):
        start = time.perf_counter()
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            self.forward_time += time.perf_counter() - start
            return x * self.mask
        else:
            self.forward_time += time.perf_counter() - start
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

    def memory_usage(self):
        # ReLU는 파라미터나 기울기를 갖지 않으므로 activation만 계산
        mem_params = 1
        mem_gradients = 0
        mem_activation = self.mask.nbytes if self.mask is not None else 0
        return {
            "params": mem_params,
            "gradients": mem_gradients,
            "activation": mem_activation,
        }

"""Network layers."""

import numpy as np
from tinynn.core.initializer import Ones
from tinynn.core.initializer import XavierUniform
from tinynn.core.initializer import Zeros
from tinynn.utils.math import sigmoid


def empty(shape, dtype=np.float32):
    return np.empty(shape, dtype=dtype)


class Layer:
    """Base class for layers."""

    def __init__(self):
        self.params ={p: None for p in self.param_names}
        self.nt_params = {p: None for p in self.nt_param_names}
        self.initializers = {}

        self.grads = {}
        self.shapes = {}

        self._is_training = True  # used in BatchNorm / dropout layers
        self._is_init = False

        self.ctx = {}

    def __repr__(self):
        shape = None if not self.shapes else self.shapes
        return f"layer: {self.name}\tshape: {shape}"

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError

    @property
    def is_init(self):
        return self._is_init

    @is_init.setter
    def is_init(self, is_init):
        self._is_init = is_init
        for name in self.param_names:
            self.shapes[name] = self.params[name].shape

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, is_train):
        self._is_training = is_train

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def param_names(self):
        return ()

    @property
    def nt_param_names(self):
        return ()

    def _init_params(self):
        for name in self.param_names:
            self.params[name] = self.initializers[name](self.shapes[name])
        self.is_init = True


class Dense(Layer):
    """
    A dense layer operates 'outputs = dot(inputs, weights) + bias'
    :param num_out: A positive integer, number of output neurons
    :param w_init: weight initializer
    :param b_init: bias initializer
    """
    def __init__(self, num_out, w_init=XavierUniform(), b_init=Zeros()):
        super().__init__()

        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": [None, num_out], 'b': [num_out]}

    def forward(self, inputs):
        if not self.is_init:
            self.shapes['w'][0] = inputs.shape[1]
            self._init_params()
        self.ctx = {'X': inputs}
        return inputs @ self.params["w"].T

    @property
    def param_names(self):
        return 'w', 'b'


def im2col(img, k_h, k_w, s_h, s_w):
    """
    Transform padded image into column matrix.
    :param img: padded inputs of shape (B, in_h, in_w, in_c)
    :param k_h: kernel height
    :param k_w: kernel width
    :param s_h: stride height
    :param s_w: stride width
    :return col: column matrix of shape: (B * out_h * out_w, k_h, k * h * in_c)
    """
    batch_sz, h, w, in_c = img.shape
    # calculate output feature map size
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1

    # allocate space for column matrix
    col = empty((batch_sz * out_h * out_w, k_h * k_w * in_c))
    # fill in the column matrix
    batch_span = out_w * out_h
    for r in range(out_h):
        r_start = r * s_h
        matrix_r = r * out_w
        for c in range(out_w):
            c_start = c * s_w
            patch = img[:, r_start: r_start + k_h, c_start: c_start + k_w, :]
            patch = patch.reshape(batch_sz, -1)
            col[matrix_r + c:: batch_span, :] = patch

    return col


def get_padding_2d(in_shape, k_shape, mode):

    def get_padding_1d(w, k):
        if mode == 'SAME':
            pads = (w - 1) + k - w
            half = pads // 2
            padding = (half, half) if pads % 2 == 0 else (half, half + 1)

        else:
            padding = (0, 0)
        return padding

    h_pad = get_padding_1d(in_shape[0], k_shape[0])
    w_pad = get_padding_1d(in_shape[1], k_shape[1])
    return (0, 0), h_pad, w_pad, (0, 0)


class Conv2D(Layer):
    """
    Implement 2D convolution layer
    :param kernel: a list/tuple of int that has length 4-(height, width, in_channels, out_channels)
    :param stride: a list/tuple of int that has length 2-(height, width)
    :param padding: string ['SAME', 'VALID'']
    :param w_init: weight initializer
    :param b_init: bias initializer
    """
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding='SAME',
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {'w': w_init, 'b': b_init}
        self.shapes = {'w': self.kernel_shape[0], 'b': self.kernel_shape[-1]}

        self.padding_mode = padding
        self. padding = None

    def forward(self, inputs):
        """Accelerate convolution via im2col trick.
        An example (assuming only one channel and one filter):
         input = | 43  16  78 |         kernel = | 4  6 |
          (X)    | 34  76  95 |                  | 7  9 |
                 | 35   8  46 |

        After im2col and kernel flattening:
         col  = | 43  16  34  76 |     kernel = | 4 |
                | 16  78  76  95 |      (W)     | 6 |
                | 34  76  35   8 |              | 7 |
                | 76  95   8  46 |              | 9 |
        """
        if not self.is_init:
            self._init_params()

        k_h, k_w, _, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(inputs)

        # padded inputs to column matrix
        col = im2col(X, k_h, k_w, s_h, s_w)
        # perform convolution by matrix product.
        W = self.params['w'].reshape(-1, out_c)
        Z = col @ W
        # reshape output
        batch_sz, in_h, in_w, _ = X.shape
        Z = Z.reshape(batch_sz, Z.shape[0] // batch_sz, out_c)
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        Z = Z.reshape(batch_sz, out_h, out_w, out_c)

        Z += self.params['b']

        # save results for backward function
        self.ctx = {'X_shape': X.shape, 'col': col, 'W': W}
        return Z

    def backward(self, grad):
        """Compute gradients w.r.t. layer parameters and backward gradients.
        :param grad: gradients from previous layer
            with shape (batch_sz, out_h, out_w, out_c)
        :return d_in: gradients to next layers
            with shape (batch_sz, in_h, in_w, in_c)
        """
        # read size parameters
        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        batch_sz, in_h, in_w, in_c = self.ctx["X_shape"]
        pad_h, pad_w = self.padding[1:3]

        # grads w.r.t. parameters
        flat_grad = grad.reshape((-1, out_c))
        d_W = self.ctx["col"].T @ flat_grad
        self.grads["w"] = d_W.reshape(self.kernel_shape)
        self.grads["b"] = np.sum(flat_grad, axis=0)

        # grads w.r.t. inputs
        d_X = grad @ self.ctx["W"].T
        # cast gradients back to original shape as d_in
        d_in = np.zeros(shape=self.ctx["X_shape"], dtype=np.float32)
        for i, r in enumerate(range(0, in_h - k_h + 1, s_h)):
            for j, c in enumerate(range(0, in_w - k_w + 1, s_w)):
                patch = d_X[:, i, j, :]
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r:r+k_h, c:c+k_w, :] += patch

        # cut off gradients of padding
        d_in = d_in[:, pad_h[0]:in_h-pad_h[1], pad_w[0]:in_w-pad_w[1], :]
        return self._grads_postprocess(d_in)

    def _inputs_preprocess(self, inputs):
        _, in_h, in_w, _ = inputs.shape
        k_h, k_w, _, _ = self.kernel_shape
        # padding calculation
        if self.params is None:
            self.padding = get_padding_2d((in_h, in_w), (k_h, k_w), self.padding_mode)
        return np.pad(inputs, pad_width=self.padding, mode='constant')

    def _grads_postprocess(self, grads):
        return  grads

    @property
    def param_names(self):
        return 'w', 'b'


class BatchNormalization(Layer):

    def __init__(self,
                 momentum=0.99,
                 gamma_init=Ones(),
                 beta_init=Zeros(),
                 epsilon=1e-5):
        super().__init__()
        self.m = momentum
        self.epsilon = epsilon

        self.initializers = {'gamma': gamma_init, 'beta': beta_init}
        self.reduce = None

    def forward(self, inputs):
        self.reduce = (0,)
        if not self.is_init:
            for p in self.param_names:
                self.shapes[p] = inputs.shape[-1]
            self._init_params()

        if self.nt_params['r_mean'] is None:
            self.nt_params['r_mean'] = inputs.mean(self.reduce, keepdims=True)
            self.nt_params['r_var'] = inputs.var(self.reduce, keepdims=True)

        if self.is_training:
            mean = inputs.mean(self.reduce, keepdims=True)
            var = inputs.var(self.reduce, keepdims=True)
            self.nt_params['r_mean'] = (self.m * self.nt_params['r_mean'] + (1.0 - self.m) * mean)
            self.nt_params['r_var'] = (self.m * self.nt_params['r_var'] + (1.0 - self.m) * var)
        else:
            mean = self.nt_params['r_mean']
            var = self.nt_params['r_var']

        # standardize
        X_center = inputs - mean
        std = (var + self.epsilon) ** 0.5
        X_norm = X_center / std
        self.ctx = {'X_norm': X_norm, 'std': std, 'x_center': X_center}
        return self.params['gamma'] * X_norm + self.params['beta']

    def backward(self, grad):
        # grads w.r.t. params
        self.grads["gamma"] = (self.ctx["X_norm"] * grad).sum(self.reduce)
        self.grads["beta"] = grad.sum(self.reduce)

        # N = grad.shape[0]
        N = np.prod([grad.shape[d] for d in self.reduce])
        std_inv = 1.0 / self.ctx["std"]
        # grads w.r.t. inputs
        # ref: http://cthorey.github.io./backpropagation/
        d_in = (1.0 / N) * self.params["gamma"] * std_inv * (
                N * grad - np.sum(grad, axis=self.reduce, keepdims=True) -
                self.ctx["X_center"] * std_inv ** 2 *
                (grad * self.ctx["X_center"]).sum(axis=self.reduce, keepdims=True))
        return d_in

    @property
    def param_names(self):
        return 'gamma', 'beta'

    @property
    def nt_param_names(self):
        return 'r_mean', 'r_var'


class Reshape(Layer):

    def __init__(self, *output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], *self.output_shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class Flatten(Reshape):

    def __init__(self):
        super().__init__(-1)


class Dropout(Layer):

    def __init__(self, keep_prob=0.5):
        super().__init__()
        self._keep_prob = keep_prob
        self._multiplier = None

    def forward(self, inputs):
        if self.is_training:
            multiplier = np.random.binomial(1, self._multiplier, inputs.shape)
            self._multiplier = multiplier / self._keep_prob
            outputs = inputs * self._multiplier
        else:
            outputs = inputs
        return outputs

    def backward(self, grad):
        assert self.is_training is True
        return grad * self._multiplier


# TODO (6) ALL ACTIVATIONS: SIGMOID, SOFTPLUS, TANH, RELU, LEAKYRELU, GELU, ELU,
class Activation(Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.ctx['X'] = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative(self.ctx['X']) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def func(self, x):
        return sigmoid(x)

    def derivative(self, x):
        act = self.func(x)
        return act * (1.0 - act)


class Softplus(Activation):

    def func(self, x):
        return np.log(1.0 * np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def derivative(self, x):
        return sigmoid(x)


class Tanh(Activation):

    def func(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 - self.func(x) ** 2


class ReLU(Activation):

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative(self, x):
        return x > 0.0


class LeakyReLU(Activation):

    def __init__(self, slope=0.2):
        super().__init__()
        self._slope = slope

    def func(self, x):
        x = x.copy()
        x[x < 0.0] *= self._slope
        return x

    def derivative(self, x):
        dx = np.ones_like(x)
        dx[x < 0.0] = self._slope
        return dx


class GELU(Activation):
    """
    Gaussian Error linear units
    """

    def __init__(self):
        super().__init__()
        self._alpha = 0.1702
        self._cache = None

    def func(self, x):
        self._cache = sigmoid(self._alpha * x)
        return x * self._cache

    def derivative(self, x):  # change test
        return self._cache + x * self._alpha * self._cache * (1.0 - self._cache)


class ELU(Activation):

    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    def func(self, x):
        return np.maximum(x, 0) + np.minimum(0, self._alpha * (np.exp(x) - 1))

    def derivative(self, x):
        return x > 0.0 + (x < 0.0) * self._alpha * np.exp(x)


def main():
    img = np.random.uniform(0.0, 1.0, size=(32, 32, 32, 16))
    k_h, k_w, s_h, s_w = 1, 1, 1, 1
    out = im2col(img, k_h, k_w, s_h, s_w)
    print(out.shape)


if __name__ == '__main__':
    main()

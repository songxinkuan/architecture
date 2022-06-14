"""Various of network parameter initializers."""

import numpy as np


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


class Initializer:

    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    def int(self, shape):
        raise NotImplementedError


class Normal(Initializer):

    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def init(self, shape):
        return np.random.normal(loc=self._mean, scale=self._std, size=shape)


class TruncatedNormal(Initializer):

    def __init__(self, low, high, mean=0.0, std=1.0):
        self._mean, self._std = mean, std
        self._low, self._high = low, high

    def init(self, shape):
        data = np.random.normal(loc=self._mean, scale=self._std, size=shape)
        while True:
            mask = (data > self._low) & (data < self._high)
            if mask.all():
                break
            data[~mask] = np.random.normal(loc=self._mean, scale=self._std,
                                           size=(~mask).sum())
        return data


def main():
    initializer = TruncatedNormal(low=-1.0, high=1.0)
    weights = initializer.init(shape=(4, 4))
    print(weights)


if __name__ == '__main__':
    main()

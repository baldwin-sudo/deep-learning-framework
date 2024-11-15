import numpy as np

def sigmoid():
    def _sigmoid(x):
        x = np.clip(x, -709, 709)  # Clip to avoid overflow (exp(709) is near the limit of floating-point
        return 1/(1+np.exp(-x))
    def _d_sigmoid(x):
        x = np.clip(x, -709, 709)  # Clip to avoid overflow (exp(709) is near the limit of floating-point
        return _sigmoid(x)*(1-_sigmoid(x))
    # returns both activation,and its deriviative
    return _sigmoid,_d_sigmoid


def linear():
    def _linear(x):
        return x
    def _d_linear(x):
        return 1
    return _linear,_d_linear

def relu():
    def _relu(x):
        return np.maximum(x,0)
    def _d_relu(x):
        return np.where(x > 0, 1, 0)
    return _relu,_d_relu

def tanh():
    def _tanh(x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    def _d_tanh(x):
        return 1-_tanh(x)**2
    return _tanh,_d_tanh
    
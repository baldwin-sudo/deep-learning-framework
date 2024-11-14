import numpy as np
class Optimizer:
    def __init__(self, learning_rate, neural_network, batch_size=None) -> None:
        self.lr = learning_rate
        self.batch_size = batch_size
        self.nn = neural_network

    def step(self, error, step=None):
        """Update weights for a single training example or a batch."""
        pass  # This method will be implemented by subclasses.

class Batch_GD(Optimizer):
    # Batch Gradient Descent (GD)
    def __init__(self, learning_rate, neural_network,len_data) -> None:
        super().__init__(learning_rate, neural_network,len_data)

    def step(self, error, step=None):
        # In batch GD, we update the weights after processing the entire dataset (full batch).
            #here batchsize == len(data)
        if step % self.batch_size == 0 :
            self.batch_step(error)
    def batch_step(self, batch_error):
        """Updates weights for a full batch of training examples."""
        layers_reversed = list(reversed(self.nn.layers))
        for layer in layers_reversed:
            grad,dW,dB = layer.backward(batch_error)
            layer.weights -=dW *self.lr
            layer.bias -=dB *self.lr
            batch_error = grad




class SGD(Optimizer):
    # Stochastic Gradient Descent (SGD) - one update per example
    def __init__(self, learning_rate, neural_network) -> None:
        super().__init__(learning_rate, neural_network)

    def step(self,batch_error, step=None):
        # In SGD, we update the weights for each individual data point.
        layers_reversed = list(reversed(self.nn.layers))
        for layer in layers_reversed:
            grad,dW,dB = layer.backward(batch_error)
            layer.weights -=dW *self.lr
            layer.bias -=dB *self.lr
            batch_error = grad


class MBGD(Optimizer):
    # Mini-Batch Gradient Descent (MBGD) - compromise between BGD and SGD
    def __init__(self, learning_rate, neural_network, batch_size) -> None:
        super().__init__(learning_rate, neural_network, batch_size)

    def step(self, batch_error, step=None):
        # In Mini-Batch GD, we update weights after a mini-batch is processed.
        if step is None or step % self.batch_size == 0:
            layers_reversed = list(reversed(self.nn.layers))
            for layer in layers_reversed:
                grad,dW,dB = layer.backward(batch_error)
                layer.weights -=dW *self.lr
                layer.bias -=dB *self.lr
                batch_error = grad
class SGDM(Optimizer) :
    def __init__(self, learning_rate, neural_network, batch_size=None,momentum=0.9) -> None:
        """
            its a sgd with momentum
            (equivalent to velocity because it uses first order error)
            it  uses pre the vious gradient as momentum
        """
        
        super().__init__(learning_rate, neural_network, batch_size)
        self.momentum=momentum
        # the velocities
        self.layer_velocity_weights=[]
        self.layer_velocity_bias=[]
        # init velocities
        # we reverse  because we start the backprop from the last layer
        for layer in reversed(self.nn.layers):
            self.layer_velocity_weights.append(np.zeros_like(layer.weights))
            self.layer_velocity_bias.append(np.zeros_like(layer.bias))
        
    def step(self, batch_error, step=None):
        if step is None or step % self.batch_size == 0:
            layers_reversed = list(reversed(self.nn.layers))
            for i,layer in enumerate(layers_reversed):
                grad,dW,dB = layer.backward(batch_error)
                # update velocities
                self.layer_velocity_weights[i] = self.momentum * self.layer_velocity_weights[i] + (1-self.momentum)*dW
                self.layer_velocity_bias[i] = self.momentum * self.layer_velocity_bias[i] + (1-self.momentum)*dB
                # update weights :
                layer.weights -=self.layer_velocity_weights[i] *self.lr
                layer.bias -=self.layer_velocity_bias[i] *self.lr
                # propagate error :
                batch_error = grad
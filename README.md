# Deep learning Framework from scratch :

This repository demonstrates a simple implementation of a neural network with support for various data loading strategies and optimization techniques. The focus is on implementing different Optimization algorithms. Additionally, different types of data loaders (for different batch sizes) are included, such as **SimpleLoader**, **MiniBatch_Loader**, and **Batch_Loader**.Different types of losses(criterions) are implemented too .

## Components

### 1. **Data Loaders**

- **DataLoader**: Base class for loading data in batches.
  - **Parameters**:
    - `data`: A tuple containing input features (`X`) and labels (`Y`).
    - `batch_size`: The number of samples per batch.
    - `shuffle`: Whether to shuffle the data before loading.
  
- **Batch_Loader**: Loads the entire dataset in one batch (full batch training).
  - Inherits from `DataLoader`, with the `batch_size` set to the size of the dataset.
  
- **MiniBatch_Loader**: Loads the dataset in mini-batches.
  - Inherits from `DataLoader`, with a defined `batch_size`.

- **SimpleLoader**: A subclass of `MiniBatch_Loader` with a batch size of 1, used for stochastic gradient descent (SGD).

### 2. **Loss Functions (Criteria)**

- **Criterion**: Base class for defining loss functions.
  - `step_error(y, y_pred)`: Calculates the error for a single data point.
  - `step_d_error(y, y_pred)`: Computes the gradient (derivative) of the error.
 
- **MSE (Mean Squared Error)**: Implements the squared error loss for regression problems.
  - `step_error(y, y_pred)`: Calculates the mean squared error.
  - `step_d_error(y, y_pred)`: Computes the gradient of MSE.

- **BinaryCrossEntropy**: Implements the binary cross-entropy loss, commonly used for binary classification.
  - `step_error(y_true, y_proba)`: Calculates the binary cross-entropy error.
  - `step_d_error(y_true, y_proba)`: Computes the gradient of the binary cross-entropy error.

### 3. **Neural Network Layers**

- **Dense Layer**: A fully connected layer that performs a linear transformation followed by an activation function (ReLU, Sigmoid).
  - Each layer consists of weights and biases, which are updated during training.

### 4. **Optimizers**

- **Optimizer**: Base class for gradient descent optimizers.
  - `step(error, step)`: Updates the weights based on the computed error.
  
- **Batch_GD (Batch Gradient Descent)**: Performs weight updates after processing the entire dataset.
  
- **SGD (Stochastic Gradient Descent)**: Updates the weights after processing each individual training example.
  
- **MBGD (Mini-Batch Gradient Descent)**: A hybrid approach that updates weights after processing a mini-batch of training examples.
- **SGDM (Stochastic Gradient Descent with MoMENTUM)**: This approach updates the weights of training examples by incorporating a memory system that tracks the optimization directions. By factoring in past gradients with a momentum term, it helps stabilize the process and accelerates convergence. The momentum term determines the influence of previous gradients on the current update, guiding the optimization toward a more efficient path.

### 5. **Training Function**

- **train_with_dataloader**: Trains the neural network using a given data loader, optimizer, and loss function.
  - **Parameters**:
    - `dataloader`: The data loader instance (e.g., `MiniBatch_Loader`, `SimpleLoader`, `Batch_Loader`).
    - `network`: The neural network to train.
    - `optimizer`: The optimizer to use for weight updates.
    - `criterion`: The loss function (e.g., `MSE`, `BinaryCrossEntropy`).
    - `epochs`: The number of epochs for training (default is 10,000).

  The training process involves:
  - Forward pass: Computes predictions.
  - Backward pass: Updates weights using the optimizer.

### Example Usage

Below is an example of using different data loaders and optimizers for training a neural network on a simple XOR dataset.

```python
import numpy as np
from Layers import Dense
from Activations import relu, sigmoid
from Optimizers import SGD, MBGD, Batch_GD
from Criterions import MSE, BinaryCrossEntropy
from NeuralNetwork import Network
from DataLoader import SimpleLoader, MiniBatch_Loader, Batch_Loader

def test():
    np.random.seed(42)
    
    # Sample data (XOR dataset)
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    Y = np.array([1, 1, 0, 0])

    # Create layers and neural network
    hidden_layer_1 = Dense(input_size=2, num_neurons=3, activation=relu)
    hidden_layer_2 = Dense(input_size=3, num_neurons=4, activation=relu)
    output_layer = Dense(input_size=4, num_neurons=1, activation=sigmoid)

    # Initialize the network and set layers
    network = Network()
    network.set_layers([hidden_layer_1, hidden_layer_2, output_layer])
    
    # Criterion (Loss function)
    criterion = BinaryCrossEntropy()
    
    # Test with MiniBatch_Loader and MBGD optimizer
    print("\nTesting with MiniBatch_Loader (mini-batch iteration) and MBGD optimizer:")
    minibatch_loader = MiniBatch_Loader((X, Y), batch_size=3, shuffle=True)
    mbgd_optimizer = MBGD(learning_rate=0.01, neural_network=network, batch_size=3)
    train_with_dataloader(minibatch_loader, network, mbgd_optimizer, criterion)
    
    network.reset_network()

    # Test with SimpleLoader and SGD
    print("\nTesting with SimpleLoader (stochastic iteration) and SGD optimizer:")
    simple_loader = SimpleLoader((X, Y), shuffle=True)
    sgd_optimizer = SGD(learning_rate=0.1, neural_network=network)
    train_with_dataloader(simple_loader, network, sgd_optimizer, criterion)

    network.reset_network()
    
    # Test with Batch_Loader and Batch_GD optimizer
    print("\nTesting with Batch_Loader (full-batch iteration) and Batch_GD optimizer:")
    batch_loader = Batch_Loader((X, Y), shuffle=True)
    batch_gd_optimizer = Batch_GD(learning_rate=0.1, neural_network=network, len_data=len(X))
    train_with_dataloader(batch_loader, network, batch_gd_optimizer, criterion)

if __name__ == "__main__":
    test()
```

### 6. **Explanation of the Training Process**

- **Forward Pass**: For each batch, the input is passed through the layers of the network, and the predictions are computed.
- **Loss Calculation**: The predicted values are compared with the true labels using the specified loss function (e.g., MSE or Binary Cross-Entropy).
- **Backward Pass**: The gradients are computed based on the error (difference between predictions and actual values) and used to update the weights using the optimizer.
- **Optimization Step**: The optimizer (SGD, MBGD, or BGD) is responsible for updating the weights based on the computed gradients.

### Conclusion

This framework provides a flexible way to train a neural network using different types of gradient descent and data loading strategies. It can be extended with additional features like different activation functions, more advanced optimizers, and custom loss functions.


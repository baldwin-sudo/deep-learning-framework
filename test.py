from Layers import Dense
from DataLoader import SimpleLoader, Batch_Loader, MiniBatch_Loader
from Activations import relu, sigmoid,linear
from Optimizers import SGD, MBGD, Batch_GD,SGDM,NAG
from Criterions import MSE,BinaryCrossEntropy
from NeuralNetwork import Network
import numpy as np 

def train_with_dataloader(dataloader, network, optimizer, criterion, epochs=10000):
    for epoch in range(epochs):
        
        
        for step, (batch_x, batch_y) in enumerate(dataloader):
            y_preds = []
            y_true = []
            # print(step)
            #batch_x = np.array(batch_x)  # Ensure it is a numpy array
            #batch_x = batch_x.reshape(batch_x.shape[0], -1)  # Reshape to (batch_size, input_size)
            # print(batch_x)
            # Forward pass through the entire batch
            predictions = network.forward(batch_x)  # Forward pass for the entire batch
            y_preds.extend(predictions)  # Append batch predictions
            y_true.extend(batch_y)  # Append true values for the batch
            
            # Compute error (mean squared error)
            error=criterion.step_error(batch_y, predictions)
            
            # Backward pass
            d_error = criterion.step_d_error(batch_y, predictions)
            optimizer.step( d_error,step)
            
            
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Error: {error}")
    
    # After training, evaluate the model's accuracy
    y_preds = np.array([1 if proba > 0.5 else 0 for proba in y_preds])
    y_true = np.array(y_true)
    accuracy = np.mean(y_true == y_preds)
    print(f"Accuracy: {accuracy}")

def test():
    np.random.seed(4)
    
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
    # print("\nTesting with MiniBatch_Loader (mini-batch iteration) and MBGD optimizer:")
    # minibatch_loader = MiniBatch_Loader((X, Y), batch_size=2, shuffle=True)
    # mbgd_optimizer = MBGD(learning_rate=0.01, neural_network=network, batch_size=2)
    # train_with_dataloader(minibatch_loader, network, mbgd_optimizer, criterion)
    
    # network.reset_network()

    # Test with SimpleLoader and SGD
    batch_size=1
    print("\nTesting with SimpleLoader (stochastic iteration) and Nestrov optimizer:")
    simple_loader = MiniBatch_Loader((X, Y), shuffle=True,batch_size=batch_size)
    sgd_optimizer = NAG(learning_rate=0.01, neural_network=network,batch_size=batch_size)
    train_with_dataloader(simple_loader, network, sgd_optimizer, criterion)

    network.reset_network()
    # # Test with SimpleLoader and SGD
    print("\nTesting with SimpleLoader (stochastic iteration) and SGDM optimizer:")
    simple_loader = MiniBatch_Loader((X, Y), shuffle=True,batch_size=batch_size)
    sgd_optimizer = SGDM(learning_rate=0.01, neural_network=network,batch_size=batch_size)
    train_with_dataloader(simple_loader, network, sgd_optimizer, criterion)

    network.reset_network()
    
    # Test with Batch_Loader and Batch_GD optimizer
    # print("\nTesting with Batch_Loader (full-batch iteration) and Batch_GD optimizer:")
    # batch_loader = Batch_Loader((X, Y), shuffle=True)
    # batch_gd_optimizer = Batch_GD(learning_rate=0.001, neural_network=network, len_data=len(X))
    # train_with_dataloader(batch_loader, network, batch_gd_optimizer, criterion,epochs=10000)
    
# test()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def test_with_different_dataset():
    # Load the Iris dataset and select two classes (for binary classification)
    iris = load_iris()
    X = iris.data
    Y = iris.target
    
    # We'll select only the first two classes (0 and 1) for binary classification
    X = X[Y != 2]
    Y = Y[Y != 2]
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Standardize the input data (important for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    np.random.seed(42)
    # Crenpate layers and neural network
    hidden_layer_1 = Dense(input_size=4, num_neurons=5, activation=relu)
    hidden_layer_2 = Dense(input_size=5, num_neurons=4, activation=relu)
    output_layer = Dense(input_size=4, num_neurons=1, activation=sigmoid)

    # Initialize the network and set layers
    network = Network()
    network.set_layers([hidden_layer_1, hidden_layer_2, output_layer])
    
    # Criterion (Loss function)
    criterion = BinaryCrossEntropy()
    
    # Test with MiniBatch_Loader and MBGD optimizer
    batch_size=32
    epochs=2000
    print("\nTesting with MiniBatch_Loader (mini-batch iteration) and MBGD optimizer:")
    minibatch_loader = MiniBatch_Loader((X_train, Y_train), batch_size=batch_size, shuffle=True)
    mbgd_optimizer = SGDM(learning_rate=0.001, neural_network=network,momentum=0.8,batch_size=batch_size)
    train_with_dataloader(minibatch_loader, network, mbgd_optimizer, criterion,epochs=epochs)

    network.reset_network()

    # Test with SimpleLoader and SGDM optimizer
    print("\nTesting with SimpleLoader (stochastic iteration) and Nestrov optimizer:")
    simple_loader = MiniBatch_Loader((X_train, Y_train), shuffle=True,batch_size=batch_size)
    sgdm_optimizer = NAG(learning_rate=0.001, neural_network=network,momentum=0.8, batch_size=batch_size)
    train_with_dataloader(simple_loader, network, sgdm_optimizer, criterion,epochs=epochs)

    # network.reset_network()
    
    # Test with Batch_Loader and Batch_GD optimizer
    # print("\nTesting with Batch_Loader (full-batch iteration) and Batch_GD optimizer:")
    # batch_loader = Batch_Loader((X_train, Y_train), shuffle=True)
    # batch_gd_optimizer = Batch_GD(learning_rate=0.01, neural_network=network, len_data=len(X_train))
    # train_with_dataloader(batch_loader, network, batch_gd_optimizer, criterion, epochs=10000)
    
    # Evaluate the trained network on the test data
    print("\nEvaluating on test data:")
    predictions = network.forward(X_test)
    predictions = np.array([1 if proba > 0.5 else 0 for proba in predictions])
    accuracy = np.mean(predictions == Y_test)
    print(f"Test Accuracy: {accuracy}")

test_with_different_dataset()
import numpy as np
from Layers import Dense
from DataLoader import SimpleLoader, Batch_Loader, MiniBatch_Loader
from Optimizers import SGD, MBGD, Batch_GD, SGDM
from Criterions import MSE
from NeuralNetwork import Network

def train_with_dataloader(dataloader, network, optimizer, criterion, epochs=1000):
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            predictions = network.forward(batch_x)  # Forward pass
            error = criterion.step_error(batch_y, predictions)  # Compute error
            
            # Backward pass
            d_error = criterion.step_d_error(batch_y, predictions)
            optimizer.step(d_error, step)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Error: {error}")
    
    # After training, evaluate the model's performance
    total_preds = []
    total_targets = []
    for batch_x, batch_y in dataloader:
        preds = network.forward(batch_x)
        total_preds.extend(preds)
        total_targets.extend(batch_y)
    
    # Calculate RMSE as evaluation metric
    rmse = np.sqrt(np.mean((np.array(total_preds) - np.array(total_targets)) ** 2))
    print(f"RMSE: {rmse}")

def test_regression():
    np.random.seed(42)
    
    # Generate synthetic regression data (sine wave with noise)
    X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    Y = np.sin(X) + 0.1 * np.random.randn(*X.shape)  # Sine wave with noise

    # Create layers and neural network
    hidden_layer_1 = Dense(input_size=1, num_neurons=10, activation=relu)
    hidden_layer_2 = Dense(input_size=10, num_neurons=5, activation=relu)
    output_layer = Dense(input_size=5, num_neurons=1, activation=linear)  # Linear activation for regression

    # Initialize the network and set layers
    network = Network()
    network.set_layers([hidden_layer_1, hidden_layer_2, output_layer])
    
    # Criterion (Loss function)
    criterion = MSE()

    # Test with MiniBatch_Loader and MBGD optimizer
    print("\nTesting regression with MiniBatch_Loader (mini-batch iteration) and MBGD optimizer:")
    minibatch_loader = MiniBatch_Loader((X, Y), batch_size=16, shuffle=True)
    mbgd_optimizer = MBGD(learning_rate=0.01, neural_network=network, batch_size=16)
    train_with_dataloader(minibatch_loader, network, mbgd_optimizer, criterion)
    
    network.reset_network()

    # Test with SimpleLoader and SGD optimizer
    print("\nTesting regression with SimpleLoader (stochastic iteration) and SGDM optimizer:")
    simple_loader = SimpleLoader((X, Y), shuffle=True)
    sgd_optimizer = SGDM(learning_rate=0.1, neural_network=network, batch_size=1)
    train_with_dataloader(simple_loader, network, sgd_optimizer, criterion)

    network.reset_network()
    
    # Test with Batch_Loader and Batch_GD optimizer
    print("\nTesting regression with Batch_Loader (full-batch iteration) and Batch_GD optimizer:")
    batch_loader = Batch_Loader((X, Y), shuffle=True)
    batch_gd_optimizer = Batch_GD(learning_rate=0.001, neural_network=network, len_data=len(X))
    train_with_dataloader(batch_loader, network, batch_gd_optimizer, criterion, epochs=1000)
    
# test_regression()

import numpy as np

from Activations import sigmoid,relu,linear
class Layer :
    def __init__(self,input_size,num_neurons,activation):
        self.activation,self.d_activation=activation()
        # data point size
        self.input_size=input_size
        #num of neurons
        self.num_neurons=num_neurons
        # matrix of random size=(num_neurons,input_size)
        self.weights=np.random.random(size=(num_neurons,input_size))
        # array of random size=(num_neurons)
        self.bias=np.random.random(size=(num_neurons))

    def reset_layer(self):
        # matrix of random size=(num_neurons,input_size)
        self.weights=np.random.random(size=(self.num_neurons,self.input_size))
        # array of random size=(num_neurons)
        self.bias=np.random.random(size=(self.num_neurons))
    def forward(self,x):
        pass
    def backward(self,dA):
        pass
class Dense(Layer) :
    def __init__(self,input_size,num_neurons,activation) -> None:
        super().__init__(input_size,num_neurons,activation)    
        
    def forward(self,batch_x):
        # assert len(x)== self.input_size
            
        self.batch_x=batch_x
        # print("weights :",self.weights)
        # print("batch x : ",self.batch_x)
        #pre-activation
        self.z=self.batch_x @ self.weights.T +self.bias # Shape: (batch_size, num_neurons)
        
        #post-activation
        output= self.activation(self.z)
        return output
    def backward(self,dA):
        """
        Backpropagate the error from the next layer (dA) to update the weights and biases.
        dA is the gradient of the loss with respect to the output activation (a) of next layer.
        """
        # gradient of loss with respect to the pre-activation

        dz = dA * self.d_activation(self.z)
        # gradient of loss with respect to weights
        dW =np.dot(dz.T ,self.batch_x)/len(self.batch_x)
        dB = np.sum(dz,axis=0)/len(self.batch_x)
        # the gradient with respect to the input of layer i ,
        # which is equivalent to the gradient with respect to output of all neurons in  layer i-1 ,so the previous layer need this to calculate its own gradient
        dX = np.dot(dz,self.weights)  
        
        # update weights
        # self.weights-=learning_rate*dW
        # self.bias -=learning_rate*dB
        return dX,dW,dB

if __name__=="__main__":
    X = [[1, 0], [0, 1], [1, 1], [0, 0]]
    Y = [1, 1, 0, 0]
    # an optimizer needs to know the neural network architecture , the num of iter ,learning rate ?
    #TO-DO : REFACTOR OPTIMIZER TO ITS OWN CLASS
def sgd():
    num_iter = 5000
    learning_rate=0.1
    hidden_layer_1 = Dense(input_size=2, num_neurons=3, activation=relu)
    hidden_layer_2 = Dense(input_size=3, num_neurons=4, activation=relu)
    output_layer = Dense(input_size=4, num_neurons=1,  activation=sigmoid)

    for epoch in range(num_iter):
        error = 0  # Accumulate error for monitoring purposes
        for y, x in zip(Y, X):
            # Forward pass
            out_h1 = hidden_layer_1.forward(x)
            out_h2 = hidden_layer_2.forward(out_h1)
            prediction = output_layer.forward(out_h2)

            # Calculate error for monitoring purposes
            error += 1/2 * (y - prediction) ** 2

            # Backpropagation through each layer
            d_error = (prediction - y)  # Derivative of the loss w.r.t output
            grad_out_layer = output_layer.backward(d_error,learning_rate=learning_rate)
            grad_hidden_layer_2 = hidden_layer_2.backward(grad_out_layer,learning_rate=learning_rate)
            grad_hidden_layer_1 = hidden_layer_1.backward(grad_hidden_layer_2,learning_rate=learning_rate)

        # Print error after each epoch
        # print(f"Error at iteration {epoch}: {error}")

    # Test prediction
    print("###### sgd prediction #######")
    for x in X :
        out_h1 = hidden_layer_1.forward(x)
        out_h2 = hidden_layer_2.forward(out_h1)
        prediction = output_layer.forward(out_h2)
        print(f"Prediction [{x}]probability: {prediction}")
        print(1 if prediction > 0.5 else 0)
def batch_gd():
    # batch gradient descent
    num_iter = 10000
    learning_rate=0.1
    hidden_layer_1 = Dense(input_size=2, num_neurons=3, activation=relu)
    hidden_layer_2 = Dense(input_size=3, num_neurons=4, activation=relu)
    output_layer = Dense(input_size=4, num_neurons=1,  activation=sigmoid)
    m=len(X)
    for _ in range(num_iter):
        error=0
        d_error=0
        for x,y in zip(X,Y):
            # forward
            h1=hidden_layer_1.forward(x)
            h2=hidden_layer_2.forward(h1)
            predition=output_layer.forward(h2)
            error+=1/2*(y-predition)**2
            d_error+=(predition-y)
        error /=m
        print(f"epoch {_} , error {error}")
        d_error /=m
        dA1=output_layer.backward(d_error,learning_rate=learning_rate)
        dA2=hidden_layer_2.backward(dA1,learning_rate=learning_rate)
        hidden_layer_1.backward(dA2,learning_rate=learning_rate)
     # Test prediction
    print("###### batch gd prediction #######")
    for x in X :
        out_h1 = hidden_layer_1.forward(x)
        out_h2= hidden_layer_2.forward(out_h1)
        prediction = output_layer.forward(out_h2)
        print(f"Prediction [{x}]probability: {prediction}")
        print(1 if prediction > 0.5 else 0)
if __name__=="__main__":
    batch_gd()
    sgd()